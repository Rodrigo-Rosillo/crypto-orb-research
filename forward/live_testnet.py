from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from backtester.risk import RiskLimits
from forward.artifacts import append_csv_rows, append_jsonl, build_signals_df
from forward.binance_live import (
    fetch_recent_klines_df,
    fetch_server_time_ms,
    interval_to_seconds,
)
from forward.data_service import DataService
from forward.shadow import build_signals
from forward.schemas import FILLS_COLUMNS, ORDERS_COLUMNS, SIGNALS_COLUMNS, validate_df_columns
from forward.state_store_sqlite import (
    SQLiteStateStore,
    migrate_state_json_to_sqlite,
)
from forward.testnet_broker import (
    BinanceFuturesTestnetBroker,
    RateLimitConfig,
    TestnetConfig,
    TestnetAPIError,
    TestnetAuthError,
)
from forward.trader_service import (
    TraderService,
    _extract_order_id,
    _order_avg_price,
)


async def _write_heartbeat_loop(
    heartbeat_path: Path,
    stop_event: asyncio.Event,
    interval_seconds: int = 60,
) -> None:
    """Write a UTC timestamp to heartbeat_path every interval_seconds.

    This allows the Docker healthcheck to verify the main loop is alive
    (healthcheck: find /data/heartbeat -mmin -10 -type f | grep -q .)
    Best-effort: errors are printed to stderr and never propagate.
    """
    while not stop_event.is_set():
        try:
            _write_heartbeat(heartbeat_path)
        except Exception as e:
            print(f"[heartbeat] write failed: {e}", file=sys.stderr)
        try:
            await asyncio.wait_for(
                asyncio.shield(stop_event.wait()),
                timeout=float(interval_seconds),
            )
        except asyncio.TimeoutError:
            pass


def _write_heartbeat(path: Path) -> None:
    """Best-effort heartbeat write target for Docker healthchecks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_rows(path, rows, columns, name: str) -> None:
    if rows:
        validate_df_columns(pd.DataFrame(rows), columns, name)
    else:
        validate_df_columns(pd.DataFrame(columns=columns), columns, name)
    append_csv_rows(path, rows, columns)


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        p = path.resolve()
        r = root.resolve()
        return p == r or r in p.parents
    except Exception:
        return False


def _should_cancel_on_exit(config_flag: bool, runtime_skip: bool) -> bool:
    return bool(config_flag) and not bool(runtime_skip)


async def run_live_testnet(
    *,
    run_dir,  # Path
    cfg: Dict[str, Any],
    ft_cfg: Dict[str, Any],
    risk_limits: Optional[RiskLimits],
    symbol: str,
    timeframe: str,
    initial_capital: float,
    position_size: float,
    taker_fee_rate: float,
    leverage: float,
    delay_bars: int,
    slippage_bps: float,
    max_bars: Optional[int] = None,
    duration_minutes: Optional[int] = None,
    smoke_test: bool = False,
    external_stop_event: Optional[asyncio.Event] = None,
) -> int:
    """Phase 5 Step 4: live market data + TESTNET order placement.

    This runner uses live market data ingestion (Step 3) but sends orders to Binance Futures TESTNET.

    Supports two workflows:
      - smoke_test=True: place a tiny entry+flatten cycle and exit.
      - normal run: wait for strategy signals and trade them on testnet.

    State is persisted to run_dir/state.db (SQLite) and mirrored to run_dir/state.json.
    """

    live_cfg = ft_cfg.get("live") if isinstance(ft_cfg.get("live"), dict) else {}
    market = str((live_cfg or {}).get("market", "futures"))
    bootstrap_limit = int((live_cfg or {}).get("bootstrap_limit", 1000))
    max_backoff_seconds = int((live_cfg or {}).get("max_backoff_seconds", 60))
    heartbeat_seconds = int((live_cfg or {}).get("heartbeat_seconds", 120))
    stale_check_interval_seconds = int((live_cfg or {}).get("stale_check_interval_seconds", 30))

    testnet_cfg = ft_cfg.get("testnet") if isinstance(ft_cfg.get("testnet"), dict) else {}
    base_url = str((testnet_cfg or {}).get("base_url", "https://demo-fapi.binance.com"))
    if not base_url.lower().startswith("https://"): 
        raise ValueError("forward_test.testnet.base_url must start with https://")
    recv_window_ms = int((testnet_cfg or {}).get("recv_window_ms", 5000))
    poll_interval_seconds = float((testnet_cfg or {}).get("poll_interval_seconds", 2.0))
    cancel_open_orders_on_exit = bool((testnet_cfg or {}).get("cancel_open_orders_on_exit", True))
    flatten_on_mismatch = bool((testnet_cfg or {}).get("flatten_on_mismatch", False))
    # Smoke test settings
    smoke_qty = float((testnet_cfg or {}).get("smoke_qty", 0.1))
    smoke_auto_flatten = bool((testnet_cfg or {}).get("smoke_auto_flatten", True))

    # Rate limit tuning
    rl = RateLimitConfig(
        max_retries=int((testnet_cfg or {}).get("max_retries", 6)),
        base_backoff_seconds=float((testnet_cfg or {}).get("base_backoff_seconds", 0.5)),
        max_backoff_seconds=float((testnet_cfg or {}).get("max_backoff_seconds", 20.0)),
    )

    bar_seconds = interval_to_seconds(timeframe)

    # Output files
    signals_path = run_dir / "signals.csv"
    orders_path = run_dir / "orders.csv"
    fills_path = run_dir / "fills.csv"
    positions_path = run_dir / "positions.csv"
    events_path = run_dir / "events.jsonl"
    state_path = run_dir / "state.json"
    # run_dir is timestamped per run; to resume state across restarts, the caller
    # must pass the same run_dir (e.g., via scripts/forward_test.py --run-dir).
    db_path_raw = os.environ.get("STATE_DB_PATH") or ft_cfg.get("state_db_path") or (run_dir / "state.db")
    db_path = Path(str(db_path_raw))
    if not db_path.is_absolute():
        db_path = run_dir / db_path

    heartbeat_path = Path(os.environ.get("HEARTBEAT_PATH", "/data/heartbeat"))

    # Allow the CLI runner to inject an external stop_event (e.g., for graceful
    # Ctrl+C handling). If not provided, create an internal event.
    stop_event = external_stop_event or asyncio.Event()

    # Bootstrap history
    rest_df, rest_meta = fetch_recent_klines_df(
        symbol=symbol,
        interval=timeframe,
        limit=bootstrap_limit,
        market=market,
    )
    df_raw = rest_df.copy()

    # --- Bootstrap validation (freshness + local clock sanity) ---
    default_age_bars = int(risk_limits.kill_switch.max_data_gap_bars) if risk_limits is not None else 2
    bam = (live_cfg or {}).get("bootstrap_max_age_bars", default_age_bars)
    if bam is None:
        bam = default_age_bars
    bootstrap_max_age_bars = max(1, int(bam))
    allowed_bootstrap_age_s = float(bar_seconds * bootstrap_max_age_bars)

    last_close_iso = str((rest_meta or {}).get("last_close_time", "") or "")
    last_close_ts: Optional[pd.Timestamp] = None
    if last_close_iso:
        try:
            last_close_ts = pd.to_datetime(last_close_iso, utc=True)
        except Exception:
            last_close_ts = None
    if last_close_ts is None and len(df_raw.index):
        last_close_ts = df_raw.index[-1] + pd.Timedelta(seconds=int(bar_seconds))

    now_utc = datetime.now(timezone.utc)
    bootstrap_age_s = None
    if last_close_ts is not None:
        bootstrap_age_s = float((now_utc - last_close_ts.to_pydatetime()).total_seconds())

    # Local clock skew check vs Binance server time
    clock_skew_ms = None
    clock_meta: Dict[str, Any] = {}
    try:
        server_ms, clock_meta = fetch_server_time_ms(market=market)
        local_ms = int(now_utc.timestamp() * 1000)
        clock_skew_ms = int(local_ms - int(server_ms))
    except Exception as e:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "CLOCK_SKEW_CHECK_FAILED", "error": str(e)}])

    trading_start_ts: Optional[pd.Timestamp] = df_raw.index[-1] if len(df_raw.index) else None

    append_jsonl(
        events_path,
        [
            {
                "ts": _utcnow_iso(),
                "type": "LIVE_RUN_START",
                "mode": "testnet",
                "source": "live",
                "symbol": symbol,
                "timeframe": timeframe,
                "market": market,
                "bootstrap": rest_meta,
                "trading_start_ts": trading_start_ts.isoformat() if trading_start_ts is not None else "",
            }
        ],
    )

    # Clock skew thresholds
    skew_warn_ms = 5_000
    skew_fatal_ms = 30_000
    if clock_skew_ms is not None:
        if abs(clock_skew_ms) > skew_fatal_ms:
            append_jsonl(
                events_path,
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "CLOCK_SKEW_FATAL",
                        "clock_skew_ms": int(clock_skew_ms),
                        "warn_ms": int(skew_warn_ms),
                        "fatal_ms": int(skew_fatal_ms),
                        "server_time": clock_meta,
                    }
                ],
            )
            append_jsonl(
                events_path,
                [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "CLOCK_SKEW_FATAL"}],
            )
            return 0
        if abs(clock_skew_ms) > skew_warn_ms:
            append_jsonl(
                events_path,
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "CLOCK_SKEW_WARN",
                        "clock_skew_ms": int(clock_skew_ms),
                        "warn_ms": int(skew_warn_ms),
                        "fatal_ms": int(skew_fatal_ms),
                        "server_time": clock_meta,
                    }
                ],
            )

    if last_close_ts is None:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "BOOTSTRAP_EMPTY_OR_INVALID", "bootstrap_rows": int(len(df_raw.index))}])
        append_jsonl(
            events_path,
            [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "BOOTSTRAP_EMPTY_OR_INVALID"}],
        )
        return 0

    append_jsonl(
        events_path,
        [
            {
                "ts": _utcnow_iso(),
                "type": "BOOTSTRAP_VALIDATION",
                "last_close_time": last_close_ts.isoformat(),
                "age_seconds": float(bootstrap_age_s) if bootstrap_age_s is not None else None,
                "allowed_seconds": float(allowed_bootstrap_age_s),
                "bootstrap_max_age_bars": int(bootstrap_max_age_bars),
                "bar_seconds": int(bar_seconds),
            }
        ],
    )

    if bootstrap_age_s is not None and bootstrap_age_s > allowed_bootstrap_age_s:
        append_jsonl(
            events_path,
            [{"ts": _utcnow_iso(), "type": "BOOTSTRAP_STALE", "age_seconds": float(bootstrap_age_s), "allowed_seconds": float(allowed_bootstrap_age_s), "bootstrap_max_age_bars": int(bootstrap_max_age_bars)}],
        )
        append_jsonl(
            events_path,
            [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "BOOTSTRAP_STALE"}],
        )
        return 0
    else:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "BOOTSTRAP_OK", "age_seconds": float(bootstrap_age_s) if bootstrap_age_s is not None else None}])

    # Create broker
    try:
        broker = BinanceFuturesTestnetBroker(cfg=TestnetConfig(base_url=base_url, recv_window_ms=recv_window_ms, rate_limit=rl))
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_BROKER_CONFIG", "base_url": base_url, "recv_window_ms": int(recv_window_ms)}])
    except TestnetAuthError as e:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_AUTH_ERROR", "error": str(e)}])
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "TESTNET_AUTH_ERROR"}])
        return 0

    # Set leverage and abort startup if exchange confirmation fails.
    requested_leverage = int(leverage)
    try:
        leverage_resp = broker.set_leverage(symbol, requested_leverage)
        raw = (leverage_resp or {}).get("leverage")
        if raw is None:
            raise RuntimeError(f"leverage key missing from set_leverage response: {leverage_resp!r}")
        confirmed_leverage = int(raw)
        if confirmed_leverage != requested_leverage:
            raise RuntimeError(
                f"Leverage mismatch after set_leverage: requested={requested_leverage}, confirmed={confirmed_leverage}"
            )
        append_jsonl(
            events_path,
            [{"ts": _utcnow_iso(), "type": "TESTNET_SET_LEVERAGE", "leverage": int(confirmed_leverage)}],
        )
    except Exception as e:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_SET_LEVERAGE_FAILED", "error": str(e)}])
        append_jsonl(
            events_path,
            [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "SET_LEVERAGE_FAILED"}],
        )
        return 0

    had_state_db = bool(db_path.exists())
    had_state_json = bool(state_path.exists())
    migrated_from_json = False
    if (not had_state_db) and had_state_json:
        migrate_state_json_to_sqlite(db_path=db_path, json_path=state_path, events_path=events_path, force=False)
        migrated_from_json = True
        append_jsonl(
            events_path,
            [{"ts": _utcnow_iso(), "type": "MIGRATION_COMPLETE", "db_path": str(db_path), "json_path": str(state_path)}],
        )

    with SQLiteStateStore(db_path=db_path, events_path=events_path) as store:
        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "STATE_DB_OPEN", "db_path": str(db_path)}])
        if not _is_path_within(db_path, run_dir):
            append_jsonl(
                events_path,
                [{"ts": _utcnow_iso(), "type": "STATE_DB_PATH_EXTERNAL", "db_path": str(db_path), "run_dir": str(run_dir)}],
            )

        # Load and reconcile state
        had_prior_state = bool(had_state_db or migrated_from_json or had_state_json)
        state = store.load_state()
        append_jsonl(
            events_path,
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "STATE_LOADED",
                    "path": str(db_path),
                    "found": bool(had_prior_state),
                    "bars_processed": int(state.bars_processed or 0),
                    "last_bar_open_time_utc": str(state.last_bar_open_time_utc or ""),
                }
            ],
        )

        # Risk kill-switch thresholds (apply even if limits.enabled is False; these are operational safety checks)
        max_order_rejects_per_day = int(risk_limits.kill_switch.max_order_rejects_per_day) if risk_limits is not None else 3
        max_margin_ratio = float(risk_limits.kill_switch.max_margin_ratio) if risk_limits is not None else 0.85

        emit_event = lambda rows: append_jsonl(events_path, rows)

        trader_service = TraderService(
            broker=broker,
            store=store,
            state=state,
            symbol=symbol,
            leverage=float(int(leverage)),
            position_size=float(position_size),
            initial_capital=float(initial_capital),
            slippage_bps=float(slippage_bps),
            taker_fee_rate=float(taker_fee_rate),
            state_path=state_path,
            events_path=events_path,
            run_id=str(run_dir.name),
            stop_event=stop_event,
            risk_limits=risk_limits,
            max_order_rejects_per_day=int(max_order_rejects_per_day),
            margin_ratio_threshold=float(max_margin_ratio),
            orders_path=orders_path,
            fills_path=fills_path,
            positions_path=positions_path,
            append_rows=_append_rows,
            emit_event=emit_event,
        )

        def _emit_runtime_event(event_type: str, *, stage: Optional[str] = None, **fields: Any) -> None:
            event: Dict[str, Any] = {"ts": _utcnow_iso(), "type": str(event_type)}
            if stage is not None:
                event["stage"] = str(stage)
            event.update(fields)
            append_jsonl(events_path, [event])

        def _persist_state_best_effort() -> None:
            try:
                trader_service.persist_state()
            except Exception as e:
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "STATE_SAVE_FAILED", "error": str(e)}])

        def _handle_reconciliation_result(
            result: dict[str, Any],
            *,
            stage: Optional[str] = None,
            persist_state_on_stop: bool = False,
        ) -> Optional[str]:
            if str(result.get("status") or "") != "mismatch":
                return None

            payload = dict(result.get("payload") or {})
            _emit_runtime_event("RECON_MISMATCH", stage=stage, **payload)

            if bool(flatten_on_mismatch) and bool(result.get("flatten_on_mismatch")):
                raw_snapshot = result.get("snapshot")
                if isinstance(raw_snapshot, dict):
                    ex_side = str(raw_snapshot.get("side") or "")
                    ex_qty = float(raw_snapshot.get("qty") or 0.0)
                else:
                    ex_side = ""
                    ex_qty = 0.0
                try:
                    side = "SELL" if ex_side == "LONG" else "BUY"
                    broker.place_market_order(symbol=symbol, side=side, quantity=ex_qty, reduce_only=True)
                    _emit_runtime_event("RECON_FLATTENED", stage=stage)
                except Exception as e:
                    _emit_runtime_event("RECON_FLATTEN_FAILED", stage=stage, error=str(e))

            if persist_state_on_stop:
                _persist_state_best_effort()
            return "RECON_MISMATCH"

        # Persist an initial state snapshot immediately so state.json exists even if the
        # run stops before the first closed candle arrives (common for short duration runs).
        if state.last_bar_open_time_utc is None and trading_start_ts is not None:
            try:
                state.last_bar_open_time_utc = trading_start_ts.tz_convert("UTC").isoformat()
            except Exception:
                state.last_bar_open_time_utc = str(trading_start_ts)
        try:
            trader_service.persist_state()
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "STATE_SAVED", "path": "state.json"}])
        except Exception as e:
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "STATE_SAVE_FAILED", "error": str(e)}])

        try:
            ex_side, ex_qty, ex_entry, ex_upl = trader_service.fetch_exchange_position()
        except TestnetAPIError as e:
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_POSITION_RISK_FAILED", "error": str(e), "payload": getattr(e, "payload", None)}])
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "TESTNET_POSITION_RISK_FAILED"}])
            return 0

        startup_reconciliation = trader_service.classify_exchange_position_reconciliation((ex_side, ex_qty, ex_entry, ex_upl))
        startup_reason = _handle_reconciliation_result(startup_reconciliation)
        if startup_reason is not None:
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": startup_reason}])
            return 0

        # Smoke test: place tiny order and optionally flatten
        if smoke_test:
            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_SMOKE_START", "qty": float(smoke_qty)}])
            try:
                entry_resp = broker.place_market_order(symbol=symbol, side="SELL", quantity=float(smoke_qty), reduce_only=False)
                entry_oid = _extract_order_id(entry_resp)
                fill_price = _order_avg_price(entry_resp)
                ex_side2, ex_qty2, ex_entry2, _ = trader_service.fetch_exchange_position()
                append_jsonl(
                    events_path,
                    [{"ts": _utcnow_iso(), "type": "TESTNET_SMOKE_ENTRY", "order_id": entry_oid, "fill_price": float(fill_price), "exchange_side": ex_side2, "exchange_qty": float(ex_qty2), "exchange_entry": float(ex_entry2)}],
                )
                _append_rows(
                    orders_path,
                    [{"timestamp_utc": _utcnow_iso(), "due_timestamp_utc": "", "order_id": str(entry_oid or ""), "symbol": symbol, "side": "SHORT", "qty": float(smoke_qty), "order_type": "MARKET", "limit_price": "", "status": "sent", "status_detail": "smoke_entry", "reason": ""}],
                    ORDERS_COLUMNS,
                    "orders.csv",
                )
                _append_rows(
                    fills_path,
                    [{"timestamp_utc": _utcnow_iso(), "order_id": str(entry_oid or ""), "symbol": symbol, "side": "SHORT", "qty": float(smoke_qty), "fill_price": float(fill_price), "fee": 0.0, "slippage_bps": float(slippage_bps), "exec_model": "testnet_market"}],
                    FILLS_COLUMNS,
                    "fills.csv",
                )

                if smoke_auto_flatten and ex_side2 != "FLAT":
                    flat_resp = broker.place_market_order(symbol=symbol, side="BUY", quantity=float(ex_qty2), reduce_only=True)
                    flat_oid = _extract_order_id(flat_resp)
                    flat_price = _order_avg_price(flat_resp)
                    append_jsonl(
                        events_path,
                        [{"ts": _utcnow_iso(), "type": "TESTNET_SMOKE_FLATTEN", "order_id": flat_oid, "fill_price": float(flat_price), "qty": float(ex_qty2)}],
                    )
                    _append_rows(
                        orders_path,
                        [{"timestamp_utc": _utcnow_iso(), "due_timestamp_utc": "", "order_id": str(flat_oid or ""), "symbol": symbol, "side": "LONG", "qty": float(ex_qty2), "order_type": "MARKET", "limit_price": "", "status": "sent", "status_detail": "smoke_flatten", "reason": ""}],
                        ORDERS_COLUMNS,
                        "orders.csv",
                    )
                    _append_rows(
                        fills_path,
                        [{"timestamp_utc": _utcnow_iso(), "order_id": str(flat_oid or ""), "symbol": symbol, "side": "LONG", "qty": float(ex_qty2), "fill_price": float(flat_price), "fee": 0.0, "slippage_bps": float(slippage_bps), "exec_model": "testnet_market_reduce_only"}],
                        FILLS_COLUMNS,
                        "fills.csv",
                    )
            except Exception as e:
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "TESTNET_SMOKE_FAILED", "error": str(e)}])

            try:
                state.open_position = None
                trader_service.persist_state()
            except Exception:
                pass

            append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": 0, "final_equity": float(initial_capital), "reason": "SMOKE_TEST"}])
            return 0

        gap_bars = int(risk_limits.kill_switch.max_data_gap_bars) if risk_limits is not None else 2

        def on_data_kill_switch(reason: str, bar_time_utc: str) -> None:
            store.append_trade_log(
                event_type="KILL_SWITCH",
                symbol=symbol,
                side=None,
                qty=None,
                price=None,
                realized_pnl=None,
                fee=None,
                funding_applied=None,
                reason=reason,
                bar_time_utc=bar_time_utc,
            )

        data_service = DataService(
            symbol=symbol,
            interval=timeframe,
            market=market,
            stale_allowed_seconds=float(bar_seconds * gap_bars),
            max_backoff_seconds=int(max_backoff_seconds),
            stale_check_interval_seconds=int(stale_check_interval_seconds),
            heartbeat_seconds=int(heartbeat_seconds),
            emit_event=emit_event,
            on_kill_switch=on_data_kill_switch,
        )

        last_connect_count = data_service.connect_count

        try:
            data_service.last_closed_bar_at = last_close_ts.to_pydatetime()
        except Exception:
            data_service.last_closed_bar_at = datetime.now(timezone.utc)

        last_bar_open_ts: Optional[pd.Timestamp] = None
        if state.last_bar_open_time_utc:
            try:
                last_bar_open_ts = pd.to_datetime(state.last_bar_open_time_utc, utc=True)
            except Exception:
                last_bar_open_ts = trading_start_ts
        else:
            last_bar_open_ts = trading_start_ts

        bars_processed = int(state.bars_processed or 0)
        start_wall = datetime.now(timezone.utc)
        stop_requested = False
        run_end_reason: Optional[str] = None

        def _persist_bar_progress() -> None:
            nonlocal bars_processed
            bars_processed += 1
            state.bars_processed = int(bars_processed)
            _persist_state_best_effort()
            try:
                _write_heartbeat(heartbeat_path)
            except Exception as e:
                print(f"[heartbeat] write failed: {e}", file=sys.stderr)

        hb = asyncio.create_task(data_service.heartbeat_task(stop_event))
        hb_writer = asyncio.create_task(
            _write_heartbeat_loop(heartbeat_path, stop_event)
        )

        try:
            async for bar in data_service.stream_closed(stop_event):
                if stop_event.is_set():
                    continue

                if data_service.connect_count != last_connect_count:
                    append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "WS_CONNECTED", "connect_count": int(data_service.connect_count)}])
                    last_connect_count = data_service.connect_count

                data_service.last_closed_bar_at = datetime.now(timezone.utc)

                if last_bar_open_ts is not None and bar.open_time <= last_bar_open_ts:
                    append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "BAR_DUPLICATE_IGNORED", "open_time": bar.open_time.isoformat()}])
                    continue

                last_bar_open_ts = bar.open_time
                state.last_bar_open_time_utc = bar.open_time.tz_convert("UTC").isoformat()

                day_str = bar.open_time.tz_convert("UTC").date().isoformat()
                if state.current_day_utc != day_str:
                    state.current_day_utc = day_str
                    state.order_rejects_today = 0
                    append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "DAY_START", "message": "New UTC day", "day": day_str}])

                append_jsonl(
                    events_path,
                    [
                        {
                            "ts": _utcnow_iso(),
                            "type": "BAR_CLOSED",
                            "open_time": bar.open_time.isoformat(),
                            "close_time": bar.close_time.isoformat(),
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": float(bar.volume),
                        }
                    ],
                )

                df_raw.loc[bar.open_time] = bar.to_row()
                df_raw = df_raw.sort_index()

                valid_days = set(df_raw.index.date)
                df_sig, _, _ = build_signals(
                    df_raw=df_raw,
                    valid_days=valid_days,
                    cfg=cfg,
                )
                if bar.open_time not in df_sig.index:
                    continue
                row = df_sig.loc[bar.open_time]

                if int(row.get("signal", 0) or 0) != 0:
                    sig_df = build_signals_df(df_sig.loc[[bar.open_time]], symbol=symbol)
                    sig_rows = sig_df.to_dict(orient="records")
                    _append_rows(signals_path, sig_rows, SIGNALS_COLUMNS, "signals.csv")

                if trading_start_ts is not None and bar.open_time <= trading_start_ts:
                    continue

                await trader_service.poll_open_orders()
                try:
                    exchange_position = trader_service.fetch_exchange_position()
                except Exception as e:
                    _emit_runtime_event(
                        "TESTNET_POSITION_RISK_FAILED",
                        stage="runtime",
                        error=str(e),
                        payload=getattr(e, "payload", None),
                    )
                    run_end_reason = "TESTNET_POSITION_RISK_FAILED"
                    _persist_bar_progress()
                    break
                trader_service.record_position_snapshot(exchange_position)
                runtime_reconciliation = trader_service.classify_exchange_position_reconciliation(exchange_position)
                run_end_reason = _handle_reconciliation_result(
                    runtime_reconciliation,
                    stage="runtime",
                    persist_state_on_stop=False,
                )
                if run_end_reason is not None:
                    _persist_bar_progress()
                    break
                trader_service.maybe_kill_on_margin_ratio()
                if stop_event.is_set():
                    _persist_bar_progress()
                    break

                await trader_service.maybe_place_trade_from_signal(bar.open_time, row)

                _persist_bar_progress()

                if stop_requested:
                    continue
                if max_bars is not None and bars_processed >= int(max_bars):
                    append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "STOP_MAX_BARS", "max_bars": int(max_bars)}])
                    stop_event.set()
                    stop_requested = True
                    continue

                if duration_minutes is not None:
                    elapsed = (datetime.now(timezone.utc) - start_wall).total_seconds() / 60.0
                    if elapsed >= float(duration_minutes):
                        append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "STOP_DURATION", "duration_minutes": int(duration_minutes)}])
                        stop_event.set()
                        stop_requested = True
                        continue

                await asyncio.sleep(max(0.0, float(poll_interval_seconds)))

        finally:
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

            hb_writer.cancel()
            try:
                await hb_writer
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

            open_position = trader_service.state.open_position
            if open_position is not None:
                try:
                    flatten_ok, flatten_detail = trader_service._emergency_flatten(reason="SHUTDOWN_GUARD")
                    if flatten_ok:
                        trader_service.state.open_position = None
                        trader_service.skip_cancel_open_orders_on_exit_runtime = False
                        try:
                            trader_service.persist_state()
                        except Exception:
                            pass
                    else:
                        trader_service.skip_cancel_open_orders_on_exit_runtime = True
                        append_jsonl(
                            events_path,
                            [{"ts": _utcnow_iso(), "type": "SHUTDOWN_GUARD_FLATTEN_FAILED", "detail": str(flatten_detail)}],
                        )
                except Exception as e:
                    trader_service.skip_cancel_open_orders_on_exit_runtime = True
                    append_jsonl(
                        events_path,
                        [{"ts": _utcnow_iso(), "type": "SHUTDOWN_GUARD_FLATTEN_ERROR", "error": str(e)}],
                    )
                    try:
                        ex_side, ex_qty, _, _ = trader_service.fetch_exchange_position()
                        if ex_side == "FLAT" or ex_qty < 1e-9:
                            trader_service.state.open_position = None
                            trader_service.skip_cancel_open_orders_on_exit_runtime = False
                            try:
                                trader_service.persist_state()
                            except Exception:
                                pass
                    except Exception:
                        pass

            runtime_skip = bool(getattr(trader_service, "skip_cancel_open_orders_on_exit_runtime", False))
            if _should_cancel_on_exit(cancel_open_orders_on_exit, runtime_skip):
                try:
                    broker.cancel_all_open_orders(symbol=symbol)
                    append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "CANCEL_ALL_OPEN_ORDERS"}])
                except Exception as e:
                    append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "CANCEL_ALL_OPEN_ORDERS_FAILED", "error": str(e)}])
            elif bool(cancel_open_orders_on_exit) and runtime_skip:
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "CANCEL_ALL_OPEN_ORDERS_SKIPPED_RUNTIME_GUARD"}])

            append_jsonl(
                events_path,
                [{"ts": _utcnow_iso(), "type": "LIVE_RUN_END", "bars_processed": int(bars_processed), "final_equity": "", "reason": str(run_end_reason or ("STOP" if stop_event.is_set() else "END"))}],
            )

        return 0


from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from backtester.futures_engine import FuturesEngineConfig
from backtester.risk import RiskLimits
from forward.artifacts import append_csv_rows, append_jsonl, build_signals_df
from forward.binance_live import (
    BinanceLiveKlineSource,
    fetch_recent_klines_df,
    fetch_server_time_ms,
    interval_to_seconds,
)
from forward.shadow import build_signals
from forward.schemas import FILLS_COLUMNS, ORDERS_COLUMNS, POSITIONS_COLUMNS, SIGNALS_COLUMNS, validate_df_columns
from forward.stream_engine import StreamingFuturesShadowEngine


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_rows(path, rows, columns, name: str) -> None:
    if rows:
        validate_df_columns(pd.DataFrame(rows), columns, name)
    else:
        validate_df_columns(pd.DataFrame(columns=columns), columns, name)
    append_csv_rows(path, rows, columns)


async def run_live_shadow(
    *,
    run_dir,  # Path
    cfg: Dict[str, Any],
    ft_cfg: Dict[str, Any],
    risk_limits: Optional[RiskLimits],
    symbol: str,
    timeframe: str,
    orb_start,
    orb_end,
    orb_cutoff,
    adx_period: int,
    adx_threshold: float,
    initial_capital: float,
    position_size: float,
    taker_fee_rate: float,
    leverage: float,
    delay_bars: int,
    slippage_bps: float,
    funding_rate_per_8h: float = 0.0,
    max_bars: Optional[int] = None,
    duration_minutes: Optional[int] = None,
    external_stop_event: Optional[asyncio.Event] = None,
) -> int:
    """Phase 5 Step 3: live market data ingestion + shadow execution.

    This runner:
      - bootstraps a recent history window via REST
      - streams CLOSED 30m candles via WebSocket
      - ensures no duplicate candle processing
      - reconnects on network failures
      - runs a stateful shadow engine and writes forward-test artifacts incrementally
      - triggers a kill switch if candles go stale
    """

    live_cfg = ft_cfg.get("live") if isinstance(ft_cfg.get("live"), dict) else {}
    market = str((live_cfg or {}).get("market", "futures"))
    bootstrap_limit = int((live_cfg or {}).get("bootstrap_limit", 1000))
    max_backoff_seconds = int((live_cfg or {}).get("max_backoff_seconds", 60))
    heartbeat_seconds = int((live_cfg or {}).get("heartbeat_seconds", 120))
    stale_check_interval_seconds = int((live_cfg or {}).get("stale_check_interval_seconds", 30))

    bar_seconds = interval_to_seconds(timeframe)

    # Output files
    signals_path = run_dir / "signals.csv"
    orders_path = run_dir / "orders.csv"
    fills_path = run_dir / "fills.csv"
    positions_path = run_dir / "positions.csv"
    events_path = run_dir / "events.jsonl"

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
    # If bootstrap_max_age_bars not set, default to the kill-switch max_data_gap_bars (even if risk is disabled).
    default_age_bars = int(risk_limits.kill_switch.max_data_gap_bars) if risk_limits is not None else 2
    bam = (live_cfg or {}).get("bootstrap_max_age_bars", default_age_bars)
    if bam is None:
        bam = default_age_bars
    bootstrap_max_age_bars = int(bam)
    bootstrap_max_age_bars = max(1, bootstrap_max_age_bars)
    allowed_bootstrap_age_s = float(bar_seconds * bootstrap_max_age_bars)

    # Parse last closed candle time from metadata if available.
    last_close_iso = str((rest_meta or {}).get("last_close_time", "") or "")
    last_close_ts: Optional[pd.Timestamp] = None
    if last_close_iso:
        try:
            last_close_ts = pd.to_datetime(last_close_iso, utc=True)
        except Exception:
            last_close_ts = None
    if last_close_ts is None and len(df_raw.index):
        # Fallback: approximate close as open + interval.
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
        append_jsonl(
            events_path,
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "CLOCK_SKEW_CHECK_FAILED",
                    "error": str(e),
                }
            ],
        )


    # Marker: we only start trading AFTER the latest bootstrapped candle.
    trading_start_ts: Optional[pd.Timestamp] = None
    if len(df_raw.index):
        trading_start_ts = df_raw.index[-1]

    append_jsonl(
        events_path,
        [
            {
                "ts": _utcnow_iso(),
                "type": "LIVE_RUN_START",
                "mode": "shadow",
                "source": "live",
                "symbol": symbol,
                "timeframe": timeframe,
                "market": market,
                "bootstrap": rest_meta,
                "trading_start_ts": trading_start_ts.isoformat() if trading_start_ts is not None else "",
            }
        ],
    )

    # Emit bootstrap freshness + clock skew signals before streaming.
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
            # Clean shutdown: don't start WS/heartbeat.
            append_jsonl(
                events_path,
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "LIVE_RUN_END",
                        "bars_processed": 0,
                        "final_equity": float(initial_capital),
                        "reason": "CLOCK_SKEW_FATAL",
                    }
                ],
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
        append_jsonl(
            events_path,
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "BOOTSTRAP_EMPTY_OR_INVALID",
                    "bootstrap_rows": int(len(df_raw.index)),
                }
            ],
        )
        append_jsonl(
            events_path,
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "LIVE_RUN_END",
                    "bars_processed": 0,
                    "final_equity": float(initial_capital),
                    "reason": "BOOTSTRAP_EMPTY_OR_INVALID",
                }
            ],
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
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "BOOTSTRAP_STALE",
                    "age_seconds": float(bootstrap_age_s),
                    "allowed_seconds": float(allowed_bootstrap_age_s),
                    "bootstrap_max_age_bars": int(bootstrap_max_age_bars),
                }
            ],
        )
        append_jsonl(
            events_path,
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "LIVE_RUN_END",
                    "bars_processed": 0,
                    "final_equity": float(initial_capital),
                    "reason": "BOOTSTRAP_STALE",
                }
            ],
        )
        return 0
    else:
        append_jsonl(
            events_path,
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "BOOTSTRAP_OK",
                    "age_seconds": float(bootstrap_age_s) if bootstrap_age_s is not None else None,
                }
            ],
        )

    engine_cfg = FuturesEngineConfig(
        initial_capital=float(initial_capital),
        position_size=float(position_size),
        leverage=float(leverage),
        taker_fee_rate=float(taker_fee_rate),
        slippage_bps=float(slippage_bps),
        delay_bars=int(delay_bars),
        funding_rate_per_8h=float(funding_rate_per_8h),
    )
    engine = StreamingFuturesShadowEngine(engine_cfg, risk_limits=risk_limits, expected_bar_seconds=bar_seconds)

    # Track last WS message time (heartbeat) + last CLOSED bar time (staleness)
    # Seed "last_closed_bar_at" from the bootstrapped last closed candle, so stale detection
    # reflects data freshness (not program start time).
    try:
        last_closed_bar_at = last_close_ts.to_pydatetime() if last_close_ts is not None else datetime.now(timezone.utc)
    except Exception:
        last_closed_bar_at = datetime.now(timezone.utc)
    last_bar_open_ts: Optional[pd.Timestamp] = trading_start_ts

    # Trade/order ids (stable + readable)
    trade_counter = 0

    async def heartbeat_task() -> None:
        nonlocal last_closed_bar_at
        while not stop_event.is_set():
            await asyncio.sleep(max(1, stale_check_interval_seconds))
            if stop_event.is_set():
                break

            now = datetime.now(timezone.utc)

            # Heartbeat is based on websocket message activity (not closed bars)
            last_msg = src.last_message_at or src.last_connect_at
            if last_msg is not None:
                since_msg = (now - last_msg).total_seconds()
                if since_msg >= heartbeat_seconds:
                    append_jsonl(
                        events_path,
                        [
                            {
                                "ts": _utcnow_iso(),
                                "type": "DATA_HEARTBEAT_MISSED",
                                "since_seconds": float(since_msg),
                                "threshold_seconds": int(heartbeat_seconds),
                            }
                        ],
                    )

            # Staleness kill-switch is based on CLOSED bars not arriving
            since_closed = (now - last_closed_bar_at).total_seconds()
            # Data staleness is operational safety; apply kill-switch config even if risk limits are disabled.
            gap_bars = int(risk_limits.kill_switch.max_data_gap_bars) if risk_limits is not None else 2
            allowed = float(bar_seconds * gap_bars)
            if since_closed > allowed:
                append_jsonl(
                    events_path,
                    [
                        {
                            "ts": _utcnow_iso(),
                            "type": "KILL_SWITCH_DATA_STALE",
                            "since_seconds": float(since_closed),
                            "allowed_seconds": float(allowed),
                        }
                    ],
                )
                stop_event.set()
                stop_requested = True
                continue

    src = BinanceLiveKlineSource(
        symbol=symbol,
        interval=timeframe,
        market=market,
        max_backoff_seconds=max_backoff_seconds,
    )

    last_connect_count = src.connect_count

    hb = asyncio.create_task(heartbeat_task())

    bars_processed = 0
    start_wall = datetime.now(timezone.utc)
    stop_requested = False

    try:
        async for bar in src.stream_closed(stop_event=stop_event):
            # If a stop was requested, ignore any late-arriving buffered bars;
            # the stream should terminate promptly once stop_event is set.
            if stop_event.is_set():
                continue

            # Log (re)connects
            if src.connect_count != last_connect_count:
                append_jsonl(
                    events_path,
                    [
                        {
                            "ts": _utcnow_iso(),
                            "type": "WS_CONNECTED",
                            "connect_count": int(src.connect_count),
                        }
                    ],
                )
                last_connect_count = src.connect_count

            last_closed_bar_at = datetime.now(timezone.utc)

            # De-dupe (stream already ignores <= last open_time, but we log if it happens)
            if last_bar_open_ts is not None and bar.open_time <= last_bar_open_ts:
                append_jsonl(
                    events_path,
                    [
                        {
                            "ts": _utcnow_iso(),
                            "type": "BAR_DUPLICATE_IGNORED",
                            "open_time": bar.open_time.isoformat(),
                        }
                    ],
                )
                continue

            last_bar_open_ts = bar.open_time

            append_jsonl(
                events_path,
                [
                    {
                        "ts": _utcnow_iso(),
                        "type": "BAR_CLOSED",
                        "open_time": bar.open_time.isoformat(),
                        "close_time": bar.close_time.isoformat(),
                        # Full OHLCV so Step 5 divergence report can compare to reference parquet.
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                    }
                ],
            )

            # Update df_raw
            df_raw.loc[bar.open_time] = bar.to_row()
            df_raw = df_raw.sort_index()

            # Recompute signals on the current history window (slow but fine for 30m cadence)
            valid_days = set(df_raw.index.date)
            df_sig, orb_ranges = build_signals(
                df_raw=df_raw,
                valid_days=valid_days,
                orb_start=orb_start,
                orb_end=orb_end,
                orb_cutoff=orb_cutoff,
                adx_period=adx_period,
                adx_threshold=adx_threshold,
            )

            if bar.open_time not in df_sig.index:
                continue

            row = df_sig.loc[bar.open_time]
            signal = int(row.get("signal", 0) or 0)
            signal_type = str(row.get("signal_type", "") or "")
            day = row.get("date")

            # Log signals (only on the bar where the signal fires)
            if signal != 0:
                sig_df = build_signals_df(df_sig.loc[[bar.open_time]], symbol=symbol)
                sig_rows = sig_df.to_dict(orient="records")
                _append_rows(signals_path, sig_rows, SIGNALS_COLUMNS, "signals.csv")

            # Do not trade on the bootstrapped last candle; start after it.
            if trading_start_ts is not None and bar.open_time <= trading_start_ts:
                continue

            # Engine step
            step = engine.on_bar(
                ts=bar.open_time,
                bar_open=float(bar.open),
                bar_high=float(bar.high),
                bar_low=float(bar.low),
                bar_close=float(bar.close),
                current_date=day,
                signal=signal,
                signal_type=signal_type,
                orb_high=(None if pd.isna(row.get("orb_high")) else float(row.get("orb_high"))),
                orb_low=(None if pd.isna(row.get("orb_low")) else float(row.get("orb_low"))),
                valid_days=valid_days,
            )

            # Fill in symbol + IDs, and due timestamps for scheduled orders
            order_rows = []
            for o in step.orders:
                # Use signal bar time for order id (deterministic)
                oid = f"LIVESIG_{bar.open_time.tz_convert('UTC').strftime('%Y%m%dT%H%M%SZ')}_{signal_type}".replace(" ", "")
                due_ts = (bar.open_time + pd.Timedelta(seconds=bar_seconds * delay_bars)).tz_convert("UTC")
                o2 = {
                    **o,
                    "order_id": oid,
                    "symbol": symbol,
                    "due_timestamp_utc": due_ts.isoformat(),
                }
                order_rows.append(o2)

            fill_rows = []
            for f in step.fills:
                tid = f"T{trade_counter:05d}_{bar.open_time.tz_convert('UTC').strftime('%Y%m%dT%H%M%SZ')}_{signal_type}".replace(
                    " ", ""
                )
                trade_counter += 1
                f2 = {**f, "order_id": tid, "symbol": symbol}
                fill_rows.append(f2)

            pos_rows = []
            for p in step.positions:
                p2 = {**p, "symbol": symbol, "leverage": float(engine.leverage)}
                pos_rows.append(p2)

            _append_rows(orders_path, order_rows, ORDERS_COLUMNS, "orders.csv")
            _append_rows(fills_path, fill_rows, FILLS_COLUMNS, "fills.csv")
            _append_rows(positions_path, pos_rows, POSITIONS_COLUMNS, "positions.csv")
            append_jsonl(events_path, step.events)

            bars_processed += 1

            # Stop conditions
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
                    append_jsonl(
                        events_path,
                        [{"ts": _utcnow_iso(), "type": "STOP_DURATION", "duration_minutes": int(duration_minutes)}],
                    )
                    stop_event.set()
                    stop_requested = True
                    continue

            # If risk manager triggered a global halt, stop.
            if engine.risk_mgr is not None and engine.risk_mgr.halted_global:
                append_jsonl(
                    events_path,
                    [{"ts": _utcnow_iso(), "type": "STOP_RISK_HALT", "reason": engine.risk_mgr.halt_reason}],
                )
                stop_event.set()
                stop_requested = True
                continue

    finally:
        stop_event.set()
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

        append_jsonl(
            events_path,
            [
                {
                    "ts": _utcnow_iso(),
                    "type": "LIVE_RUN_END",
                    "bars_processed": int(bars_processed),
                    "final_equity": float(engine.equity(mark_price=float(df_raw["close"].iloc[-1])) if len(df_raw) else initial_capital),
                }
            ],
        )

    return 0

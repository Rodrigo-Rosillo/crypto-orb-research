from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from backtester.futures_engine import FuturesEngineConfig
from backtester.risk import RiskLimits
from forward.artifacts import append_csv_rows, append_jsonl, build_signals_df
from forward.binance_live import BinanceLiveKlineSource, fetch_recent_klines_df, interval_to_seconds
from forward.shadow import build_signals
from forward.stream_engine import StreamingFuturesShadowEngine


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    # Columns (keep aligned with scripts/forward_test.py)
    signals_cols = ["timestamp_utc", "symbol", "side", "reason", "adx", "orb_low", "orb_high", "close"]
    orders_cols = [
        "timestamp_utc",
        "due_timestamp_utc",
        "order_id",
        "symbol",
        "side",
        "qty",
        "order_type",
        "limit_price",
        "status",
        "status_detail",
        "reason",
    ]
    fills_cols = ["timestamp_utc", "order_id", "symbol", "side", "qty", "fill_price", "fee", "slippage_bps", "exec_model"]
    positions_cols = [
        "timestamp_utc",
        "symbol",
        "side",
        "qty",
        "entry_price",
        "mark_price",
        "unrealized_pnl",
        "equity",
        "margin_used",
        "leverage",
    ]

    stop_event = asyncio.Event()

    # Bootstrap history
    rest_df, rest_meta = fetch_recent_klines_df(
        symbol=symbol,
        interval=timeframe,
        limit=bootstrap_limit,
        market=market,
    )
    df_raw = rest_df.copy()

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
            allowed = float(bar_seconds * (risk_limits.kill_switch.max_data_gap_bars if (risk_limits and risk_limits.enabled) else 2))
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
                break

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

    try:
        async for bar in src.stream_closed(stop_event=stop_event):
            if stop_event.is_set():
                break

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
                        "close": float(bar.close),
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
                append_csv_rows(signals_path, sig_df.to_dict(orient="records"), signals_cols)

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

            append_csv_rows(orders_path, order_rows, orders_cols)
            append_csv_rows(fills_path, fill_rows, fills_cols)
            append_csv_rows(positions_path, pos_rows, positions_cols)
            append_jsonl(events_path, step.events)

            bars_processed += 1

            # Stop conditions
            if max_bars is not None and bars_processed >= int(max_bars):
                append_jsonl(events_path, [{"ts": _utcnow_iso(), "type": "STOP_MAX_BARS", "max_bars": int(max_bars)}])
                stop_event.set()
                break

            if duration_minutes is not None:
                elapsed = (datetime.now(timezone.utc) - start_wall).total_seconds() / 60.0
                if elapsed >= float(duration_minutes):
                    append_jsonl(
                        events_path,
                        [{"ts": _utcnow_iso(), "type": "STOP_DURATION", "duration_minutes": int(duration_minutes)}],
                    )
                    stop_event.set()
                    break

            # If risk manager triggered a global halt, stop.
            if engine.risk_mgr is not None and engine.risk_mgr.halted_global:
                append_jsonl(
                    events_path,
                    [{"ts": _utcnow_iso(), "type": "STOP_RISK_HALT", "reason": engine.risk_mgr.halt_reason}],
                )
                stop_event.set()
                break

    finally:
        stop_event.set()
        hb.cancel()
        try:
            await hb
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

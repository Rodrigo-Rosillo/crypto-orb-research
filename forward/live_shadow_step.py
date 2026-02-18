from __future__ import annotations

import pandas as pd


def process_bar_step(
    engine,
    bar,
    row: pd.Series,
    symbol: str,
    bar_seconds: int,
    delay_bars: int,
    trade_counter: int,
) -> tuple[list, list, list, list, int]:
    signal = int(row.get("signal", 0) or 0)
    signal_type = str(row.get("signal_type", "") or "")
    day = row.get("date")
    valid_days = row.get("_valid_days", None)
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

    order_rows = []
    for o in step.orders:
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

    return order_rows, fill_rows, pos_rows, list(step.events), trade_counter

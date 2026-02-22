from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import pandas as pd

from forward.state_store_sqlite import SQLiteStateStore
from tests.integration.mocks import FakeBinanceClient, build_trader_service


def _entry_row() -> pd.Series:
    return pd.Series(
        {
            "signal": -1,
            "signal_type": "downtrend_breakdown",
            "close": 100.0,
            "orb_high": 200.0,
        }
    )


async def _attempt_entry(db_path: Path) -> None:
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        broker = FakeBinanceClient(
            reject_entry=False,
            simulate_partial_fill_poll=False,
            fill_price=100.0,
        )
        trader = build_trader_service(
            broker=broker,
            store=store,
            state=state,
            work_dir=db_path.parent,
            leverage=1.0,
            position_size=0.1,
            initial_capital=1000.0,
            max_order_rejects_per_day=10,
        )
        bar_open_time = pd.Timestamp("2024-01-01 00:30:00", tz="UTC")
        await trader.maybe_place_trade_from_signal(bar_open_time, _entry_row())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--mode", required=True, choices=["run1", "run2"])
    parser.add_argument("--ready-file", required=True)
    args = parser.parse_args()

    db_path = Path(args.db_path)
    ready_file = Path(args.ready_file)

    if args.mode == "run1":
        asyncio.run(_attempt_entry(db_path))
        ready_file.parent.mkdir(parents=True, exist_ok=True)
        ready_file.write_text("ready\n", encoding="utf-8")
        while True:
            time.sleep(0.05)

    asyncio.run(_attempt_entry(db_path))


if __name__ == "__main__":
    main()

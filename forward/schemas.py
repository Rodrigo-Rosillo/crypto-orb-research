from __future__ import annotations


SIGNALS_COLUMNS = ["timestamp_utc", "symbol", "side", "reason", "adx", "orb_low", "orb_high", "close"]

ORDERS_COLUMNS = [
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

FILLS_COLUMNS = ["timestamp_utc", "order_id", "symbol", "side", "qty", "fill_price", "fee", "slippage_bps", "exec_model"]

POSITIONS_COLUMNS = [
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


def validate_df_columns(df, required_cols, name: str) -> None:
    if df is None:
        raise ValueError(f"{name}: df is None")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")

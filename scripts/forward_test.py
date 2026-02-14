import os

# Determinism locks (must be set before Python does much work)
os.environ["PYTHONHASHSEED"] = "0"

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure repo root is on sys.path so `backtester`, `strategy`, and `forward` imports work
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtester.risk import risk_limits_from_config  # noqa: E402
from forward.artifacts import (  # noqa: E402
    build_orders_fills_positions,
    build_signals_df,
    write_csv,
    write_jsonl,
)
from forward.replay import load_processed_parquet  # noqa: E402
from forward.shadow import run_shadow_futures  # noqa: E402
from forward.utils import ensure_repo_path, maybe_get_forward_cfg, parse_hhmm, parse_utc_ts, utc_run_id  # noqa: E402


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_head() -> str:
    """Best-effort git HEAD hash (empty if not available)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


def write_skeleton(run_dir: Path) -> None:
    """Create placeholder artifacts for not-yet-implemented forward-test modes."""
    (run_dir / "signals.csv").write_text(
        "timestamp_utc,symbol,side,reason,adx,orb_low,orb_high,close\n", encoding="utf-8"
    )
    (run_dir / "orders.csv").write_text(
        "timestamp_utc,due_timestamp_utc,order_id,symbol,side,qty,order_type,limit_price,status,reason\n",
        encoding="utf-8",
    )
    (run_dir / "fills.csv").write_text(
        "timestamp_utc,order_id,symbol,side,qty,fill_price,fee,slippage_bps,exec_model\n",
        encoding="utf-8",
    )
    (run_dir / "positions.csv").write_text(
        "timestamp_utc,symbol,side,qty,entry_price,mark_price,unrealized_pnl,equity,margin_used,leverage\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").touch(exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 forward test runner")
    parser.add_argument(
        "--config",
        type=str,
        default="config_forward_test.yaml",
        help="Path to YAML config (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["shadow", "testnet"],
        help="Forward-test mode. If omitted, uses config.forward_test.mode or 'shadow'.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["replay", "live"],
        help="Data source. If omitted, uses config.forward_test.source or 'replay'.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override output directory for this run (will create run_id subfolder).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id. Default is current UTC timestamp, e.g. 20260214T210300Z.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Replay start timestamp/date (UTC). Example: 2025-01-01",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Replay end timestamp/date (UTC). Example: 2025-06-30",
    )
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")

    ft_cfg = maybe_get_forward_cfg(cfg)
    mode = args.mode or str(ft_cfg.get("mode", "shadow"))
    source = args.source or str(ft_cfg.get("source", "replay"))

    out_root = args.out or ft_cfg.get("out_dir") or "reports/forward_test"
    out_root = ensure_repo_path(REPO_ROOT, str(out_root))

    run_id = args.run_id or utc_run_id()
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist config used (for audit/repro)
    (run_dir / "config_used.yaml").write_text(cfg_text, encoding="utf-8")

    # Risk limits snapshot
    try:
        risk_limits = risk_limits_from_config(cfg)
        risk_limits_dict = risk_limits.to_dict() if hasattr(risk_limits, "to_dict") else {}
    except Exception:
        risk_limits = None
        risk_limits_dict = {}

    # Metadata placeholders
    note = ""
    parquet_path_used = ""
    parquet_sha256 = ""
    rows_used = 0

    # Artifact columns
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

    events: list[dict] = []

    if source == "replay" and mode == "shadow":
        note = "Phase 5 Step 2: replay source + shadow execution (deterministic)."

        # Strategy/execution params
        symbol = str(cfg.get("symbol", "SOLUSDT"))
        timeframe = str(cfg.get("timeframe", "30m"))

        orb_start = parse_hhmm(cfg["orb"]["start"])
        orb_end = parse_hhmm(cfg["orb"]["end"])
        orb_cutoff = parse_hhmm(cfg["orb"]["cutoff"])

        adx_period = int(cfg["adx"]["period"])
        adx_threshold = float(cfg["adx"]["threshold"])

        initial_capital = float(cfg["risk"]["initial_capital"])
        position_size = float(cfg["risk"]["position_size"])
        taker_fee_rate = float(cfg["fees"]["taker_fee_rate"])

        lev_cfg = cfg.get("leverage") or {}
        leverage = float(lev_cfg.get("max_leverage", 1.0)) if bool(lev_cfg.get("enabled", True)) else 1.0

        exec_cfg = (ft_cfg.get("execution_model") or {}) if isinstance(ft_cfg, dict) else {}
        delay_bars = int(exec_cfg.get("delay_bars", 1))
        slippage_bps = float(exec_cfg.get("slippage_bps", 0.0))

        # Replay bounds (CLI overrides config)
        replay_cfg = ft_cfg.get("replay") if isinstance(ft_cfg.get("replay"), dict) else {}
        start_utc = parse_utc_ts(args.start) or parse_utc_ts((replay_cfg or {}).get("start"))
        end_utc = parse_utc_ts(args.end) or parse_utc_ts((replay_cfg or {}).get("end"))

        # Valid days
        valid_days_path = ensure_repo_path(
            REPO_ROOT, str(cfg.get("valid_days", "data/processed/valid_days.csv"))
        )
        valid_days_df = pd.read_csv(valid_days_path)
        valid_days = set(pd.to_datetime(valid_days_df["date_utc"], utc=True).dt.date)

        # Load replay dataset
        df_raw, parquet_path = load_processed_parquet(
            REPO_ROOT, symbol=symbol, timeframe=timeframe, start_utc=start_utc, end_utc=end_utc
        )
        parquet_path_used = str(parquet_path)
        parquet_sha256 = sha256_file(parquet_path)
        rows_used = int(len(df_raw))

        events.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": "RUN_START",
                "mode": mode,
                "source": source,
                "rows": rows_used,
                "start_utc": start_utc.isoformat() if start_utc is not None else "",
                "end_utc": end_utc.isoformat() if end_utc is not None else "",
            }
        )

        shadow_res = run_shadow_futures(
            df_raw=df_raw,
            valid_days=valid_days,
            orb_start=orb_start,
            orb_end=orb_end,
            orb_cutoff=orb_cutoff,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            initial_capital=initial_capital,
            position_size=position_size,
            taker_fee_rate=taker_fee_rate,
            leverage=leverage,
            delay_bars=delay_bars,
            slippage_bps=slippage_bps,
            fee_mult=1.0,
            funding_rate_per_8h=float(cfg.get("funding", {}).get("rate_per_8h", 0.0)) if isinstance(cfg.get("funding"), dict) else 0.0,
            risk_limits=risk_limits,
        )

        # Artifacts
        signals_df = build_signals_df(shadow_res.df_sig, symbol=symbol)
        orders_df, fills_df, positions_df, derived_events = build_orders_fills_positions(
            df_sig=shadow_res.df_sig,
            trades=shadow_res.trades,
            equity_curve=shadow_res.equity_curve,
            symbol=symbol,
            delay_bars=delay_bars,
        )

        # Add risk events (if any)
        risk_ev = (shadow_res.stats.get("risk") or {}).get("events") if isinstance(shadow_res.stats, dict) else None
        if isinstance(risk_ev, list):
            for e in risk_ev:
                events.append({"ts": str(e.get("ts", "")), "type": "RISK", **e})

        events.extend(derived_events)
        events.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": "RUN_END",
                "final_equity": shadow_res.stats.get("final_equity"),
                "trades": shadow_res.stats.get("trades"),
            }
        )

        write_csv(signals_df, run_dir / "signals.csv", signals_cols)
        write_csv(orders_df, run_dir / "orders.csv", orders_cols)
        write_csv(fills_df, run_dir / "fills.csv", fills_cols)
        write_csv(positions_df, run_dir / "positions.csv", positions_cols)
        write_jsonl(run_dir / "events.jsonl", events)

        (run_dir / "shadow_stats.json").write_text(
            json.dumps(shadow_res.stats, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    else:
        # Other combinations are wired in later Phase 5 steps.
        note = f"Not implemented yet: mode={mode}, source={source}. Implemented in Step 2: replay+shadow."
        write_skeleton(run_dir)

    meta = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "source": source,
        "out_dir": str(run_dir),
        "config_path": str(config_path),
        "config_sha256": sha256_text(cfg_text),
        "dataset": {
            "processed_parquet": parquet_path_used,
            "processed_parquet_sha256": parquet_sha256,
            "rows_used": rows_used,
            "start_utc": args.start,
            "end_utc": args.end,
        },
        "repo_git_head": get_git_head(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "risk_limits": risk_limits_dict,
        "note": note,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"[forward_test] created run folder: {run_dir}")
    print("[forward_test] artifacts:")
    for name in [
        "signals.csv",
        "orders.csv",
        "fills.csv",
        "positions.csv",
        "events.jsonl",
        "run_metadata.json",
    ]:
        if (run_dir / name).exists():
            print("  -", name)
    if (run_dir / "shadow_stats.json").exists():
        print("  - shadow_stats.json")

    if source == "replay" and mode == "shadow":
        print("[forward_test] ✅ replay+shadow run completed.")
    else:
        print("[forward_test] ℹ️ Only replay+shadow is implemented in Step 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

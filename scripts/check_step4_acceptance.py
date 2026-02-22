"""Phase 5 / Step 4 acceptance checks.

This script validates that the Binance USD-M Futures demo/testnet broker loop is working.

It reads artifacts from a forward-test run directory:
  reports/forward_test/<RUN_ID>/

Checks performed:
  A) Smoke test round-trip (entry + flatten) is fully recorded (events + CSVs)
  B) Testnet live-run operational health (no auth errors, no reconcile mismatch)
  C) Restart-resume (optional): detects multiple LIVE_RUN_START events in the same run folder

Exit codes:
  0 = PASS
  1 = FAIL
  2 = INCOMPLETE (e.g., restart check not observed)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            # Keep going; acceptance check should be robust to one bad line.
            continue
    return out


def _event_type(e: Dict[str, Any]) -> str:
    return str(e.get("type") or e.get("kind") or "")


def _find_latest_run_dir(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    # Filter out hidden dirs if any
    dirs = [d for d in dirs if not d.name.startswith(".")]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name)[-1]


def _read_csv_optional(path: Path):
    """Read CSV using pandas if available, else fallback to csv module."""
    if not path.exists():
        return None
    try:
        import pandas as pd  # type: ignore

        return pd.read_csv(path)
    except Exception:
        import csv

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)


def _df_has_col(df: Any, col: str) -> bool:
    try:
        return col in df.columns
    except Exception:
        # list[dict]
        return bool(df) and col in (df[0].keys() if isinstance(df[0], dict) else {})


def _df_len(df: Any) -> int:
    try:
        return int(len(df))
    except Exception:
        return 0


def _df_filter_equals(df: Any, col: str, value: str) -> int:
    """Return count of rows where df[col] == value."""
    try:
        # pandas
        return int((df[col].astype(str) == value).sum())
    except Exception:
        if isinstance(df, list):
            return sum(1 for r in df if str(r.get(col, "")) == value)
        return 0


def _net_signed_qty_from_fills(fills_df: Any) -> Optional[float]:
    """Infer whether we ended flat from fills alone.

    Returns signed net qty if possible, else None.
    """
    if fills_df is None or _df_len(fills_df) == 0:
        return None
    if not _df_has_col(fills_df, "side") or not _df_has_col(fills_df, "qty"):
        return None

    def signed(side: str, qty: float) -> float:
        s = str(side).upper()
        if "SHORT" in s or s == "SELL":
            return -abs(qty)
        if "LONG" in s or s == "BUY":
            return abs(qty)
        return 0.0

    total = 0.0
    try:
        # pandas
        for _, row in fills_df.iterrows():
            try:
                qty = float(row.get("qty", 0.0))
            except Exception:
                qty = 0.0
            total += signed(row.get("side", ""), qty)
        return float(total)
    except Exception:
        if isinstance(fills_df, list):
            for r in fills_df:
                try:
                    qty = float(r.get("qty", 0.0))
                except Exception:
                    qty = 0.0
                total += signed(r.get("side", ""), qty)
            return float(total)
        return None


def _is_flat_from_positions(pos_df: Any) -> Optional[bool]:
    if pos_df is None:
        return None
    if _df_len(pos_df) == 0:
        return None
    # Try common quantity columns
    for qty_col in ("qty", "position_amt", "positionAmt", "position_size", "positionSize"):
        if _df_has_col(pos_df, qty_col):
            try:
                last = pos_df.iloc[-1]  # pandas
                q = float(last[qty_col])
                return abs(q) < 1e-9
            except Exception:
                try:
                    # list[dict]
                    q = float(pos_df[-1].get(qty_col, 0.0))
                    return abs(q) < 1e-9
                except Exception:
                    continue
    # Try side col
    if _df_has_col(pos_df, "side"):
        try:
            last_side = str(pos_df.iloc[-1]["side"]).upper()
        except Exception:
            last_side = str(pos_df[-1].get("side", "")).upper()
        if last_side == "FLAT":
            return True
    return None


def _summarize_result(
    header: str,
    failures: List[str],
    warnings: List[str],
    lines: List[str],
    *,
    require_restart: bool,
    restart_observed: bool,
) -> Tuple[int, str]:
    """Format a section result and choose an exit code."""

    if failures:
        lines.append(f"\n{header}: FAIL")
        for f in failures:
            lines.append(f"- {f}")
        if warnings:
            lines.append("\nWARNINGS")
            for w in warnings:
                lines.append(f"- {w}")
        return 1, "\n".join(lines)

    if require_restart and not restart_observed:
        lines.append(f"\n{header}: FAIL")
        lines.append("- Restart not observed: expected >=2 LIVE_RUN_START events in the same run folder")
        if warnings:
            lines.append("\nWARNINGS")
            for w in warnings:
                lines.append(f"- {w}")
        return 1, "\n".join(lines)

    # If restart is not required, lack of restart is just informational; do not mark INCOMPLETE.
    if not restart_observed and not require_restart:
        warnings.append("Restart not observed (run twice with same --run-id to validate resume)")

    if warnings:
        lines.append(f"\n{header}: PASS (with warnings)")
        for w in warnings:
            lines.append(f"- {w}")
        return 0, "\n".join(lines)

    lines.append(f"\n{header}: PASS")
    return 0, "\n".join(lines)


def check_smoke_run(run_dir: Path, require_restart: bool = False) -> Tuple[int, str]:
    """Acceptance check for a smoke-test run."""

    events_path = run_dir / "events.jsonl"
    orders_path = run_dir / "orders.csv"
    fills_path = run_dir / "fills.csv"
    positions_path = run_dir / "positions.csv"
    state_path = run_dir / "state.json"

    failures: List[str] = []
    warnings: List[str] = []

    events = _read_jsonl(events_path)
    etypes = [_event_type(e) for e in events]

    # --- A) Smoke test round-trip ---
    if "TESTNET_SMOKE_ENTRY" not in etypes:
        failures.append("Missing TESTNET_SMOKE_ENTRY in events.jsonl")
    if "TESTNET_SMOKE_FLATTEN" not in etypes:
        failures.append("Missing TESTNET_SMOKE_FLATTEN in events.jsonl")
    # LIVE_RUN_END reason SMOKE_TEST
    live_ends = [e for e in events if _event_type(e) == "LIVE_RUN_END"]
    if not live_ends:
        failures.append("Missing LIVE_RUN_END in events.jsonl")
    else:
        if not any(str(e.get("reason", "")).upper() == "SMOKE_TEST" for e in live_ends):
            failures.append("LIVE_RUN_END exists but none has reason=SMOKE_TEST")

    if "TESTNET_SMOKE_FAILED" in etypes:
        failures.append("TESTNET_SMOKE_FAILED present")

    # Broker base URL sanity (warning)
    for e in events:
        if _event_type(e) == "TESTNET_BROKER_CONFIG":
            base = str(e.get("base_url", ""))
            if "demo-fapi.binance.com" not in base:
                warnings.append(f"Broker base_url is '{base}' (expected to contain demo-fapi.binance.com)")

    orders_df = _read_csv_optional(orders_path)
    fills_df = _read_csv_optional(fills_path)
    pos_df = _read_csv_optional(positions_path)

    if orders_df is None:
        failures.append("Missing orders.csv")
    else:
        if not _df_has_col(orders_df, "status_detail"):
            failures.append("orders.csv missing 'status_detail' column")
        else:
            n_entry = _df_filter_equals(orders_df, "status_detail", "smoke_entry")
            n_flat = _df_filter_equals(orders_df, "status_detail", "smoke_flatten")
            if n_entry < 1:
                failures.append("orders.csv has no row with status_detail=smoke_entry")
            if n_flat < 1:
                failures.append("orders.csv has no row with status_detail=smoke_flatten")

    if fills_df is None:
        failures.append("Missing fills.csv")
    else:
        if _df_len(fills_df) < 2:
            failures.append(f"fills.csv expected >=2 fills for smoke test; got {_df_len(fills_df)}")
        if _df_has_col(fills_df, "exec_model"):
            try:
                # pandas
                has_reduce = bool(fills_df["exec_model"].astype(str).str.contains("reduce", case=False, na=False).any())
            except Exception:
                has_reduce = any("reduce" in str(r.get("exec_model", "")).lower() for r in (fills_df or []))
            if not has_reduce:
                warnings.append("fills.csv has no exec_model containing 'reduce' (flatten fill may not be flagged)")

    # Flat check (positions preferred, fallback to fills net qty)
    flat = _is_flat_from_positions(pos_df)
    if flat is None:
        net = _net_signed_qty_from_fills(fills_df)
        if net is None:
            warnings.append("Could not determine flat/position state from positions.csv or fills.csv")
        else:
            if abs(net) > 1e-6:
                failures.append(f"Net signed qty from fills is {net:.6f} (expected ~0 for smoke test)")
    else:
        if flat is False:
            failures.append("positions.csv indicates an open position at end of run")

    # --- B) State persistence + reconciliation ---
    if not state_path.exists():
        warnings.append("state.json not found (restart-resume checks will be limited)")
    else:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                warnings.append("state.json is not a dict")
        except Exception as e:
            warnings.append(f"state.json could not be parsed: {e}")

    if "RECON_MISMATCH" in etypes:
        failures.append("RECON_MISMATCH present in events.jsonl")
    if any(_event_type(e) == "LIVE_RUN_END" and str(e.get("reason", "")) == "RECON_MISMATCH" for e in events):
        failures.append("LIVE_RUN_END reason=RECON_MISMATCH")

    # --- C) Restart-resume (optional) ---
    n_starts = sum(1 for e in events if _event_type(e) == "LIVE_RUN_START")
    restart_observed = n_starts >= 2
    # --- Result ---
    lines: List[str] = []
    lines.append(f"Run dir: {run_dir}")
    lines.append(f"Smoke events: entry={'TESTNET_SMOKE_ENTRY' in etypes}, flatten={'TESTNET_SMOKE_FLATTEN' in etypes}, live_end_smoke={any(str(e.get('reason','')).upper()=='SMOKE_TEST' for e in live_ends)}")
    lines.append(f"orders.csv rows: {_df_len(orders_df) if orders_df is not None else 0}, fills.csv rows: {_df_len(fills_df) if fills_df is not None else 0}")
    lines.append(f"Restart observed: {restart_observed} (LIVE_RUN_START count={n_starts})")

    return _summarize_result("SMOKE", failures, warnings, lines, require_restart=require_restart, restart_observed=restart_observed)


def check_testnet_run(run_dir: Path, require_restart: bool = False) -> Tuple[int, str]:
    """Operational health checks for a non-smoke testnet live run.

    This does NOT require that a strategy trade occurred (signals may be rare).
    """

    events_path = run_dir / "events.jsonl"
    state_path = run_dir / "state.json"

    failures: List[str] = []
    warnings: List[str] = []

    events = _read_jsonl(events_path)
    etypes = [_event_type(e) for e in events]

    n_starts = sum(1 for e in events if _event_type(e) == "LIVE_RUN_START")
    restart_observed = n_starts >= 2

    if "LIVE_RUN_START" not in etypes:
        failures.append("Missing LIVE_RUN_START in events.jsonl")

    # Must not have fatal auth / API errors
    fatal_types = {
        "TESTNET_AUTH_ERROR",
        "TESTNET_POSITION_RISK_FAILED",
        "TESTNET_SMOKE_FAILED",
    }
    for t in fatal_types:
        if t in etypes:
            failures.append(f"Fatal event present: {t}")

    # Reconcile safety
    if "RECON_MISMATCH" in etypes:
        failures.append("RECON_MISMATCH present in events.jsonl")
    if any(_event_type(e) == "LIVE_RUN_END" and str(e.get("reason", "")) == "RECON_MISMATCH" for e in events):
        failures.append("LIVE_RUN_END reason=RECON_MISMATCH")

    bar_closed_count = sum(1 for e in events if _event_type(e) == "BAR_CLOSED")
    if not state_path.exists():
        warnings.append("state.json not found (restart-resume checks will be limited)")
    else:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(state, dict):
                bp = int(state.get("bars_processed", 0) or 0)
                # Only warn about bars_processed if we actually observed BAR_CLOSED events.
                if bar_closed_count > 0 and bp <= 0:
                    warnings.append(f"state.json bars_processed={bp} but BAR_CLOSED count={bar_closed_count}")
            else:
                warnings.append("state.json is not a dict")
        except Exception as e:
            warnings.append(f"state.json could not be parsed: {e}")

    # End reason could be STOP_DURATION, BOOTSTRAP_STALE, etc. That's OK.
    if "LIVE_RUN_END" not in etypes:
        warnings.append("LIVE_RUN_END not found (run may still be running or was killed without logging)")

    lines: List[str] = []
    lines.append(f"Run dir: {run_dir}")
    lines.append(f"LIVE_RUN_START count: {n_starts}")
    lines.append(f"Restart observed: {restart_observed} (>=2 starts)")

    return _summarize_result("TESTNET", failures, warnings, lines, require_restart=require_restart, restart_observed=restart_observed)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 Step 4 acceptance checks (Binance demo/testnet broker).")
    ap.add_argument("--run-id", default=None, help="Run id folder under reports/forward_test/. If omitted, uses latest.")
    ap.add_argument("--smoke-run-id", default=None, help="Optional: run id for a smoke-test run to validate round-trip artifacts.")
    ap.add_argument("--resume-run-id", default=None, help="Optional: run id for a normal testnet run to validate restart/resume safety.")
    ap.add_argument("--repo-root", default=None, help="Repo root path. Defaults to auto-detected.")
    ap.add_argument("--require-restart", action="store_true", help="Fail unless a restart is observed (>=2 LIVE_RUN_START events).")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    base = root / "reports" / "forward_test"

    def resolve_run_dir(run_id: Optional[str]) -> Optional[Path]:
        if run_id:
            d = base / run_id
            return d if d.exists() else None
        return _find_latest_run_dir(base)

    # If explicit smoke/resume run ids are provided, run both checks and combine results.
    if args.smoke_run_id or args.resume_run_id:
        parts: List[Tuple[int, str]] = []
        if args.smoke_run_id:
            smoke_dir = resolve_run_dir(args.smoke_run_id)
            if smoke_dir is None:
                print(f"FAIL\nCould not find smoke run dir: {base / args.smoke_run_id}")
                return 1
            parts.append(check_smoke_run(smoke_dir, require_restart=False))
        if args.resume_run_id:
            resume_dir = resolve_run_dir(args.resume_run_id)
            if resume_dir is None:
                print(f"FAIL\nCould not find resume run dir: {base / args.resume_run_id}")
                return 1
            parts.append(check_testnet_run(resume_dir, require_restart=bool(args.require_restart)))

        worst = 0
        for i, (code, msg) in enumerate(parts):
            if i:
                print("\n" + ("=" * 60) + "\n")
            print(msg)
            worst = max(worst, code)
        return worst

    # Single-run auto mode
    run_dir = resolve_run_dir(args.run_id)
    if run_dir is None:
        print(f"FAIL\nCould not find run dir under: {base}")
        return 1

    # Auto-detect smoke vs normal based on events
    events = _read_jsonl(run_dir / "events.jsonl")
    is_smoke = any(str(e.get("reason", "")).upper() == "SMOKE_TEST" for e in events if _event_type(e) == "LIVE_RUN_END") or any(
        _event_type(e).startswith("TESTNET_SMOKE") for e in events
    )
    if is_smoke:
        code, msg = check_smoke_run(run_dir, require_restart=bool(args.require_restart))
    else:
        code, msg = check_testnet_run(run_dir, require_restart=bool(args.require_restart))
    print(msg)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

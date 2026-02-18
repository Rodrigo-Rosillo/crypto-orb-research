from __future__ import annotations
import sys
import json
from pathlib import Path
from html import escape
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
from core.utils import stable_json


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    baseline_dir = repo_root / "reports" / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    results_path = baseline_dir / "results.json"
    run_meta_path = baseline_dir / "run_metadata.json"
    trades_path = baseline_dir / "trades.csv"
    equity_path = baseline_dir / "equity_curve.csv"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}. Run: python scripts/run_baseline.py")

    results = load_json(results_path)
    run_meta = load_json(run_meta_path) if run_meta_path.exists() else {}

    metrics = results.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    # 1) Write metrics.json (as requested by Step 8)
    (baseline_dir / "metrics.json").write_text(stable_json(metrics), encoding="utf-8")

    # Load equity curve for plotting
    eq_df = pd.read_csv(equity_path) if equity_path.exists() else pd.DataFrame(columns=["timestamp", "equity"])
    x = eq_df["timestamp"].astype(str).tolist() if "timestamp" in eq_df else []
    y = eq_df["equity"].astype(float).tolist() if "equity" in eq_df else []

    # Load trades preview
    trades_preview_html = "<p>No trades.csv found or it is empty.</p>"
    if trades_path.exists() and trades_path.stat().st_size > 0:
        tdf = pd.read_csv(trades_path)
        if len(tdf):
            show_cols = [c for c in ["entry_time", "exit_time", "type", "signal_type", "pnl", "return", "exit_reason"] if c in tdf.columns]
            tail = tdf[show_cols].tail(20).copy()
            trades_preview_html = tail.to_html(index=False, escape=True)

    # Metadata rows
    meta_rows = []
    for k in ["git_commit", "config_sha256", "dataset_sha256", "manifest_sha256", "python_version", "platform"]:
        if k in run_meta and run_meta[k]:
            meta_rows.append((k, str(run_meta[k])))

    used_files = run_meta.get("used_files", [])
    if used_files:
        meta_rows.append(("used_files", "\n".join(map(str, used_files))))

    meta_table = "".join(
        f"<tr><th>{escape(k)}</th><td><pre style='margin:0; white-space:pre-wrap'>{escape(v)}</pre></td></tr>"
        for k, v in meta_rows
    ) or "<tr><td colspan='2'>No run_metadata.json found.</td></tr>"

    # Metrics table
    metrics_rows = "".join(
        f"<tr><th>{escape(str(k))}</th><td style='text-align:right'>{escape(str(v))}</td></tr>"
        for k, v in metrics.items()
    ) or "<tr><td colspan='2'>No metrics found in results.json</td></tr>"

    # Simple self-contained HTML report (canvas plot, no external libs)
    report_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>ORB Baseline Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color:#111; }}
    h1,h2 {{ margin: 0 0 12px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }}
    .card {{ border: 1px solid #e5e5e5; border-radius: 12px; padding: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
    th {{ text-align: left; width: 40%; color:#333; }}
    pre {{ font-size: 12px; }}
    canvas {{ width: 100%; height: 260px; border: 1px solid #eee; border-radius: 10px; }}
    .small {{ color:#666; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>ORB Baseline Report</h1>
  <p class="small">Generated from reports/baseline outputs. Open this file locally in a browser.</p>

  <div class="grid">
    <div class="card">
      <h2>Run Metadata</h2>
      <table>{meta_table}</table>
    </div>

    <div class="card">
      <h2>Metrics</h2>
      <table>{metrics_rows}</table>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>Equity Curve</h2>
    <canvas id="eq"></canvas>
    <p class="small">Data source: equity_curve.csv</p>
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>Recent Trades (last 20)</h2>
    {trades_preview_html}
  </div>

<script>
  const x = {json.dumps(x)};
  const y = {json.dumps(y)};

  const canvas = document.getElementById("eq");
  const ctx = canvas.getContext("2d");

  // Fit canvas to device pixels
  const cssW = canvas.clientWidth || 900;
  const cssH = 260;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  ctx.scale(dpr, dpr);

  function draw() {{
    ctx.clearRect(0,0,cssW,cssH);

    if (!y.length) {{
      ctx.fillText("No equity data found.", 10, 20);
      return;
    }}

    const padL = 40, padR = 10, padT = 10, padB = 30;
    const w = cssW - padL - padR;
    const h = cssH - padT - padB;

    const ymin = Math.min(...y);
    const ymax = Math.max(...y);
    const yr = (ymax - ymin) || 1;

    // axes
    ctx.strokeStyle = "#ddd";
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + h);
    ctx.lineTo(padL + w, padT + h);
    ctx.stroke();

    // line
    ctx.strokeStyle = "#111";
    ctx.beginPath();
    for (let i = 0; i < y.length; i++) {{
      const px = padL + (i / (y.length - 1)) * w;
      const py = padT + (1 - (y[i] - ymin) / yr) * h;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }}
    ctx.stroke();

    // labels
    ctx.fillStyle = "#666";
    ctx.font = "12px system-ui";
    ctx.fillText(ymax.toFixed(2), 6, padT + 12);
    ctx.fillText(ymin.toFixed(2), 6, padT + h);

    const left = x[0] || "";
    const right = x[x.length - 1] || "";
    ctx.fillText(left, padL, padT + h + 20);
    const tw = ctx.measureText(right).width;
    ctx.fillText(right, padL + w - tw, padT + h + 20);
  }}

  draw();
</script>
</body>
</html>
"""

    (baseline_dir / "report.html").write_text(report_html, encoding="utf-8")
    print(f"[OK] Wrote: {baseline_dir / 'metrics.json'}")
    print(f"[OK] Wrote: {baseline_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

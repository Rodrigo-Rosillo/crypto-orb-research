from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from core.utils import stable_json


def build_html(report: Dict[str, Any]) -> str:
    def kv_table(d: Dict[str, Any]) -> str:
        rows = "".join(
            f"<tr><th>{escape(str(k))}</th><td><pre style='margin:0; white-space:pre-wrap'>{escape(str(v))}</pre></td></tr>"
            for k, v in d.items()
        )
        return f"<table>{rows or '<tr><td colspan=2>No data</td></tr>'}</table>"

    def list_table(items: List[Dict[str, Any]], limit: int = 20) -> str:
        if not items:
            return "<p class='small'>No examples.</p>"
        df = pd.DataFrame(items[:limit])
        return df.to_html(index=False, escape=True)

    run = report.get("run", {})
    timing = report.get("timing_divergence", {})
    data = report.get("data_divergence", {})
    exe = report.get("execution_divergence", {})
    rej = report.get("reject_divergence", {})
    fund = report.get("funding_divergence", {})

    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <title>Forward Test Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color:#111; }}
    h1,h2,h3 {{ margin: 0 0 12px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }}
    .card {{ border: 1px solid #e5e5e5; border-radius: 12px; padding: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
    th {{ text-align: left; width: 40%; color:#333; }}
    pre {{ font-size: 12px; }}
    .small {{ color:#666; font-size: 12px; }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#f3f4f6; font-size:12px; }}
  </style>
</head>
<body>
  <h1>Forward Test Divergence Report</h1>
  <p class='small'>Generated at {escape(str(report.get('generated_at_utc','')))} for run <span class='pill'>{escape(str(run.get('run_id','')))}</span></p>

  <div class='grid'>
    <div class='card'>
      <h2>Run Summary</h2>
      {kv_table(run)}
    </div>
    <div class='card'>
      <h2>Timing Divergence</h2>
      {kv_table(timing.get('ingest_delay_seconds', {}))}
      <h3 style='margin-top:12px;'>Missed Bars</h3>
      {kv_table(timing.get('missed_bars', {}))}
    </div>
  </div>

  <div class='card' style='margin-top:16px;'>
    <h2>Data Divergence (Live vs Reference Parquet)</h2>
    <p class='small'>{escape(str(data.get('notes','')))}</p>
    {kv_table(data.get('summary', {}))}
    <h3 style='margin-top:12px;'>Top Examples (bps)</h3>
    {list_table(data.get('examples', []))}
  </div>

  <div class='card' style='margin-top:16px;'>
    <h2>Execution Divergence (Fill vs Next-Open Reference)</h2>
    <p class='small'>{escape(str(exe.get('notes','')))}</p>
    {kv_table(exe.get('summary', {}))}
    <h3 style='margin-top:12px;'>Top Examples (bps)</h3>
    {list_table(exe.get('examples', []))}
  </div>

  <div class='card' style='margin-top:16px;'>
    <h2>Reject Divergence</h2>
    {kv_table(rej.get('summary', {}))}
    <h3 style='margin-top:12px;'>Examples</h3>
    {list_table(rej.get('examples', []))}
  </div>

  <div class='card' style='margin-top:16px;'>
    <h2>Funding Divergence</h2>
    {kv_table(fund)}
  </div>

  <div class='card' style='margin-top:16px;'>
    <h2>Raw JSON</h2>
    <p class='small'>For programmatic use, open forward_test_report.json in the same run folder.</p>
    <pre style='white-space:pre-wrap'>{escape(json.dumps(report, indent=2, ensure_ascii=False)[:8000])}</pre>
  </div>
</body>
</html>
"""


def write_report(run_dir: Path, report: Dict[str, Any]) -> Tuple[Path, Path]:
    json_out = run_dir / "forward_test_report.json"
    html_out = run_dir / "forward_test_report.html"
    json_out.write_text(stable_json(report), encoding="utf-8")
    html_out.write_text(build_html(report), encoding="utf-8")
    return json_out, html_out

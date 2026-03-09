from __future__ import annotations

import argparse
import csv
import json
import html
import logging
import math
import os
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from shutil import copyfile

import pandas as pd
import requests
import yfinance as yf


# =========================
# Config
# =========================

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "credit_spread": "BAMLH0A0HYM2",
    "us10y": "DGS10",
    "us2y": "DGS2",
    "fed_balance": "WALCL",
    "vix": "VIXCLS",
    "m2": "WM2NS",
    "cpi": "CPIAUCSL",
}

YF_TICKERS = {
    "hyg": "HYG",
    "nasdaq": "^IXIC",
    "move": "^MOVE",
    "usd": "DX-Y.NYB",
    "usd_fallback": "UUP",
    "spy": "SPY",
    "rsp": "RSP",
    "brent": "BZ=F",
}

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "cyan": "\033[36m",
}

STATUS_EXPLANATION = {
    "OK": "normal conditions",
    "ELEVATED": "early warning / rising tension",
    "WARNING": "clear market stress",
    "ALARM": "rare, true crisis signal",
    "UNAVAILABLE": "data unavailable",
}


# =========================
# Attribution / history config
# =========================

HISTORY_FILE = Path("crash_dashboard_history.json")
SCORE_CHANGE_EPSILON = 0.5        # below this => "stable"
CONTRIBUTION_EPSILON = 0.2        # below this => do not show in top summary
MAX_TOP_DRIVERS = 3

# IMPORTANT:
# Adjust these keys to YOUR exact indicator names
INDICATOR_META = {
    "credit_spread": {"label": "HY Spread", "system": "Credit Market", "weight": 1.8},
    "hyg": {"label": "HYG", "system": "Risk Appetite", "weight": 1.7},
    "move": {"label": "MOVE", "system": "Bond Market", "weight": 1.5},
    "vix": {"label": "VIX", "system": "Equity Stress", "weight": 1.2},
    "vol_regime": {"label": "Vol Regime", "system": "Equity Stress", "weight": 1.0},
    "yield_curve_2s10s": {"label": "Yield Curve", "system": "Rates / Macro", "weight": 1.0},
    "global_liquidity": {"label": "Global Liquidity", "system": "Liquidity", "weight": 1.0},
    "usd_index": {"label": "USD", "system": "Global Liquidity", "weight": 0.7},
    "sp_breadth": {"label": "Breadth", "system": "Market Structure", "weight": 0.7},
    "nasdaq": {"label": "Nasdaq", "system": "Risk Assets", "weight": 0.7},
    "fed_balance": {"label": "Fed Balance Sheet", "system": "Liquidity", "weight": 0.4},
}

# =========================
# Logging / file helpers
# =========================

LOGGER = logging.getLogger("macro_crash_radar")


class Tee:
    """
    Write stdout to multiple streams at the same time.
    Useful for terminal + file output.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

def html_status_color(status: str) -> str:
    if status == "ALARM":
        return "#b91c1c"   # red-700
    if status == "WARNING":
        return "#d97706"   # amber-600
    if status == "ELEVATED":
        return "#ca8a04"   # yellow-600
    if status == "UNAVAILABLE":
        return "#0891b2"   # cyan-600
    return "#15803d"       # green-700     

def format_indicator_value(ind: "Indicator") -> str:
    if not isinstance(ind.value, (int, float)) or ind.value is None:
        return "n/a"

    if ind.key == "fed_balance":
        return f"{ind.value / 1e6:.4f}T"

    return f"{ind.value:.2f}"           


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> Path:
    """
    Professional logging setup:
    - creates logs directory if needed
    - writes runtime logs to timestamped file
    - also logs to console
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
    logfile = log_path / f"macro_crash_radar_{timestamp}.log"

    LOGGER.setLevel(level)
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)
    LOGGER.propagate = False

    LOGGER.info("Logging initialized")
    return logfile


def build_report_filename(
    report_dir: str = "reports",
    prefix: str = "macro_crash_radar_report",
    extension: str = "txt",
) -> Path:
    """
    Creates a dated filename like:
    reports/macro_crash_radar_report_2026-03-09_10-42-15_UTC.txt
    """
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
    return out_dir / f"{prefix}_{timestamp}.{extension}"

def build_html_report_filename(
    report_dir: str = "reports",
    prefix: str = "macro_crash_radar_report",
) -> Path:
    return build_report_filename(
        report_dir=report_dir,
        prefix=prefix,
        extension="html",
    )    

def write_html_report(output: "RadarOutput", html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_content = build_html_report(output)

    print(f"DEBUG html_path        = {html_path}")
    print(f"DEBUG html_path.parent = {html_path.parent}")

    # write the timestamped archive report
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # ALSO create the latest report for GitHub Pages
    latest_path = html_path.parent / "index.html"
    print(f"DEBUG latest_path      = {latest_path}")

    copyfile(html_path, latest_path)

    print("DEBUG index.html copied successfully")
    print(f"DEBUG archive exists? {html_path.exists()}")
    print(f"DEBUG index exists?   {latest_path.exists()}")


def write_report_to_file(
    output: "RadarOutput",
    report_path: Path,
    also_print_to_console: bool = False,
) -> None:
    """
    Writes the formatted report with exactly the same print layout.
    Optionally also prints to console at the same time.
    """
    with open(report_path, "w", encoding="utf-8") as f:
        if also_print_to_console:
            with redirect_stdout(Tee(sys.stdout, f)):
                print_report(output)
        else:
            with redirect_stdout(f):
                print_report(output)

def build_html_report(output: "RadarOutput", csv_path: str = "") -> str:
    weights = indicator_weights()
    sections = grouped_indicator_sections(output.indicators)

    stage_names = ["Normal", "Elevated Risk", "High Risk", "Severe Risk", "Crash Warning"]
    stage_classes = ["stage-normal", "stage-elevated", "stage-high", "stage-severe", "stage-crash"]
    current_stage = risk_stage_index(output.regime)

    trend_symbol, trend_label = trend_symbol_and_label(output.risk_trend)
    trend_compare = trend_comparison_text(output, csv_path)

    stage_bar_html = []
    for idx, (name, cls) in enumerate(zip(stage_names, stage_classes)):
        active_class = " active" if idx == current_stage else ""
        current_badge = '<div class="stage-current">Current</div>' if idx == current_stage else ""
        stage_bar_html.append(
            f"""
            <div class="risk-stage {cls}{active_class}">
                <div class="stage-name">{html.escape(name)}</div>
                {current_badge}
            </div>
            """
        )

    def esc(x: str) -> str:
        return html.escape(str(x))

    top_drivers_html = ""
    if output.top_risk_drivers:
        top_drivers_html = "".join(
            f"<li>{esc(driver)}</li>" for driver in output.top_risk_drivers
        )
    else:
        top_drivers_html = "<li>No material weighted risk drivers detected.</li>"

    section_tables = []

    for section_name, section_indicators in sections:
        rows = []
        for ind in section_indicators:
            weight = weights.get(ind.key, 0.0)
            weight_txt = f"{weight:.1f}" if weight > 0 else "0"

            status_color = html_status_color(ind.status)
            value_txt = format_indicator_value(ind)

            rows.append(
                f"""
                <tr>
                    <td class="mono">{esc(ind.name)}</td>
                    <td class="mono right">{esc(value_txt)}</td>
                    <td class="mono right">{esc(weight_txt)}</td>
                    <td class="mono" style="color:{status_color}; font-weight:700;">{esc(ind.status)}</td>
                    <td>{esc(ind.detail)} <span class="muted">({esc(STATUS_EXPLANATION.get(ind.status, ""))})</span></td>
                </tr>
                """
            )

        section_tables.append(
            f"""
            <section class="card">
                <h2>{esc(section_name)}</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Indicator</th>
                            <th class="right">Value</th>
                            <th class="right">Weight</th>
                            <th>Status</th>
                            <th>Detail</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows)}
                    </tbody>
                </table>
            </section>
            """
        )

    market_dashboard = build_market_stress_dashboard({i.key: i for i in output.indicators})
    macro_dashboard = build_macro_cycle_dashboard({i.key: i for i in output.indicators})
    structural_dashboard = build_structural_vulnerability_dashboard({i.key: i for i in output.indicators})

    def dashboard_rows(items: list[tuple[str, str, str]]) -> str:
        return "".join(
            f"""
            <tr>
                <td>{esc(name)}</td>
                <td>{esc(system)}</td>
                <td style="color:{html_status_color(status)}; font-weight:700;">{esc(status)}</td>
            </tr>
            """
            for name, system, status in items
        )

    regime, crash_view, positioning, explanation = market_summary(output)
    comparison_json = json.dumps(output.comparison, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Macro Crash Radar Report</title>
<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        margin: 24px;
        color: #111827;
        background: #f8fafc;
        line-height: 1.4;
    }}
    h1, h2, h3 {{
        margin: 0 0 10px 0;
    }}
    h1 {{
        font-size: 28px;
    }}
    h2 {{
        font-size: 20px;
        margin-top: 0;
    }}
    .meta {{
        color: #475569;
        margin-bottom: 20px;
    }}
    .card {{
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .summary-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }}

    .compare-toggle {{
        display: inline-flex;
        gap: 8px;
        padding: 4px;
        background: #f1f5f9;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }}
    .summary {{
        font-size: 16px;
        font-weight: 600;
    }}
    .grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 18px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }}
    th, td {{
        padding: 8px 10px;
        border-bottom: 1px solid #e5e7eb;
        vertical-align: top;
        text-align: left;
    }}
    th {{
        background: #f8fafc;
    }}
    .mono {{
        font-family: Consolas, Menlo, Monaco, monospace;
    }}
    .right {{
        text-align: right;
    }}
    .muted {{
        color: #64748b;
    }}
    ul {{
        margin: 8px 0 0 18px;
    }}
    .pill {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #e2e8f0;
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 13px;
    }}
    .small {{
        font-size: 13px;
        color: #475569;
    }}
    .risk-bar {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin-top: 12px;
    }}
    .risk-stage {{
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        border: 2px solid transparent;
        background: #e5e7eb;
        min-height: 72px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }}
    .risk-stage.active {{
        border-color: #111827;
        box-shadow: 0 2px 10px rgba(0,0,0,0.10);
        transform: translateY(-1px);
    }}
    .stage-name {{
        font-weight: 700;
        font-size: 14px;
    }}
    .stage-current {{
        margin-top: 8px;
        font-size: 12px;
        background: rgba(255,255,255,0.8);
        padding: 3px 8px;
        border-radius: 999px;
    }}
    .stage-normal {{
        background: #dcfce7;
        color: #166534;
    }}
    .stage-elevated {{
        background: #fef9c3;
        color: #854d0e;
    }}
    .stage-high {{
        background: #fed7aa;
        color: #9a3412;
    }}
    .stage-severe {{
        background: #fdba74;
        color: #9a3412;
    }}
    .stage-crash {{
        background: #fecaca;
        color: #991b1b;
    }}
    .trend-chip {{
        display: inline-block;
        margin-top: 14px;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 700;
        background: #e2e8f0;
    }}
    .trend-up {{
        color: #b45309;
        background: #fef3c7;
    }}
    .trend-down {{
        color: #166534;
        background: #dcfce7;
    }}
    .trend-flat {{
        color: #334155;
        background: #e2e8f0;
    }}
    .compare-toggle-wrap {{
        margin: 18px 0 10px 0;
    }}
    .compare-toggle {{
        display: inline-flex;
        gap: 8px;
        padding: 4px;
        background: #f1f5f9;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }}
    .compare-btn {{
        border: 0;
        padding: 10px 14px;
        border-radius: 10px;
        cursor: pointer;
        background: transparent;
        font-weight: 600;
        color: #334155;
    }}
    .compare-btn.active {{
        background: #111827;
        color: white;
    }}
    .compare-subtle {{
        margin-top: 8px;
        color: #64748b;
        font-size: 13px;
    }}
    .summary-headline {{
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 6px;
    }}
    .summary-subtitle {{
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
    }}
    .summary-driver-line,
    .summary-system-line {{
        font-size: 14px;
        color: #334155;
        margin-top: 4px;
    }}
    .driver-row {{
        display: grid;
        grid-template-columns: 180px 1fr 90px;
        align-items: center;
        gap: 12px;
        margin: 10px 0;
    }}
    .driver-name {{
        font-weight: 600;
    }}
    .driver-bar-wrap {{
        width: 100%;
        height: 14px;
        background: #e5e7eb;
        border-radius: 999px;
        overflow: hidden;
    }}
    .bar {{
        height: 100%;
        border-radius: 999px;
    }}
    .bar.up {{
        background: #dc2626;
    }}
    .bar.down {{
        background: #16a34a;
    }}
    .bar.flat {{
        background: #94a3b8;
    }}
    .driver-val {{
        text-align: right;
        font-weight: 700;
    }}
    .pos {{
        color: #b91c1c;
        font-weight: 700;
    }}
    .neg {{
        color: #15803d;
        font-weight: 700;
    }}
    .neu {{
        color: #64748b;
    }}
    @media (max-width: 1000px) {{
        .grid {{
            grid-template-columns: 1fr;
        }}
    }}
</style>
</head>
<body>

    <h1>Macro Crash Radar V3</h1>
    <div class="meta">UTC: {esc(output.timestamp_utc)}</div>

    <section class="card">

    <div class="summary-header">
        <h2>Summary</h2>

        <div class="compare-toggle">
            <button id="btn-previous-run" class="compare-btn active" onclick="setCompareMode('previous_run')">
                Compare to previous run
            </button>

            <button id="btn-yesterday" class="compare-btn" onclick="setCompareMode('yesterday')">
                Compare to yesterday
            </button>
        </div>
    </div>

    <div class="compare-subtle">
        Switch the attribution view without changing the main dashboard snapshot.
    </div>

    <div style="margin-top:18px;">
        <div id="summary-headline" class="summary-headline"></div>
        <div id="summary-subtitle" class="summary-subtitle"></div>
        <div id="summary-driver-line" class="summary-driver-line"></div>
        <div id="summary-system-line" class="summary-system-line"></div>
    </div>

</section>

    <section class="card">
        <h2>System Risk Position</h2>
        <div><strong>Current level:</strong> {esc(output.regime)}</div>
        <div><strong>Crash probability:</strong> {esc(output.crash_probability_pct)}%</div>

        <div class="risk-bar">
            {''.join(stage_bar_html)}
        </div>

        <div class="trend-chip {'trend-up' if output.risk_trend in ('rising', 'rising fast') else 'trend-down' if output.risk_trend in ('falling', 'falling fast') else 'trend-flat'}">
            {esc(trend_symbol)} {esc(trend_label)} {esc(trend_compare)}
        </div>
    </section>

    <section class="card" id="drivers-card">
        <h2>Drivers of Change</h2>
        <div id="drivers-bars"></div>
    </section>

     <section class="card">
        <details>
            <summary style="cursor:pointer; font-weight:700; font-size:20px;">
                Detailed Attribution
            </summary>
            <div style="margin-top:14px;">
                <table>
                    <thead>
                        <tr>
                            <th>Indicator</th>
                            <th>System</th>
                            <th class="right">Prev raw</th>
                            <th class="right">Now raw</th>
                            <th class="right">Prev score</th>
                            <th class="right">Now score</th>
                            <th class="right">Weight</th>
                            <th class="right">Contribution</th>
                        </tr>
                    </thead>
                    <tbody id="drivers-table-body"></tbody>
                </table>
            </div>
        </details>
    </section>

    

    <div class="grid">
        <section class="card">
            <h2>Top Risk Drivers</h2>
            <div class="small">Ranked by current risk impact.</div>
            <ul>
                {top_drivers_html}
            </ul>
        </section>

        <section class="card">
            <h2>Risk Snapshot</h2>
            <div class="pill">Total score: {esc(output.total_score)}</div>
            <div class="pill">Crash probability: {esc(output.crash_probability_pct)}%</div>
            <div class="pill">Regime: {esc(output.regime)}</div>
            <div class="pill">Trend: {esc(output.risk_trend)}</div>
            <div class="pill">Crash setup: {esc(output.crash_setup_level)}</div>
            <div class="pill">Calm before storm: {"YES" if output.calm_before_storm else "NO"}</div>
        </section>
    </div>

    <div class="grid">
        <section class="card">
            <h2>Market Stress Dashboard</h2>
            <table>
                <thead>
                    <tr><th>Indicator</th><th>System</th><th>Status</th></tr>
                </thead>
                <tbody>
                    {dashboard_rows(market_dashboard)}
                </tbody>
            </table>
        </section>

        <section class="card">
            <h2>Macro Cycle Indicators</h2>
            <table>
                <thead>
                    <tr><th>Indicator</th><th>System</th><th>Status</th></tr>
                </thead>
                <tbody>
                    {dashboard_rows(macro_dashboard)}
                </tbody>
            </table>
        </section>
    </div>

    <section class="card">
        <h2>Structural Vulnerability</h2>
        <div><strong>Status:</strong> {esc(output.structural_vulnerability_status)}</div>
        <div><strong>Explanation:</strong> {esc(output.structural_vulnerability_explanation)}</div>
        {"<div><strong>Drivers:</strong> " + esc("; ".join(output.structural_vulnerability_drivers)) + "</div>" if output.structural_vulnerability_drivers else ""}
        <br>
        <table>
            <thead>
                <tr><th>Indicator</th><th>System</th><th>Status</th></tr>
            </thead>
            <tbody>
                {dashboard_rows(structural_dashboard)}
            </tbody>
        </table>
    </section>

    <div class="grid">
        <section class="card">
            <h2>Crash Setup Detector</h2>
            <div><strong>Level:</strong> {esc(output.crash_setup_level)}</div>
            <div><strong>Explanation:</strong> {esc(output.crash_setup_explanation)}</div>
            {"<div><strong>Present conditions:</strong> " + esc("; ".join(output.crash_setup_reasons)) + "</div>" if output.crash_setup_reasons else ""}
            {"<div><strong>Missing conditions:</strong> " + esc("; ".join(output.crash_setup_missing)) + "</div>" if output.crash_setup_missing else ""}
        </section>

        <section class="card">
            <h2>System Interpretation</h2>
            <div><strong>Market phase:</strong> {esc(output.market_phase)}</div>
            <div><strong>Bond regime:</strong> {esc(output.bond_regime)}</div>
            <div><strong>Macro regime:</strong> {esc(output.macro_regime)}</div>
            <br>
            <div>{esc(output.system_risk_interpretation)}</div>
        </section>
    </div>

    {''.join(section_tables)}

    <section class="card">
        <h2>Market Interpretation</h2>
        <div><strong>Market regime:</strong> {esc(regime)}</div>
        <div><strong>Crash risk assessment:</strong> {esc(crash_view)}</div>
        <div><strong>Suggested portfolio positioning:</strong> {esc(positioning)}</div>
        <br>
        <div>{esc(explanation)}</div>
    </section>

    <section class="card">
        <h2>Narrative</h2>
        <div>{esc(output.narrative)}</div>
    </section>

    <section class="card">
        <h2>Risk Trend</h2>
        <div>{esc(output.trend_explanation)}</div>
    </section>

   <section class="card">
        <h2>Calm Before the Storm Detector</h2>
        <div><strong>Detected:</strong> {"YES" if output.calm_before_storm else "NO"}</div>
        <div>{esc(output.calm_before_storm_explanation)}</div>
    </section>

    <section class="card">
        <details>
            <summary style="cursor:pointer; font-weight:700; font-size:18px;">
                How this dashboard works
            </summary>
            <div style="margin-top:14px; white-space:pre-wrap;">{esc(output.dashboard_explanation)}</div>
        </details>
    </section>

<script>
const COMPARISON_DATA = {comparison_json};

function fmtNum(x, digits=1) {{
    if (x === null || x === undefined || x === "") return "";
    const n = Number(x);
    if (Number.isNaN(n)) return String(x);
    return n.toFixed(digits);
}}

function fmtSigned(x, digits=1) {{
    const n = Number(x || 0);
    return (n >= 0 ? "+" : "") + n.toFixed(digits);
}}

function setCompareMode(mode) {{
    const data = COMPARISON_DATA[mode];
    if (!data) return;

    document.getElementById("btn-previous-run").classList.toggle("active", mode === "previous_run");
    document.getElementById("btn-yesterday").classList.toggle("active", mode === "yesterday");

    document.getElementById("summary-headline").textContent = data.summary.headline || "";
    document.getElementById("summary-subtitle").textContent = data.summary.subtitle || "";
    document.getElementById("summary-driver-line").textContent = data.summary.driver_line || "";
    document.getElementById("summary-system-line").textContent = data.summary.system_line || "";

    const hasMeaningfulChange = !!data.result.has_meaningful_change;
    document.getElementById("drivers-card").style.display = hasMeaningfulChange ? "block" : "none";

    renderDriverBars(data.result.top_drivers || []);
    renderDriversTable(data.result.rows || []);
}}

function renderDriverBars(rows) {{
    const el = document.getElementById("drivers-bars");

    if (!rows || rows.length === 0) {{
        el.innerHTML = '<div class="muted">No meaningful change to show.</div>';
        return;
    }}

    const maxAbs = Math.max(...rows.map(r => Math.abs(Number(r.contribution || 0))), 0.1);

    const html = rows.map(r => {{
        const val = Number(r.contribution || 0);
        const pct = Math.max(4, Math.round((Math.abs(val) / maxAbs) * 100));
        const cls = val > 0 ? "bar up" : (val < 0 ? "bar down" : "bar flat");

        return `
            <div class="driver-row">
                <div class="driver-name">${{r.label}}</div>
                <div class="driver-bar-wrap">
                    <div class="${{cls}}" style="width:${{pct}}%"></div>
                </div>
                <div class="driver-val">${{fmtSigned(val, 1)}} pts</div>
            </div>
        `;
    }}).join("");

    el.innerHTML = html;
}}

function renderDriversTable(rows) {{
    const tbody = document.getElementById("drivers-table-body");

    if (!rows || rows.length === 0) {{
        tbody.innerHTML = "";
        return;
    }}

    tbody.innerHTML = rows.map(r => `
        <tr>
            <td>${{r.label}}</td>
            <td>${{r.system}}</td>
            <td class="right">${{r.prev_raw ?? ""}}</td>
            <td class="right">${{r.curr_raw ?? ""}}</td>
            <td class="right">${{fmtNum(r.prev_score, 1)}}</td>
            <td class="right">${{fmtNum(r.curr_score, 1)}}</td>
            <td class="right">${{fmtNum(r.weight, 1)}}</td>
            <td class="right ${{Number(r.contribution) > 0 ? 'pos' : (Number(r.contribution) < 0 ? 'neg' : 'neu')}}">
                ${{fmtSigned(r.contribution, 1)}}
            </td>
        </tr>
    `).join("");
}}

setCompareMode(COMPARISON_DATA.default_mode || "previous_run");
</script>

</body>
</html>
"""



# =========================
# Data models
# =========================

@dataclass
class Indicator:
    key: str
    name: str
    value: float | None
    status: str
    score: int
    detail: str


@dataclass
class RadarOutput:
    timestamp_utc: str
    total_score: int
    crash_probability_pct: int
    regime: str
    hedge_fund_trigger: bool
    trigger_reasons: list[str]
    narrative: str
    indicators: list[Indicator]

    crash_setup_level: str
    crash_setup_reasons: list[str]
    crash_setup_missing: list[str]
    crash_setup_explanation: str

    market_phase: str
    market_phase_explanation: str
    market_phase_drivers: list[str]

    bond_regime: str
    bond_regime_explanation: str
    bond_regime_drivers: list[str]

    trend_score: int | None
    risk_trend: str
    trend_explanation: str

    macro_cycle_status: str
    macro_cycle_explanation: str
    macro_cycle_drivers: list[str]

    macro_regime: str
    macro_regime_explanation: str
    macro_regime_drivers: list[str]

    structural_vulnerability_status: str
    structural_vulnerability_explanation: str
    structural_vulnerability_drivers: list[str]

    system_risk_interpretation: str

    calm_before_storm: bool
    calm_before_storm_explanation: str

    top_risk_drivers: list[str]
    summary_line: str
    dashboard_explanation: str
    comparison: dict[str, Any]

@dataclass
class CrashSetupResult:
    active: bool
    level: str
    reasons: list[str]
    missing: list[str]
    explanation: str

# =========================
# Helpers
# =========================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def latest(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        raise ValueError("Series is empty")
    return float(s.iloc[-1])


def value_days_ago(series: pd.Series, days_back: int) -> float:
    s = series.dropna()
    if s.empty:
        raise ValueError("Series is empty")
    idx = max(0, len(s) - 1 - days_back)
    return float(s.iloc[idx])


def pct_change(current: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (current - old) / old * 100.0


def pp_change(current: float, old: float) -> float:
    return current - old


def fred_series(series_id: str, limit: int = 500) -> pd.Series:
    if not FRED_API_KEY:
        raise RuntimeError(
            "FRED_API_KEY is not set.\n"
            "PowerShell: $env:FRED_API_KEY='your_key'\n"
            "CMD: set FRED_API_KEY=your_key"
        )

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",   # newest first
        "limit": limit,
    }

    response = requests.get(FRED_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    rows: list[tuple[pd.Timestamp, float]] = []
    for obs in payload.get("observations", []):
        raw_value = obs.get("value")
        if raw_value in (None, "", "."):
            continue
        rows.append((pd.to_datetime(obs["date"]), float(raw_value)))

    if not rows:
        raise RuntimeError(f"No usable FRED data returned for {series_id}")

    rows.sort(key=lambda x: x[0])  # old -> new
    dates, values = zip(*rows)
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)


def yf_close_series(ticker: str, period: str = "2y", interval: str = "1d") -> pd.Series:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise RuntimeError(f"No Yahoo Finance data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            close = df[("Close", ticker)].dropna()
        else:
            close = df.xs("Close", axis=1, level=0).iloc[:, 0].dropna()
    else:
        close = df["Close"].dropna()

    if close.empty:
        raise RuntimeError(f"No close data returned for {ticker}")

    close.name = ticker
    return close


def color_for_status(status: str) -> str:
    if status == "ALARM":
        return ANSI["red"]
    if status in ("WARNING", "ELEVATED"):
        return ANSI["yellow"]
    if status == "UNAVAILABLE":
        return ANSI["cyan"]
    return ANSI["green"]


def safe_ma(series: pd.Series, n: int) -> float:
    s = series.dropna()
    if s.empty:
        raise ValueError("Series is empty")
    if len(s) >= n:
        return float(s.tail(n).mean())
    return float(s.mean())

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def append_history_snapshot(path: Path, snapshot: dict[str, Any], keep_last: int = 500) -> None:
    history = load_history(path)
    history.append(snapshot)
    history = history[-keep_last:]
    save_history(path, history)


def get_previous_run_snapshot(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    return history[-1]


def get_yesterday_snapshot(history: list[dict[str, Any]], current_ts_iso: str) -> dict[str, Any] | None:
    """
    Returns the latest snapshot from the previous calendar day (UTC).
    If not found, falls back to the latest snapshot older than ~20 hours.
    """
    if not history:
        return None

    current_dt = datetime.fromisoformat(current_ts_iso)
    current_date = current_dt.date()
    target_date = current_date.fromordinal(current_date.toordinal() - 1)

    candidates_prev_day = []
    older_candidates = []

    for item in history:
        ts = item.get("timestamp")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue

        if dt.date() == target_date:
            candidates_prev_day.append(item)

        hours_old = (current_dt - dt).total_seconds() / 3600.0
        if hours_old >= 20:
            older_candidates.append((hours_old, item))

    if candidates_prev_day:
        candidates_prev_day.sort(key=lambda x: x.get("timestamp", ""))
        return candidates_prev_day[-1]

    if older_candidates:
        older_candidates.sort(key=lambda x: x[0])
        return older_candidates[0][1]

    return None


def build_history_snapshot(
    *,
    crash_score: float,
    indicator_values: dict[str, Any],
    indicator_risk_scores: dict[str, float],
    stage_label: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": utc_now_iso(),
        "crash_score": round(safe_float(crash_score), 2),
        "indicator_values": indicator_values,
        "indicator_risk_scores": {k: round(safe_float(v), 4) for k, v in indicator_risk_scores.items()},
        "stage_label": stage_label or "",
    }


def calculate_driver_contributions(
    current_snapshot: dict[str, Any],
    baseline_snapshot: dict[str, Any] | None,
    indicator_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Uses existing indicator risk scores for attribution.
    Contribution = (current_indicator_risk_score - baseline_indicator_risk_score) * weight
    """

    baseline_available = baseline_snapshot is not None

    current_total = safe_float(current_snapshot.get("crash_score"))

    if baseline_available:
        baseline_total = safe_float(baseline_snapshot.get("crash_score"))
        score_delta = current_total - baseline_total
    else:
        baseline_total = None
        score_delta = None

    current_values = current_snapshot.get("indicator_values", {}) or {}
    baseline_values = (baseline_snapshot or {}).get("indicator_values", {}) or {}

    current_scores = current_snapshot.get("indicator_risk_scores", {}) or {}
    baseline_scores = (baseline_snapshot or {}).get("indicator_risk_scores", {}) or {}

    rows = []
    system_contribs: dict[str, float] = {}

    all_keys = sorted(set(current_scores.keys()) | set(baseline_scores.keys()) | set(indicator_meta.keys()))

    for key in all_keys:
        meta = indicator_meta.get(key, {})
        label = meta.get("label", key)
        system = meta.get("system", "Other")
        weight = safe_float(meta.get("weight"), 0.0)

        prev_raw = baseline_values.get(key) if baseline_available else None
        curr_raw = current_values.get(key)

        prev_score = safe_float(baseline_scores.get(key), 0.0) if baseline_available else None
        curr_score = safe_float(current_scores.get(key), 0.0)

        if baseline_available:
            risk_score_delta = curr_score - safe_float(prev_score, 0.0)
            contribution = risk_score_delta * weight
        else:
            risk_score_delta = None
            contribution = None

        if contribution is None:
            direction = "flat"
        elif contribution > 0:
            direction = "up"
        elif contribution < 0:
            direction = "down"
        else:
            direction = "flat"

        row = {
            "key": key,
            "label": label,
            "system": system,
            "weight": round(weight, 4),
            "prev_raw": prev_raw,
            "curr_raw": curr_raw,
            "prev_score": round(prev_score, 2) if prev_score is not None else None,
            "curr_score": round(curr_score, 2),
            "risk_score_delta": round(risk_score_delta, 2) if risk_score_delta is not None else None,
            "contribution": round(contribution, 2) if contribution is not None else None,
            "direction": direction,
        }
        rows.append(row)

        if contribution is not None:
            system_contribs[system] = system_contribs.get(system, 0.0) + contribution

    if baseline_available:
        rows.sort(key=lambda x: abs(safe_float(x["contribution"])), reverse=True)
    else:
        rows.sort(key=lambda x: x["label"])

    system_rows = [
        {"system": k, "contribution": round(v, 2)}
        for k, v in system_contribs.items()
    ]
    system_rows.sort(key=lambda x: abs(safe_float(x["contribution"])), reverse=True)

    meaningful_rows = [
        r for r in rows
        if r["contribution"] is not None and abs(safe_float(r["contribution"])) >= CONTRIBUTION_EPSILON
    ]

    if not baseline_available:
        summary_state = "unavailable"
    elif abs(safe_float(score_delta)) < SCORE_CHANGE_EPSILON:
        summary_state = "stable"
    elif safe_float(score_delta) > 0:
        summary_state = "up"
    else:
        summary_state = "down"

    top_driver = meaningful_rows[0] if meaningful_rows else None
    top_system = (
        system_rows[0]
        if system_rows and abs(safe_float(system_rows[0]["contribution"])) >= CONTRIBUTION_EPSILON
        else None
    )

    return {
        "baseline_available": baseline_available,
        "score_current": round(current_total, 2),
        "score_baseline": round(baseline_total, 2) if baseline_total is not None else None,
        "score_delta": round(score_delta, 2) if score_delta is not None else None,
        "summary_state": summary_state,
        "top_driver": top_driver,
        "top_drivers": meaningful_rows[:MAX_TOP_DRIVERS],
        "top_system": top_system,
        "rows": rows,
        "system_rows": system_rows,
        "has_meaningful_change": baseline_available and abs(safe_float(score_delta)) >= SCORE_CHANGE_EPSILON,
        "baseline_timestamp": (baseline_snapshot or {}).get("timestamp"),
    }


def fmt_signed(x: float, digits: int = 1) -> str:
    x = safe_float(x)
    return f"{x:+.{digits}f}"

def build_clean_summary_line(compare_result: dict[str, Any], mode_label: str) -> str:
    score_now = safe_float(compare_result.get("score_current"))
    score_delta = compare_result.get("score_delta")
    state = compare_result.get("summary_state", "stable")
    baseline_available = compare_result.get("baseline_available", False)
    top_driver = compare_result.get("top_driver")
    top_drivers = compare_result.get("top_drivers", [])

    line1 = f"Crash Risk Score: {score_now:.0f}%"

    if not baseline_available or state == "unavailable":
        line2 = f"No comparison available vs {mode_label}"
        line3 = "This is the first stored snapshot for this comparison mode."
    elif state == "stable":
        line2 = f"Stable vs {mode_label}"
        line3 = "No material change in weighted risk indicators."
    else:
        direction_word = "Up" if safe_float(score_delta) > 0 else "Down"
        line2 = f"{direction_word} {abs(safe_float(score_delta)):.1f} pts vs {mode_label}"

        if top_driver:
            if len(top_drivers) > 1:
                others = ", ".join(r["label"] for r in top_drivers[1:])
                line3 = f"Main driver: {top_driver['label']}. Secondary drivers: {others}."
            else:
                line3 = f"Main driver: {top_driver['label']}."
        else:
            line3 = "Change driven by small moves across weighted indicators."

    return f"{line1} | {line2} | {line3}"


def calculate_trend_from_snapshot(
    current_snapshot: dict[str, Any],
    baseline_snapshot: dict[str, Any] | None,
) -> tuple[float | None, str, str]:
    """
    Compare current crash score vs baseline snapshot from JSON history.
    Returns:
    - trend_score: delta in crash score
    - risk_trend: rising / rising fast / stable / falling / falling fast / unavailable
    - explanation
    """
    if not baseline_snapshot:
        return None, "unavailable", "No previous snapshot available in JSON history."

    current_score = safe_float(current_snapshot.get("crash_score"))
    baseline_score = safe_float(baseline_snapshot.get("crash_score"))
    delta = current_score - baseline_score

    if delta >= 8:
        trend = "rising fast"
        explanation = "Market risk is increasing quickly versus the previous stored snapshot."
    elif delta >= 3:
        trend = "rising"
        explanation = "Market risk is increasing versus the previous stored snapshot."
    elif delta <= -8:
        trend = "falling fast"
        explanation = "Market risk is improving quickly versus the previous stored snapshot."
    elif delta <= -3:
        trend = "falling"
        explanation = "Market risk is easing versus the previous stored snapshot."
    else:
        trend = "stable"
        explanation = "Market risk is broadly stable versus the previous stored snapshot."

    return round(delta, 2), trend, explanation    

def format_summary_text(compare_result: dict[str, Any], mode_label: str) -> dict[str, str]:
    score_now = safe_float(compare_result.get("score_current"))
    score_delta = compare_result.get("score_delta")
    state = compare_result.get("summary_state", "stable")
    top_driver = compare_result.get("top_driver")
    top_drivers = compare_result.get("top_drivers", [])
    top_system = compare_result.get("top_system")
    baseline_available = compare_result.get("baseline_available", False)

    headline = f"Crash Risk Score: {score_now:.0f}%"

    if not baseline_available or state == "unavailable":
        subtitle = f"No comparison available vs {mode_label}"
        driver_line = "This is the first stored snapshot for this comparison mode."
        system_line = ""
    elif state == "stable":
        subtitle = f"Stable vs {mode_label}"
        driver_line = "No material change in weighted risk indicators."
        system_line = ""
    else:
        direction_word = "Up" if safe_float(score_delta) > 0 else "Down"
        subtitle = f"{direction_word} {abs(safe_float(score_delta)):.1f} pts vs {mode_label}"

        if top_driver:
            if len(top_drivers) > 1:
                secondary = ", ".join(r["label"] for r in top_drivers[1:])
                driver_line = f"Main driver: {top_driver['label']} · Secondary: {secondary}"
            else:
                driver_line = f"Main driver: {top_driver['label']}"
        else:
            driver_line = "Change driven by smaller moves across weighted indicators."

        if top_system:
            system_line = f"System most responsible: {top_system['system']}"
        else:
            system_line = ""

    return {
        "headline": headline,
        "subtitle": subtitle,
        "driver_line": driver_line,
        "system_line": system_line,
    }


# =========================
# Indicator analysis
# =========================

def analyze_credit_spread(series: pd.Series) -> Indicator:
    cur = latest(series)
    prev_5 = value_days_ago(series, 5)
    delta_5 = pp_change(cur, prev_5)

    if cur >= 6.5 or delta_5 >= 1.0:
        status, score = "ALARM", 4
    elif cur >= 5.5 or delta_5 >= 0.5:
        status, score = "WARNING", 3
    elif cur >= 4.5:
        status, score = "ELEVATED", 1
    else:
        status, score = "OK", 0

    return Indicator(
        key="credit_spread",
        name="High Yield Credit Spread",
        value=cur,
        status=status,
        score=score,
        detail=f"{cur:.2f}% | 5d change {delta_5:+.2f} pp",
    )


def analyze_vix(series: pd.Series) -> Indicator:
    cur = latest(series)
    prev_5 = value_days_ago(series, 5)
    chg_pct = pct_change(cur, prev_5)

    if cur >= 30:
        status, score = "ALARM", 4
    elif cur >= 23 or chg_pct >= 25:
        status, score = "WARNING", 2
    elif cur >= 20:
        status, score = "ELEVATED", 1
    else:
        status, score = "OK", 0

    return Indicator(
        key="vix",
        name="VIX",
        value=cur,
        status=status,
        score=score,
        detail=f"{cur:.2f} | 5d change {chg_pct:+.1f}%",
    )


def analyze_vol_regime(vix_series: pd.Series) -> Indicator:
    s = vix_series.dropna()
    cur = float(s.iloc[-1])
    ma20 = safe_ma(s, 20)
    ma50 = safe_ma(s, 50)

    if cur >= 25 and cur > ma20 and cur > ma50:
        status, score = "WARNING", 2
    elif cur >= 20 or cur > ma20:
        status, score = "ELEVATED", 1
    else:
        status, score = "OK", 0

    return Indicator(
        key="vol_regime",
        name="Vol Regime Shift",
        value=cur,
        status=status,
        score=score,
        detail=f"VIX {cur:.2f} | MA20 {ma20:.2f}, MA50 {ma50:.2f}",
    )


def analyze_move(series: pd.Series) -> Indicator:
    cur = latest(series)
    prev_5 = value_days_ago(series, 5)
    chg_pct = pct_change(cur, prev_5)

    if cur >= 140:
        status, score = "ALARM", 3
    elif cur >= 120:
        status, score = "WARNING", 2
    elif cur >= 100 or chg_pct >= 15:
        status, score = "ELEVATED", 1
    else:
        status, score = "OK", 0

    return Indicator(
        key="move",
        name="MOVE",
        value=cur,
        status=status,
        score=score,
        detail=f"{cur:.2f} | 5d change {chg_pct:+.1f}%",
    )


def analyze_yield_curve(us10y: pd.Series, us2y: pd.Series) -> Indicator:
    cur10 = latest(us10y)
    cur2 = latest(us2y)
    curve = cur10 - cur2

    if curve <= -0.50:
        status, score = "WARNING", 2
    elif curve < 0:
        status, score = "ELEVATED", 1
    else:
        status, score = "OK", 0

    return Indicator(
        key="yield_curve_2s10s",
        name="Yield Curve 2s10s",
        value=curve,
        status=status,
        score=score,
        detail=f"10Y {cur10:.2f}% - 2Y {cur2:.2f}% = {curve:+.2f} pp",
    )


def analyze_fed_balance(series: pd.Series) -> Indicator:
    cur = latest(series)
    prev_12w = value_days_ago(series, 12)
    chg_pct = pct_change(cur, prev_12w)

    if chg_pct <= -3.0:
        status, score = "WARNING", 2
    elif chg_pct < 0:
        status, score = "ELEVATED", 1
    else:
        status, score = "OK", 0

    return Indicator(
        key="fed_balance",
        name="Fed Balance Sheet",
        value=cur,
        status=status,
        score=score,
        detail=f"${cur/1e6:.2f}T | ~12w change {chg_pct:+.2f}%",
    )


def analyze_global_liquidity(fed_series: pd.Series, m2_series: pd.Series) -> Indicator:
    fed_cur = latest(fed_series)
    fed_prev = value_days_ago(fed_series, 12)
    fed_chg = pct_change(fed_cur, fed_prev)

    m2_cur = latest(m2_series)
    m2_prev = value_days_ago(m2_series, 6)
    m2_chg = pct_change(m2_cur, m2_prev)

    if fed_chg < 0 and m2_chg < 0:
        status, score = "WARNING", 2
    elif fed_chg < 0 or m2_chg < 0:
        status, score = "ELEVATED", 1
    else:
        status, score = "OK", 0

    return Indicator(
        key="global_liquidity",
        name="Global Liquidity",
        value=fed_chg + m2_chg,
        status=status,
        score=score,
        detail=f"Fed ~12w {fed_chg:+.2f}% | M2 ~6m {m2_chg:+.2f}%",
    )


def analyze_usd_index(series: pd.Series, label: str = "USD Index") -> Indicator:
    s = series.dropna()
    cur = float(s.iloc[-1])
    ma20 = safe_ma(s, 20)
    ma50 = safe_ma(s, 50)
    ma200 = safe_ma(s, 200)

    # Less aggressive:
    # WARNING only if USD is above all major averages AND meaningfully strong in absolute level
    if cur > 100 and cur > ma20 and cur > ma50 and cur > ma200:
        status, score = "WARNING", 2
    elif cur > ma50 or cur > ma200:
        status, score = "ELEVATED", 1
    else:
        status, score = "OK", 0

    return Indicator(
        key="usd_index",
        name=label,
        value=cur,
        status=status,
        score=score,
        detail=f"{cur:.2f} | MA20 {ma20:.2f}, MA50 {ma50:.2f}, MA200 {ma200:.2f}",
    )


def analyze_hyg(series: pd.Series) -> Indicator:
    s = series.dropna()
    cur = float(s.iloc[-1])
    ma20 = safe_ma(s, 20)
    ma50 = safe_ma(s, 50)
    ma200 = safe_ma(s, 200)

    if cur < ma200 and cur < ma50:
        status, score = "WARNING", 3
    elif cur < ma50 or cur < ma20:
        status, score = "ELEVATED", 1
    else:
        status, score = "OK", 0

    return Indicator(
        key="hyg",
        name="HYG Trend",
        value=cur,
        status=status,
        score=score,
        detail=f"{cur:.2f} | MA20 {ma20:.2f}, MA50 {ma50:.2f}, MA200 {ma200:.2f}",
    )


def analyze_sp_breadth(rsp_series: pd.Series, spy_series: pd.Series) -> Indicator:
    aligned = pd.concat([rsp_series, spy_series], axis=1).dropna()
    aligned.columns = ["rsp", "spy"]
    ratio = aligned["rsp"] / aligned["spy"]

    cur = float(ratio.iloc[-1])
    ma20 = safe_ma(ratio, 20)
    ma50 = safe_ma(ratio, 50)
    ma200 = safe_ma(ratio, 200)

    if cur < ma50 and cur < ma200:
        status, score = "WARNING", 2
    elif cur < ma20 or cur < ma50:
        status, score = "ELEVATED", 1
    else:
        status, score = "OK", 0

    return Indicator(
        key="sp_breadth",
        name="S&P Breadth",
        value=cur,
        status=status,
        score=score,
        detail=f"RSP/SPY {cur:.4f} | MA20 {ma20:.4f}, MA50 {ma50:.4f}, MA200 {ma200:.4f}",
    )


def analyze_nasdaq(series: pd.Series) -> Indicator:
    s = series.dropna()
    cur = float(s.iloc[-1])
    ma20 = safe_ma(s, 20)
    ma50 = safe_ma(s, 50)
    ma200 = safe_ma(s, 200)

    if cur < ma200 and cur < ma50:
        status, score = "WARNING", 2
    elif cur < ma50 or cur < ma20:
        status, score = "ELEVATED", 1
    else:
        status, score = "OK", 0

    return Indicator(
        key="nasdaq",
        name="Nasdaq Trend",
        value=cur,
        status=status,
        score=score,
        detail=f"{cur:.2f} | MA20 {ma20:.2f}, MA50 {ma50:.2f}, MA200 {ma200:.2f}",
    )

def analyze_inflation_trend(series: pd.Series) -> Indicator:
    s = series.dropna()
    cur = float(s.iloc[-1])

    # Use 12 months ago for YoY proxy if enough data exists
    prev_12m = float(s.iloc[-13]) if len(s) >= 13 else float(s.iloc[0])
    yoy = ((cur - prev_12m) / prev_12m) * 100 if prev_12m != 0 else 0.0

    if yoy >= 5.0:
        status, score = "ALARM", 0
    elif yoy >= 3.0:
        status, score = "WARNING", 0
    elif yoy >= 2.0:
        status, score = "ELEVATED", 0
    else:
        status, score = "OK", 0

    return Indicator(
        key="inflation_trend",
        name="Inflation Trend",
        value=yoy,
        status=status,
        score=score,
        detail=f"CPI YoY {yoy:.2f}%",
    )

def analyze_oil_shock(series: pd.Series) -> Indicator:
    s = series.dropna()
    cur = float(s.iloc[-1])
    prev_30 = float(s.iloc[-31]) if len(s) >= 31 else float(s.iloc[0])
    chg_30d = ((cur - prev_30) / prev_30) * 100 if prev_30 != 0 else 0.0

    if chg_30d >= 50:
        status, score = "ALARM", 0
    elif chg_30d >= 25:
        status, score = "WARNING", 0
    elif chg_30d >= 10:
        status, score = "ELEVATED", 0
    else:
        status, score = "OK", 0

    return Indicator(
        key="oil_shock",
        name="Oil Shock",
        value=chg_30d,
        status=status,
        score=score,
        detail=f"Brent 30d change {chg_30d:+.2f}%",
    )

def analyze_macro_global_liquidity(fed_series: pd.Series, m2_series: pd.Series, usd_series: pd.Series) -> Indicator:
    fed_cur = latest(fed_series)
    fed_prev = value_days_ago(fed_series, 12)
    fed_chg = pct_change(fed_cur, fed_prev)

    m2_cur = latest(m2_series)
    m2_prev = value_days_ago(m2_series, 6)
    m2_chg = pct_change(m2_cur, m2_prev)

    usd_cur = latest(usd_series)
    usd_ma50 = safe_ma(usd_series, 50)

    # Strong USD is a liquidity tightening proxy
    usd_tight = usd_cur > usd_ma50

    if fed_chg < 0 and m2_chg < 0 and usd_tight:
        status, score = "WARNING", 0
    elif fed_chg < 0 or m2_chg < 0 or usd_tight:
        status, score = "ELEVATED", 0
    else:
        status, score = "OK", 0

    return Indicator(
        key="macro_global_liquidity",
        name="Global Liquidity Context",
        value=fed_chg + m2_chg,
        status=status,
        score=score,
        detail=f"Fed ~12w {fed_chg:+.2f}% | M2 ~6m {m2_chg:+.2f}% | USD vs MA50 {usd_cur:.2f}/{usd_ma50:.2f}",
    )

    

def analyze_debt_burden(fed_balance_series: pd.Series, m2_series: pd.Series) -> Indicator:
    """
    Placeholder structural indicator.
    Uses relative balance-sheet expansion / liquidity backdrop as a rough proxy
    until a better debt-series is connected.
    Context only: score = 0
    """
    fed_cur = latest(fed_balance_series)
    fed_prev = value_days_ago(fed_balance_series, 52)
    fed_chg = pct_change(fed_cur, fed_prev)

    m2_cur = latest(m2_series)
    m2_prev = value_days_ago(m2_series, 12)
    m2_chg = pct_change(m2_cur, m2_prev)

    proxy = fed_chg + m2_chg

    if proxy >= 12:
        status = "WARNING"
    elif proxy >= 4:
        status = "ELEVATED"
    else:
        status = "OK"

    return Indicator(
        key="debt_burden",
        name="Debt Burden",
        value=proxy,
        status=status,
        score=0,
        detail=f"Proxy based on balance-sheet expansion: {proxy:+.2f}",
    )


def analyze_valuation_stretch(nasdaq_series: pd.Series, spy_series: pd.Series) -> Indicator:
    """
    Placeholder structural valuation indicator.
    Uses distance vs long-term averages as a simple market-stretch proxy.
    Context only: score = 0
    """
    spy_cur = latest(spy_series)
    spy_ma200 = safe_ma(spy_series, 200)
    spy_stretch = pct_change(spy_cur, spy_ma200)

    nas_cur = latest(nasdaq_series)
    nas_ma200 = safe_ma(nasdaq_series, 200)
    nas_stretch = pct_change(nas_cur, nas_ma200)

    proxy = (spy_stretch + nas_stretch) / 2

    if proxy >= 15:
        status = "WARNING"
    elif proxy >= 7:
        status = "ELEVATED"
    else:
        status = "OK"

    return Indicator(
        key="valuation_stretch",
        name="Valuation Stretch",
        value=proxy,
        status=status,
        score=0,
        detail=f"Proxy vs 200d averages: SPY {spy_stretch:+.1f}%, Nasdaq {nas_stretch:+.1f}%",
    )


def analyze_credit_system_fragility(
    credit_spread_series: pd.Series,
    hyg_series: pd.Series,
    usd_series: pd.Series,
) -> Indicator:
    """
    Placeholder for banking/private credit fragility.
    Uses combination of HY weakness + spread pressure + USD tightness.
    Context only: score = 0
    """
    spread_cur = latest(credit_spread_series)

    hyg_cur = latest(hyg_series)
    hyg_ma200 = safe_ma(hyg_series, 200)
    hyg_weak = hyg_cur < hyg_ma200

    usd_cur = latest(usd_series)
    usd_ma200 = safe_ma(usd_series, 200)
    usd_tight = usd_cur > usd_ma200

    if spread_cur >= 5.5 and hyg_weak and usd_tight:
        status = "WARNING"
    elif spread_cur >= 4.5 or hyg_weak or usd_tight:
        status = "ELEVATED"
    else:
        status = "OK"

    return Indicator(
        key="credit_system_fragility",
        name="Credit System Fragility",
        value=spread_cur,
        status=status,
        score=0,
        detail=f"HY spread {spread_cur:.2f}% | HYG vs 200d {'weak' if hyg_weak else 'stable'} | USD vs 200d {'tight' if usd_tight else 'normal'}",
    )


# =========================
# Trigger / scoring
# =========================

def hedge_fund_trigger(indicators: dict[str, Indicator], hyg_series: pd.Series) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if indicators["vix"].value is not None and indicators["vix"].value >= 30:
        reasons.append("VIX spike > 30")

    if indicators["credit_spread"].value is not None and indicators["credit_spread"].value >= 6.0:
        reasons.append("Credit spread > 6")

    s = hyg_series.dropna()
    cur = float(s.iloc[-1])
    ma200 = safe_ma(s, 200)
    if cur < ma200:
        reasons.append("HYG below 200d MA")

    return len(reasons) >= 2, reasons

def weighted_total_score(indicators: list[Indicator]) -> int:
    weights = indicator_weights()

    total = 0.0
    for ind in indicators:
        w = weights.get(ind.key, 0.0)
        total += ind.score * w

    return int(round(total))

def crash_setup_detector(indicators: dict[str, Indicator]) -> CrashSetupResult:
    """
    V3 crash setup detector with more stable logic.

    It looks for 3 pillars that often align before major crashes:
    1. Credit deterioration
    2. Volatility stress
    3. Liquidity / funding stress

    Improvement:
    - Distinguishes between early credit-sensitive weakness (HYG)
      and true credit market stress (spreads).
    - Requires stronger evidence before escalating to YES.
    """

    reasons: list[str] = []
    missing: list[str] = []

    # -------------------------
    # 1) Credit pillar
    # -------------------------
    credit_spread_warning = indicators["credit_spread"].status in ("WARNING", "ALARM")
    hyg_warning = indicators["hyg"].status in ("WARNING", "ALARM")

    # Stronger credit stress only when spreads are actually stressed
    true_credit_stress = credit_spread_warning

    # Early warning credit weakness when HYG is weak but spreads are still calm
    early_credit_weakness = hyg_warning and not credit_spread_warning

    if true_credit_stress:
        reasons.append("Credit market stress rising")
    elif early_credit_weakness:
        reasons.append("Credit-sensitive markets weakening")
    else:
        missing.append("No meaningful credit stress")

    # -------------------------
    # 2) Volatility pillar
    # -------------------------
    vix_stress = indicators["vix"].status in ("WARNING", "ALARM")
    vol_regime_stress = indicators["vol_regime"].status in ("ELEVATED", "WARNING", "ALARM")
    move_stress = indicators["move"].status in ("WARNING", "ALARM")

    true_vol_stress = move_stress or (vix_stress and vol_regime_stress)

    if true_vol_stress:
        reasons.append("Volatility regime turning risk-off")
    else:
        missing.append("No broad volatility shock")

    # -------------------------
    # 3) Liquidity / funding pillar
    # -------------------------
    liquidity_warning = indicators["global_liquidity"].status in ("WARNING", "ALARM")
    fed_warning = indicators["fed_balance"].status in ("WARNING",)
    usd_warning = indicators["usd_index"].status in ("WARNING",)

    true_liquidity_stress = liquidity_warning or fed_warning or usd_warning

    if true_liquidity_stress:
        reasons.append("Liquidity / funding backdrop worsening")
    else:
        missing.append("No clear liquidity stress")

    # -------------------------
    # Decision logic
    # -------------------------
    strong_pillars = 0
    if true_credit_stress:
        strong_pillars += 1
    if true_vol_stress:
        strong_pillars += 1
    if true_liquidity_stress:
        strong_pillars += 1

    # Early setup if HYG is weak + volatility stress, but spreads not yet stressed
    early_setup = early_credit_weakness and true_vol_stress

    if strong_pillars == 3:
        active = True
        level = "YES"
        explanation = (
            "Crash setup detected: credit, volatility and liquidity conditions are all worsening together."
        )

    elif strong_pillars == 2:
        active = False
        level = "WATCH"
        explanation = (
            "Partial crash setup: two of the three core crash conditions are present. "
            "Risk is rising, but setup is not yet complete."
        )

    elif strong_pillars == 1 and early_setup:
        active = False
        level = "EARLY"
        explanation = (
            "Early crash setup signal: volatility stress is visible and credit-sensitive markets are weakening, "
            "but broad credit stress is not yet confirmed."
        )

    elif strong_pillars == 1:
        active = False
        level = "EARLY"
        explanation = (
            "Only one core crash condition is present. This is an early warning, not a confirmed crash setup."
        )

    else:
        active = False
        level = "NO"
        explanation = (
            "No crash setup detected. Core systemic crash conditions are not aligned."
        )

    return CrashSetupResult(
        active=active,
        level=level,
        reasons=reasons,
        missing=missing,
        explanation=explanation,
    )


def calculate_probability(indicators: list[Indicator], trigger_on: bool) -> int:
    weighted_score = weighted_total_score(indicators)

    # More conservative than v2:
    # - ignores minor technical weakness unless it clusters
    # - only escalates hard when core credit/vol/liquidity also weaken
    base = 100 / (1 + math.exp(-(weighted_score - 11) / 3.2))

    if trigger_on:
        base += 10

    return int(round(clamp(base, 0, 100)))


def regime_from_probability(prob: int) -> str:
    if prob >= 80:
        return "CRASH WARNING"
    if prob >= 65:
        return "SEVERE RISK"
    if prob >= 50:
        return "HIGH RISK"
    if prob >= 30:
        return "ELEVATED RISK"
    return "NORMAL"


def narrative(indicators: list[Indicator], regime: str, trigger_on: bool, reasons: list[str]) -> str:
    bad = [i.name for i in indicators if i.status in ("ALARM", "WARNING")]
    elevated = [i.name for i in indicators if i.status == "ELEVATED"]

    core_bad = [
        i.name for i in indicators
        if i.key in ("credit_spread", "move", "hyg", "vix", "global_liquidity")
        and i.status in ("ALARM", "WARNING")
    ]

    parts: list[str] = []

    if regime == "CRASH WARNING":
        parts.append("Several core markets are showing stress at the same time. This is a pronounced risk-off regime.")
    elif regime == "HIGH RISK":
        parts.append("There are multiple meaningful stress signals. The market is vulnerable to a sharp correction.")
    elif regime == "ELEVATED RISK":
        parts.append("There are early warning signals, but not yet a broad crisis picture.")
    else:
        parts.append("The main macro indicators still look relatively calm for now.")

    if core_bad:
        parts.append("Main core stress signals: " + ", ".join(core_bad) + ".")
    elif bad:
        parts.append("Main stress points: " + ", ".join(bad) + ".")

    if elevated:
        parts.append("Elevated attention for: " + ", ".join(elevated) + ".")

    if trigger_on:
        parts.append("Hedge-fund crash trigger is active: " + "; ".join(reasons) + ".")

    return " ".join(parts)


# =========================
# Output helpers
# =========================

def market_summary(output: RadarOutput) -> tuple[str, str, str, str]:
    """
    Creates a plain-language interpretation of the macro environment.
    """

    prob = output.crash_probability_pct

    if prob < 30:
        regime = "Stable risk environment"
        crash_view = "low crash risk"
        positioning = "normal risk exposure"
        explanation = (
            "Markets are stable and most macro indicators show normal conditions. "
            "Volatility, credit markets and liquidity do not show signs of systemic stress."
        )

    elif prob < 50:
        regime = "Early risk-off signals"
        crash_view = "moderate crash risk"
        positioning = "balanced positioning"
        explanation = (
            "Some early warning signals are appearing. "
            "Investors are becoming slightly more cautious, but financial conditions "
            "are still stable and systemic risk remains low."
        )

    elif prob < 65:
        regime = "Risk-off environment developing"
        crash_view = "elevated crash risk"
        positioning = "slightly defensive positioning"
        explanation = (
            "Multiple market indicators show rising stress. "
            "Volatility is increasing and risk assets are weakening. "
            "However, key systemic indicators such as credit spreads and bond volatility "
            "remain stable, which means a full financial crisis is not yet indicated."
        )

    elif prob < 80:
        regime = "Severe risk environment"
        crash_view = "high crash risk"
        positioning = "defensive positioning"
        explanation = (
            "Market stress is becoming broad and persistent. "
            "Volatility, credit conditions or liquidity indicators are deteriorating together. "
            "Historically this type of environment often precedes major corrections or market shocks."
        )

    else:
        regime = "Systemic stress environment"
        crash_view = "very high crash risk"
        positioning = "capital preservation"
        explanation = (
            "Several core financial indicators show systemic stress. "
            "Volatility, credit markets and liquidity conditions suggest that markets "
            "may be entering a crisis phase."
        )

    return regime, crash_view, positioning, explanation

def build_market_stress_dashboard(indicators: dict[str, Indicator]):
    """
    Compact dashboard showing core systemic stress signals.
    """

    return [
        ("Volatility (VIX)", "equity stress", indicators["vix"].status),
        ("Credit Stress (HY spreads)", "credit market", indicators["credit_spread"].status),
        ("Bond Stress (MOVE)", "bond market", indicators["move"].status),
        ("Liquidity (USD)", "global liquidity", indicators["usd_index"].status),
        ("Risk Appetite (HYG)", "risk appetite", indicators["hyg"].status),
        ("Market Breadth", "market structure", indicators["sp_breadth"].status),
    ]

def build_macro_cycle_dashboard(indicators: dict[str, Indicator]):
    return [
        ("Global Liquidity", "macro liquidity", indicators["macro_global_liquidity"].status),
        ("Inflation Trend", "price pressure", indicators["inflation_trend"].status),
        ("Oil Shock", "energy shock", indicators["oil_shock"].status),
        ("Dollar Strength", "global funding", indicators["usd_index"].status),
    ]

def build_structural_vulnerability_dashboard(indicators: dict[str, Indicator]):
    return [
        ("Debt Burden", "balance sheet risk", indicators["debt_burden"].status),
        ("Valuation Stretch", "asset pricing", indicators["valuation_stretch"].status),
        ("Credit System Fragility", "intermediation risk", indicators["credit_system_fragility"].status),
    ]

def determine_macro_regime(indicators: dict[str, Indicator]) -> tuple[str, str, list[str]]:
    """
    Context-only macro regime classifier.
    Does not feed into crash probability.
    """

    drivers: list[str] = []

    inflation_bad = indicators["inflation_trend"].status in ("WARNING", "ALARM")
    oil_bad = indicators["oil_shock"].status in ("WARNING", "ALARM")
    liquidity_bad = indicators["macro_global_liquidity"].status in ("WARNING",)
    liquidity_soft = indicators["macro_global_liquidity"].status in ("ELEVATED", "WARNING")
    usd_bad = indicators["usd_index"].status in ("ELEVATED", "WARNING", "ALARM")
    vix_bad = indicators["vix"].status in ("WARNING", "ALARM")
    hyg_bad = indicators["hyg"].status in ("WARNING", "ALARM")
    credit_bad = indicators["credit_spread"].status in ("WARNING", "ALARM")

    if credit_bad and liquidity_bad and vix_bad:
        if inflation_bad or oil_bad:
            drivers = ["credit stress", "liquidity contraction", "volatility stress", "inflation pressure"]
        else:
            drivers = ["credit stress", "liquidity contraction", "volatility stress"]
        return (
            "Crisis Transition",
            "Macro conditions are transitioning toward systemic stress. Financial stress is broadening and liquidity conditions are worsening.",
            drivers,
        )

    if inflation_bad and oil_bad and (usd_bad or liquidity_soft):
        drivers = ["inflation pressure", "energy shock", "tighter financial conditions"]
        return (
            "Stagflation Risk",
            "Inflation and energy pressures are rising while financial conditions are becoming less supportive. This is a classic stagflation-type setup.",
            drivers,
        )

    if liquidity_soft and usd_bad:
        drivers = ["liquidity tightening", "stronger USD"]
        return (
            "Liquidity Tightening",
            "Liquidity conditions are becoming less supportive and the stronger dollar suggests tighter global financial conditions.",
            drivers,
        )

    if inflation_bad or oil_bad or usd_bad:
        drivers = []
        if inflation_bad:
            drivers.append("inflation pressure")
        if oil_bad:
            drivers.append("energy pressure")
        if usd_bad:
            drivers.append("tighter financial conditions")
        return (
            "Late Cycle",
            "Macro conditions are becoming less supportive for risk assets. Inflation, energy or funding pressure suggest a later-cycle environment.",
            drivers,
        )

    return (
        "Expansion",
        "Macro conditions remain broadly supportive. Inflation, liquidity and energy signals do not currently indicate major macro stress.",
        drivers,
    )

def determine_structural_vulnerability(indicators: dict[str, Indicator]) -> tuple[str, str, list[str]]:
    drivers: list[str] = []

    debt_status = indicators["debt_burden"].status
    valuation_status = indicators["valuation_stretch"].status
    fragility_status = indicators["credit_system_fragility"].status

    warning_count = sum(
        s in ("WARNING", "ALARM")
        for s in [debt_status, valuation_status, fragility_status]
    )
    elevated_count = sum(
        s in ("ELEVATED", "WARNING", "ALARM")
        for s in [debt_status, valuation_status, fragility_status]
    )

    if debt_status in ("ELEVATED", "WARNING", "ALARM"):
        drivers.append("debt burden")
    if valuation_status in ("ELEVATED", "WARNING", "ALARM"):
        drivers.append("valuation stretch")
    if fragility_status in ("ELEVATED", "WARNING", "ALARM"):
        drivers.append("credit system fragility")

    if warning_count >= 2:
        return (
            "WARNING",
            "Structural vulnerability is high. The financial system shows multiple longer-term fragility signals.",
            drivers,
        )

    if elevated_count >= 1:
        return (
            "ELEVATED",
            "Structural vulnerability is elevated. The system may be more fragile if market stress increases.",
            drivers,
        )

    return (
        "OK",
        "Structural vulnerability appears limited. No major long-term fragility signals stand out.",
        drivers,
    )

def interpret_stress_vs_vulnerability(
    crash_probability: int,
    structural_status: str,
) -> str:
    if crash_probability < 30 and structural_status == "OK":
        return (
            "Scenario A: Market stress is low and structural vulnerability is also low. "
            "Financial conditions appear broadly stable."
        )

    if crash_probability < 30 and structural_status in ("ELEVATED", "WARNING"):
        return (
            "Scenario B: Market stress is currently low, but structural vulnerability is elevated. "
            "There is no immediate crisis signal, but the system has a fragile foundation."
        )

    if crash_probability >= 30 and structural_status in ("ELEVATED", "WARNING"):
        return (
            "Scenario C: Market stress is rising while structural vulnerability is already elevated. "
            "This is a more dangerous combination because stress is building on top of a weaker system."
        )

    return (
        "Market stress and structural vulnerability currently provide a mixed picture."
    )

def detect_calm_before_the_storm(
    indicators: dict[str, Indicator],
    crash_probability: int,
    structural_status: str,
) -> tuple[bool, str]:
    """
    Detects a deceptively calm market environment while structural fragility is building.
    Context only: does not affect score or crash probability.
    """

    vix_ok = indicators["vix"].status == "OK"
    move_ok = indicators["move"].status in ("OK", "ELEVATED")
    credit_not_broken = indicators["credit_spread"].status in ("OK", "ELEVATED")
    market_stress_low = crash_probability < 30
    structural_high = structural_status in ("ELEVATED", "WARNING")

    if market_stress_low and structural_high and vix_ok and move_ok and credit_not_broken:
        return (
            True,
            "Calm Before the Storm detected: current market stress is still relatively low, "
            "but structural vulnerability is elevated. Markets may appear calm while underlying fragility is building."
        )

    return (
        False,
        "No Calm Before the Storm pattern detected."
    )

def indicator_weights() -> dict[str, float]:
    return {
        "credit_spread": 1.8,
        "hyg": 1.7,
        "move": 1.5,

        "vix": 1.2,
        "vol_regime": 1.0,
        "yield_curve_2s10s": 1.0,
        "global_liquidity": 1.0,

        "usd_index": 0.7,
        "sp_breadth": 0.7,
        "nasdaq": 0.7,

        "fed_balance": 0.4,

        # context indicators
        "macro_global_liquidity": 0.0,
        "inflation_trend": 0.0,
        "oil_shock": 0.0,

        # structural vulnerability
        "debt_burden": 0.0,
        "valuation_stretch": 0.0,
        "credit_system_fragility": 0.0,
    }

def grouped_indicator_sections(indicators: list[Indicator]) -> list[tuple[str, list[Indicator]]]:

    groups = {
        "Core Market Stress (drives score)": [
            "vix",
            "vol_regime",
            "credit_spread",
            "move",
            "yield_curve_2s10s",
            "fed_balance",
            "global_liquidity",
            "usd_index",
            "hyg",
            "sp_breadth",
            "nasdaq",
        ],

        "Macro Context (context only)": [
            "macro_global_liquidity",
            "inflation_trend",
            "oil_shock",
        ],

        "Structural Vulnerability (context only)": [
            "debt_burden",
            "valuation_stretch",
            "credit_system_fragility",
        ],
    }

    weights = indicator_weights()

    by_key = {i.key: i for i in indicators}

    result: list[tuple[str, list[Indicator]]] = []

    for section_name, keys in groups.items():

        section_items = [by_key[k] for k in keys if k in by_key]

        # sorteren binnen categorie op weight
        section_items.sort(
            key=lambda x: weights.get(x.key, 0),
            reverse=True
        )

        if section_items:
            result.append((section_name, section_items))

    return result


def top_risk_drivers(indicators: list[Indicator], top_n: int = 3) -> list[str]:
    """
    Top risk drivers based on weighted contribution:
    contribution = score * weight
    Only indicators with contribution > 0 are shown.
    """
    weights = indicator_weights()

    ranked = sorted(
        indicators,
        key=lambda ind: ind.score * weights.get(ind.key, 0.0),
        reverse=True,
    )

    result: list[str] = []
    for ind in ranked:
        contribution = ind.score * weights.get(ind.key, 0.0)
        if contribution <= 0:
            continue

        result.append(
            f"{ind.name} (status: {ind.status}, weight: {weights.get(ind.key, 0.0):.1f}, contribution: {contribution:.1f})"
        )

        if len(result) >= top_n:
            break

    return result

def print_report(output: RadarOutput) -> None:
    print()
    print(f"{ANSI['bold']}{ANSI['cyan']}MACRO CRASH RADAR V3{ANSI['reset']}")
    print(f"UTC: {output.timestamp_utc}")

    print("--- Summary ---")
    print(output.summary_line)
    print()

    print("--- Status severity scale ---")
    print("OK > ELEVATED > WARNING > ALARM")
    print("Weight = contribution factor to crash probability model")
    print()

    print("--- Top Risk Drivers ---")
    print("Ranked by current risk impact.")
    print()

    if output.top_risk_drivers:
        for idx, driver in enumerate(output.top_risk_drivers, start=1):
            print(f"{idx}. {driver}")
    else:
        print("No material weighted risk drivers detected.")
    print()

    print("--- Market Stress Dashboard ---")
    dashboard = build_market_stress_dashboard({i.key: i for i in output.indicators})

    print(f"{'Indicator':<28} {'System':<20} {'Status'}")
    print("-" * 60)

    for name, system, status in dashboard:
        print(f"{name:<28} {system:<20} {status}")

    print()

    print("--- Macro Cycle Indicators ---")
    macro_dashboard = build_macro_cycle_dashboard({i.key: i for i in output.indicators})

    print(f"{'Indicator':<24} {'System':<18} {'Status'}")
    print("-" * 55)

    for name, system, status in macro_dashboard:
        print(f"{name:<24} {system:<18} {status}")

    print()

    print("--- Macro Regime Classifier ---")
    print(f"Macro regime: {output.macro_regime}")
    print(f"Explanation: {output.macro_regime_explanation}")
    if output.macro_regime_drivers:
        print("Drivers: " + "; ".join(output.macro_regime_drivers))
    print()

    print("--- Structural Vulnerability ---")
    structural_dashboard = build_structural_vulnerability_dashboard({i.key: i for i in output.indicators})

    print(f"{'Indicator':<28} {'System':<24} {'Status'}")
    print("-" * 70)

    for name, system, status in structural_dashboard:
        print(f"{name:<28} {system:<24} {status}")

    print()
    print(f"Structural vulnerability: {output.structural_vulnerability_status}")
    print(f"Explanation: {output.structural_vulnerability_explanation}")
    if output.structural_vulnerability_drivers:
        print("Drivers: " + "; ".join(output.structural_vulnerability_drivers))
    print()

    print("--- Crash Setup Detector ---")
    print(f"Crash setup: {output.crash_setup_level}")
    print(f"Explanation: {output.crash_setup_explanation}")
    if output.crash_setup_reasons:
        print("Present conditions: " + "; ".join(output.crash_setup_reasons))
    if output.crash_setup_missing:
        print("Missing conditions: " + "; ".join(output.crash_setup_missing))
    print()

    print("--- Market Phase Indicator ---")
    print(f"Market phase: {output.market_phase}")
    print(f"Explanation: {output.market_phase_explanation}")
    if output.market_phase_drivers:
        print("Drivers: " + "; ".join(output.market_phase_drivers))
    print()

    print("--- Bond Regime Indicator ---")
    print(f"Bond regime: {output.bond_regime}")
    print(f"Explanation: {output.bond_regime_explanation}")
    if output.bond_regime_drivers:
        print("Drivers: " + "; ".join(output.bond_regime_drivers))
    print()

    print("-" * 120)

    weights = indicator_weights()
    sections = grouped_indicator_sections(output.indicators)

    for section_name, section_indicators in sections:
        print(f"{ANSI['bold']}{section_name}{ANSI['reset']}")
        print(f"{'Indicator':<24} {'Value':>9} {'Weight':>4} {'Status':>10}   Detail")
        print("-" * 120)

        for ind in section_indicators:
            color = color_for_status(ind.status)

            if isinstance(ind.value, (int, float)) and ind.value is not None:
                if ind.key == "fed_balance":
                    value_txt = f"{ind.value / 1e6:.4f}T"
                else:
                    value_txt = f"{ind.value:.2f}"
            else:
                value_txt = "n/a"

            weight = weights.get(ind.key, 0.0)
            weight_txt = f"{weight:.1f}" if weight > 0 else "0"

            explanation = STATUS_EXPLANATION.get(ind.status, "")

            
            print(
                f"{ind.name:<24} "
                f"{value_txt:>9} "
                f"{weight_txt:>4} "
                f"{color}{ind.status:<10}{ANSI['reset']}   "
                f"{ind.detail}   ({explanation})"
            )
        print()

    print("-" * 120)
    print(f"{ANSI['bold']}Total score:{ANSI['reset']} {output.total_score}")
    print(f"{ANSI['bold']}Crash probability:{ANSI['reset']} {output.crash_probability_pct}%")
    print(f"{ANSI['bold']}Risk regime:{ANSI['reset']} {output.regime}")
    print(f"{ANSI['bold']}Hedge-fund trigger:{ANSI['reset']} {'YES' if output.hedge_fund_trigger else 'NO'}")

    trend_display = output.trend_score if output.trend_score is not None else "n/a"
    print(f"{ANSI['bold']}Trend score:{ANSI['reset']} {trend_display}")
    print(f"{ANSI['bold']}Risk trend:{ANSI['reset']} {output.risk_trend}")
    if output.hedge_fund_trigger and output.trigger_reasons:
        print(f"{ANSI['bold']}Trigger reasons:{ANSI['reset']} " + "; ".join(output.trigger_reasons))
    print("-" * 120)
    print(output.narrative)
    print()

    regime, crash_view, positioning, explanation = market_summary(output)

    print("--- Market Interpretation ---")
    print(f"Market regime: {regime}")
    print(f"Crash risk assessment: {crash_view}")
    print(f"Suggested portfolio positioning: {positioning}")
    print()

    print("Explanation:")
    print(explanation)
    print()

    print("--- System Risk Interpretation ---")
    print(output.system_risk_interpretation)
    print()

    print("--- Interpreting Market Stress vs Structural Vulnerability ---")
    print("Scenario A")
    print("Market stress low + structural vulnerability low")
    print("→ Financial system appears stable. No major systemic stress signals.")
    print()

    print("Scenario B")
    print("Market stress low + structural vulnerability high")
    print("→ Markets are currently calm, but the system has a fragile foundation.")
    print("  If a shock occurs, stress could escalate faster than usual.")
    print()

    print("Scenario C")
    print("Market stress high + structural vulnerability high")
    print("→ Potentially dangerous environment. Market stress is appearing on top of")
    print("  an already vulnerable financial system, increasing the probability of")
    print("  a larger correction or systemic crisis.")
    print()

    print("--- Calm Before the Storm Detector ---")
    print(f"Detected: {'YES' if output.calm_before_storm else 'NO'}")
    print(output.calm_before_storm_explanation)
    print()

    print("Risk trend explanation:")
    print(output.trend_explanation)
    print()

    print("Concepts explained:")
    print("Risk-off environment → investors reduce exposure to risky assets and prefer safer investments.")
    print("Defensive positioning → investors shift part of their portfolio toward safer assets or sectors.")
    print()

    print("--- Risk Regime Legend ---")
    print("NORMAL (0–29) → markets stable, no systemic stress")
    print("ELEVATED RISK (30–49) → early warning signals appearing")
    print("HIGH RISK (50–64) → multiple stress indicators, market vulnerable to correction")
    print("SEVERE RISK (65–79) → strong systemic stress signals, crash risk rising")
    print("CRASH WARNING (80+) → broad systemic stress, crash conditions possible")
    print()

    print("--- How to interpret the radar ---")
    print("Market stress typically develops in stages:")
    print("1. Market breadth weakens")
    print("2. High yield bonds weaken")
    print("3. Volatility (VIX) rises")
    print("4. Credit spreads widen")
    print("5. Liquidity stress appears")
    print()

    print("Key crash confirmation indicators:")
    print("- Credit spreads rising")
    print("- Volatility spike (VIX)")
    print("- Liquidity tightening (USD / liquidity indicators)")
    print()

    print("If these occur together, systemic market stress becomes more likely.")
    print()


def save_csv_row(output: RadarOutput, path: str) -> None:
    row: dict[str, Any] = {
        "timestamp_utc": output.timestamp_utc,
        "total_score": output.total_score,
        "crash_probability_pct": output.crash_probability_pct,
        "regime": output.regime,
        "hedge_fund_trigger": output.hedge_fund_trigger,
        "trigger_reasons": " | ".join(output.trigger_reasons),
        "narrative": output.narrative,

        # Crash setup detector
        "crash_setup_level": output.crash_setup_level,
        "crash_setup_reasons": " | ".join(output.crash_setup_reasons),
        "crash_setup_missing": " | ".join(output.crash_setup_missing),
        "crash_setup_explanation": output.crash_setup_explanation,

        # Market phase
        "market_phase": output.market_phase,
        "market_phase_explanation": output.market_phase_explanation,
        "market_phase_drivers": " | ".join(output.market_phase_drivers),

        "bond_regime": output.bond_regime,
        "bond_regime_explanation": output.bond_regime_explanation,
        "bond_regime_drivers": " | ".join(output.bond_regime_drivers),

        "trend_score": output.trend_score,
        "risk_trend": output.risk_trend,
        "trend_explanation": output.trend_explanation,

        "macro_cycle_status": output.macro_cycle_status,
        "macro_cycle_explanation": output.macro_cycle_explanation,
        "macro_cycle_drivers": " | ".join(output.macro_cycle_drivers),

        "macro_regime": output.macro_regime,
        "macro_regime_explanation": output.macro_regime_explanation,
        "macro_regime_drivers": " | ".join(output.macro_regime_drivers),

        "structural_vulnerability_status": output.structural_vulnerability_status,
        "structural_vulnerability_explanation": output.structural_vulnerability_explanation,
        "structural_vulnerability_drivers": " | ".join(output.structural_vulnerability_drivers),

        "system_risk_interpretation": output.system_risk_interpretation,
        "summary_line": output.summary_line,
        "dashboard_explanation": output.dashboard_explanation,

        "calm_before_storm": output.calm_before_storm,
        "calm_before_storm_explanation": output.calm_before_storm_explanation,

        "top_risk_drivers": " | ".join(output.top_risk_drivers),

        "indicator_count": len(output.indicators),
    }

    for ind in output.indicators:
        row[f"{ind.key}_value"] = ind.value
        row[f"{ind.key}_status"] = ind.status
        row[f"{ind.key}_score"] = ind.score
        row[f"{ind.key}_detail"] = ind.detail

    df = pd.DataFrame([row])

    if os.path.exists(path):
        df.to_csv(
            path,
            mode="a",
            header=False,
            index=False,
            quoting=csv.QUOTE_ALL
        )
    else:
        df.to_csv(
            path,
            index=False,
            quoting=csv.QUOTE_ALL
        )


def debug_series(name: str, series: pd.Series) -> None:
    print(f"DEBUG {name} latest:")
    print(series.tail(3))
    print()


# =========================
# Main radar
# =========================

def determine_market_phase(
    indicators: dict[str, Indicator],
    crash_probability_pct: int,
    crash_setup_level: str,
) -> tuple[str, str, list[str]]:
    """
    Market phase classifier:
    - Expansion
    - Late-cycle
    - Risk-off
    - Stress
    - Crisis
    """

    drivers: list[str] = []

    vix_bad = indicators["vix"].status in ("WARNING", "ALARM")
    hyg_bad = indicators["hyg"].status in ("WARNING", "ALARM")
    breadth_bad = indicators["sp_breadth"].status in ("ELEVATED", "WARNING", "ALARM")
    nasdaq_bad = indicators["nasdaq"].status in ("ELEVATED", "WARNING", "ALARM")
    usd_bad = indicators["usd_index"].status in ("ELEVATED", "WARNING", "ALARM")
    liquidity_bad = indicators["global_liquidity"].status in ("WARNING", "ALARM")
    credit_bad = indicators["credit_spread"].status in ("WARNING", "ALARM")
    move_bad = indicators["move"].status in ("WARNING", "ALARM")

    if vix_bad:
        drivers.append("volatility rising")
    if hyg_bad:
        drivers.append("credit-sensitive assets weakening")
    if breadth_bad:
        drivers.append("market breadth weakening")
    if nasdaq_bad:
        drivers.append("growth assets weakening")
    if usd_bad:
        drivers.append("USD strength / tighter conditions")
    if liquidity_bad:
        drivers.append("liquidity deteriorating")
    if credit_bad:
        drivers.append("credit spreads widening")
    if move_bad:
        drivers.append("bond volatility rising")

    # Crisis
    if crash_setup_level == "YES" or (
        crash_probability_pct >= 80 and credit_bad and (move_bad or liquidity_bad)
    ):
        return (
            "Crisis",
            "Core systemic indicators are aligned. Credit, volatility and liquidity conditions point to broad market stress.",
            drivers,
        )

    # Stress
    if crash_setup_level == "WATCH" or crash_probability_pct >= 65:
        return (
            "Stress",
            "Market stress is broadening. Risk conditions are worsening across multiple areas and the environment is vulnerable to sharp downside moves.",
            drivers,
        )

    # Risk-off
    if crash_setup_level == "EARLY" or (
        crash_probability_pct >= 50 and (vix_bad or hyg_bad) and (breadth_bad or nasdaq_bad)
    ):
        return (
            "Risk-off",
            "Investors are becoming more defensive. Volatility is rising and risk assets are weakening, but systemic crisis conditions are not yet confirmed.",
            drivers,
        )

    # Late-cycle
    if crash_probability_pct >= 30 or usd_bad or breadth_bad:
        return (
            "Late-cycle",
            "Early warning signs are appearing. Market leadership is narrowing and conditions are becoming less supportive for risk-taking.",
            drivers,
        )

    # Expansion
    return (
        "Expansion",
        "Markets remain broadly stable. Risk appetite and macro conditions do not currently signal meaningful systemic stress.",
        drivers,
    )

def determine_bond_regime(indicators: dict[str, Indicator]) -> tuple[str, str, list[str]]:
    """
    Bond Regime Indicator focused mainly on government bonds.

    Regimes:
    - Bond Bullish
    - Bond Neutral
    - Bond Volatile
    - Bond Bearish
    """

    drivers: list[str] = []

    move_status = indicators["move"].status
    yc_status = indicators["yield_curve_2s10s"].status
    vix_status = indicators["vix"].status
    usd_status = indicators["usd_index"].status
    credit_status = indicators["credit_spread"].status
    liquidity_status = indicators["global_liquidity"].status

    move_bad = move_status in ("WARNING", "ALARM")
    vix_bad = vix_status in ("WARNING", "ALARM")
    usd_bad = usd_status in ("WARNING",)
    liquidity_bad = liquidity_status in ("WARNING", "ALARM")
    credit_bad = credit_status in ("WARNING", "ALARM")
    yc_bad = yc_status in ("WARNING",)

    if move_bad:
        drivers.append("bond volatility elevated")
    if vix_bad:
        drivers.append("risk-off demand rising")
    if usd_bad:
        drivers.append("tight funding / strong USD")
    if liquidity_bad:
        drivers.append("liquidity deteriorating")
    if credit_bad:
        drivers.append("credit stress rising")
    if yc_bad:
        drivers.append("yield curve stress")

    # Bond Bearish:
    # strong USD / liquidity stress / MOVE stress often means bonds may not hedge well
    if move_bad and (usd_bad or liquidity_bad):
        return (
            "Bond Bearish",
            "Government bonds may struggle as a hedge because bond volatility and funding conditions are under pressure.",
            drivers,
        )

    # Bond Volatile:
    # bond market uncertainty elevated, outcome less clear
    if move_bad or (usd_bad and vix_bad):
        return (
            "Bond Volatile",
            "Government bonds may remain unstable because rates uncertainty or funding stress is rising.",
            drivers,
        )

    # Bond Bullish:
    # risk-off present, but bond market itself still calm
    if vix_bad and not move_bad and not usd_bad and not liquidity_bad:
        return (
            "Bond Bullish",
            "Government bonds may benefit from mild flight-to-safety demand while bond market volatility remains contained.",
            drivers,
        )

    # Default neutral
    return (
        "Bond Neutral",
        "Government bonds appear relatively stable. Current conditions do not strongly point to either a major bond rally or bond sell-off.",
        drivers,
    )
def calculate_trend_from_csv(csv_path: str, current_prob: int) -> tuple[int | None, str, str]:
    """
    Compare current crash probability with the previous saved snapshot in CSV.
    Returns:
    - trend_score: difference in probability points
    - risk_trend: rising / rising fast / stable / falling / falling fast / unavailable
    - explanation
    """
    if not csv_path or not os.path.exists(csv_path):
        return None, "unavailable", "No historical log available yet."

    try:
        df = pd.read_csv(
            csv_path,
            engine="python",
            on_bad_lines="skip"
        )

        if df.empty or "crash_probability_pct" not in df.columns:
            return None, "unavailable", "No usable crash probability history found."

        prev_prob = int(df["crash_probability_pct"].iloc[-1])
        delta = int(current_prob - prev_prob)

        if delta >= 8:
            trend = "rising fast"
            explanation = "Market risk is increasing quickly versus the previous snapshot."
        elif delta >= 3:
            trend = "rising"
            explanation = "Market risk is increasing versus the previous snapshot."
        elif delta <= -8:
            trend = "falling fast"
            explanation = "Market risk is improving quickly versus the previous snapshot."
        elif delta <= -3:
            trend = "falling"
            explanation = "Market risk is easing versus the previous snapshot."
        else:
            trend = "stable"
            explanation = "Market risk is broadly stable versus the previous snapshot."

        return delta, trend, explanation

    except Exception as e:
        return None, "unavailable", f"Could not calculate trend from history ({e})."


def indicator_display_name(key: str) -> str:
    names = {
        "vix": "VIX",
        "vol_regime": "vol regime",
        "credit_spread": "credit spreads",
        "move": "MOVE",
        "yield_curve_2s10s": "yield curve",
        "fed_balance": "Fed balance sheet",
        "global_liquidity": "global liquidity",
        "usd_index": "USD",
        "hyg": "HYG",
        "sp_breadth": "breadth",
        "nasdaq": "Nasdaq",
        "macro_global_liquidity": "macro liquidity",
        "inflation_trend": "inflation",
        "oil_shock": "oil",
        "debt_burden": "debt burden",
        "valuation_stretch": "valuation stretch",
        "credit_system_fragility": "credit fragility",
    }
    return names.get(key, key)


def status_level(status: str) -> int:
    order = {
        "OK": 0,
        "ELEVATED": 1,
        "WARNING": 2,
        "ALARM": 3,
        "UNAVAILABLE": -1,
    }
    return order.get(status, -1)


def comparison_label(prev_timestamp_raw: str | None, current_dt_utc: datetime) -> str:
    """
    Human-friendly comparison label:
    - same day → since previous run
    - 1 day difference → since yesterday
    - >1 day difference → since X days ago
    """

    if not prev_timestamp_raw:
        return "since previous run"

    try:
        prev_dt = datetime.strptime(
            prev_timestamp_raw,
            "%Y-%m-%d %H:%M:%S UTC"
        ).replace(tzinfo=timezone.utc)

        day_diff = (current_dt_utc.date() - prev_dt.date()).days

        if day_diff == 0:
            return "since previous run"

        if day_diff == 1:
            return "since yesterday"

        return f"since {day_diff} days ago"

    except Exception:
        return "since previous run"

def risk_stage_index(regime: str) -> int:
    order = {
        "NORMAL": 0,
        "ELEVATED RISK": 1,
        "HIGH RISK": 2,
        "SEVERE RISK": 3,
        "CRASH WARNING": 4,
    }
    return order.get(regime, 0)


def trend_symbol_and_label(risk_trend: str) -> tuple[str, str]:
    mapping = {
        "rising fast": ("↑", "Rising fast"),
        "rising": ("↑", "Rising"),
        "stable": ("→", "Stable"),
        "falling": ("↓", "Falling"),
        "falling fast": ("↓", "Falling fast"),
        "unavailable": ("•", "Unavailable"),
    }
    return mapping.get(risk_trend, ("•", "Unavailable"))

def trend_comparison_text(output: "RadarOutput", csv_path: str = "") -> str:
    """
    Rebuilds the same time-comparison wording for HTML display.
    Falls back gracefully if no CSV history is available.
    """
    if not csv_path or not os.path.exists(csv_path):
        return "vs previous run"

    try:
        df = pd.read_csv(
            csv_path,
            engine="python",
            on_bad_lines="skip"
        )
        if df.empty:
            return "vs previous run"

        prev = df.iloc[-1]
        prev_timestamp_raw = str(prev["timestamp_utc"]) if "timestamp_utc" in prev else None
        current_dt_utc = datetime.strptime(output.timestamp_utc, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)

        label = comparison_label(prev_timestamp_raw, current_dt_utc)

        if label == "since previous run":
            return "vs previous run"
        if label == "since yesterday":
            return "vs yesterday"
        if label.startswith("since "):
            return label.replace("since ", "vs ")

        return "vs previous run"

    except Exception:
        return "vs previous run"  

def build_summary_line(output: RadarOutput, csv_path: str) -> str:
    """
    One-line summary versus previous snapshot in CSV.
    Uses crash probability, regime change, and indicator status changes.
    """
    if not csv_path or not os.path.exists(csv_path):
        return (
            f"Crash risk is {output.regime.lower()} at {output.crash_probability_pct}%; "
            f"no previous run available yet."
        )

    try:
        df = pd.read_csv(
            csv_path,
            engine="python",
            on_bad_lines="skip"
        )
        if df.empty:
            return (
                f"Crash risk is {output.regime.lower()} at {output.crash_probability_pct}%; "
                f"no previous run available yet."
            )

        prev = df.iloc[-1]

        prev_prob = int(prev["crash_probability_pct"]) if "crash_probability_pct" in prev else output.crash_probability_pct
        delta = output.crash_probability_pct - prev_prob

        prev_timestamp_raw = str(prev["timestamp_utc"]) if "timestamp_utc" in prev else None
        current_dt_utc = datetime.strptime(output.timestamp_utc, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        compare_txt = comparison_label(prev_timestamp_raw, current_dt_utc)

        if delta >= 3:
            trend_txt = f"rising {compare_txt} ({output.crash_probability_pct}% vs {prev_prob}%)"
        elif delta <= -3:
            trend_txt = f"falling {compare_txt} ({output.crash_probability_pct}% vs {prev_prob}%)"
        else:
            trend_txt = f"stable {compare_txt} ({output.crash_probability_pct}% vs {prev_prob}%)"

        prev_regime = str(prev["regime"]) if "regime" in prev else output.regime
        if prev_regime != output.regime:
            regime_txt = f"regime changed from {prev_regime} to {output.regime}"
        else:
            regime_txt = f"regime unchanged ({output.regime})"

        worsening: list[str] = []
        improving: list[str] = []

        for ind in output.indicators:
            col = f"{ind.key}_status"
            if col not in prev:
                continue

            prev_status = str(prev[col])
            prev_level = status_level(prev_status)
            curr_level = status_level(ind.status)

            if curr_level > prev_level:
                worsening.append(indicator_display_name(ind.key))
            elif curr_level < prev_level:
                improving.append(indicator_display_name(ind.key))

        parts = [f"Crash risk is {trend_txt}", regime_txt]

        if worsening:
            parts.append("worsening in " + ", ".join(worsening[:3]))
        if improving:
            parts.append("improvement in " + ", ".join(improving[:3]))

        if not worsening and not improving:
            parts.append("no material indicator shifts")

        return "; ".join(parts) + "."

    except Exception as e:
        return (
            f"Crash risk is {output.regime.lower()} at {output.crash_probability_pct}%; "
            f"comparison unavailable ({e})."
        )

def build_dashboard_explanation() -> str:
    return """
This dashboard is designed as an early warning system for market stress and potential crash risk.

1. What this dashboard does
The model tracks a set of financial and macro indicators that often deteriorate before major market corrections or financial stress events. It does not try to predict the exact timing of a crash, but it helps identify whether market conditions are becoming more fragile or more stressed.

2. How the crash probability is built
The crash probability is based mainly on the Core Market Stress indicators. Each of these indicators receives:
- a status (OK, ELEVATED, WARNING, ALARM),
- a score,
- and a model weight.

The weighted combination of these signals becomes a total risk score, which is then translated into a crash probability percentage. The higher the score, the higher the implied market stress.

3. Core Market Stress indicators
These are the main drivers of the model and directly affect the crash probability.

- VIX: measures expected equity market volatility. A sharp rise usually signals growing fear in stock markets.
- Vol Regime Shift: shows whether volatility is moving into a more stressed regime relative to recent averages.
- High Yield Credit Spread: measures stress in lower-quality corporate debt. Wider spreads often signal deteriorating credit conditions.
- MOVE: measures bond market volatility. Rising MOVE can indicate stress in rates markets or uncertainty around liquidity and policy.
- Yield Curve 2s10s: shows the difference between 10-year and 2-year US Treasury yields. It helps indicate macro and cycle stress.
- Fed Balance Sheet: tracks the direction of central bank liquidity support.
- Global Liquidity: combines major liquidity signals and helps detect tightening financial conditions.
- USD Index: a stronger dollar can signal tighter global funding conditions.
- HYG Trend: high yield bond ETF trend. Weakness here often appears before broader equity stress.
- S&P Breadth: shows whether market strength is broad or concentrated in fewer stocks.
- Nasdaq Trend: helps monitor risk appetite and pressure in growth-oriented equities.

4. Macro Context indicators
These indicators do not directly drive the crash score, but they help explain the broader environment.

- Global Liquidity Context: broader liquidity backdrop including dollar pressure.
- Inflation Trend: helps assess whether inflation remains a macro headwind.
- Oil Shock: rising oil prices can tighten financial conditions and increase macro stress.

5. Structural Vulnerability indicators
These indicators also do not directly drive the crash score. They show whether the system may be fragile beneath the surface.

- Debt Burden: a proxy for balance sheet and debt-related vulnerability.
- Valuation Stretch: a proxy for whether markets may be expensive or stretched.
- Credit System Fragility: a proxy for weakness in credit-sensitive areas and funding conditions.

These indicators are especially useful when market stress is still low, but structural vulnerability is elevated. In that situation, markets may look calm while the underlying foundation is more fragile.

6. How to interpret the dashboard
A practical way to read the dashboard is:
- Start with the Summary line.
- Then look at the Top Risk Drivers.
- Then review the Core Market Stress section, because that drives the score.
- Use Macro Context and Structural Vulnerability as supporting interpretation.
- Use the Crash Setup Detector and Calm Before the Storm Detector as additional context.

7. What the statuses mean
- OK: normal conditions
- ELEVATED: early warning or rising tension
- WARNING: clear market stress
- ALARM: rare, severe stress signal

8. Important limitation
This dashboard is an early warning and interpretation tool. It is not a guarantee that markets will crash, nor does it predict exact timing. It is best used to monitor whether risk is rising, stable, or easing over time.
""".strip()            

def run_radar(debug: bool = False, csv_path: str = "") -> RadarOutput:
    # FRED data
    credit_spread = fred_series(FRED_SERIES["credit_spread"])
    us10y = fred_series(FRED_SERIES["us10y"])
    us2y = fred_series(FRED_SERIES["us2y"])
    fed_balance = fred_series(FRED_SERIES["fed_balance"])
    vix = fred_series(FRED_SERIES["vix"])
    m2 = fred_series(FRED_SERIES["m2"])
    cpi = fred_series(FRED_SERIES["cpi"])
   

    # Yahoo data
    hyg = yf_close_series(YF_TICKERS["hyg"])
    nasdaq = yf_close_series(YF_TICKERS["nasdaq"])
    spy = yf_close_series(YF_TICKERS["spy"])
    rsp = yf_close_series(YF_TICKERS["rsp"])
    brent = yf_close_series(YF_TICKERS["brent"])

    # MOVE
    try:
        move = yf_close_series(YF_TICKERS["move"])
        move_indicator = analyze_move(move)
    except Exception as e:
        move_indicator = Indicator(
            key="move",
            name="MOVE",
            value=None,
            status="UNAVAILABLE",
            score=0,
            detail=f"Could not fetch ^MOVE ({e})",
        )

    # USD index with fallback
    usd_label = "USD Index"
    try:
        usd = yf_close_series(YF_TICKERS["usd"])
        usd_indicator = analyze_usd_index(usd, label="USD Index")
    except Exception:
        usd = yf_close_series(YF_TICKERS["usd_fallback"])
        usd_label = "USD Proxy (UUP)"
        usd_indicator = analyze_usd_index(usd, label=usd_label)

    if debug:
        debug_series("DGS10", us10y)
        debug_series("DGS2", us2y)
        debug_series("WALCL", fed_balance)
        debug_series("VIXCLS", vix)
        debug_series("WM2NS", m2)
        debug_series("HYG", hyg)
        debug_series("NASDAQ", nasdaq)
        debug_series("SPY", spy)
        debug_series("RSP", rsp)
        if 'usd' in locals():
            debug_series(usd_label, usd)

    indicators_list = [
        analyze_vix(vix),
        analyze_vol_regime(vix),
        analyze_credit_spread(credit_spread),
        move_indicator,
        analyze_yield_curve(us10y, us2y),
        analyze_fed_balance(fed_balance),
        analyze_global_liquidity(fed_balance, m2),
        usd_indicator,
        analyze_hyg(hyg),
        analyze_sp_breadth(rsp, spy),
        analyze_nasdaq(nasdaq),

        # Macro context only
        analyze_macro_global_liquidity(fed_balance, m2, usd),
        analyze_inflation_trend(cpi),
        analyze_oil_shock(brent),

        # Structural vulnerability only
        analyze_debt_burden(fed_balance, m2),
        analyze_valuation_stretch(nasdaq, spy),
        analyze_credit_system_fragility(credit_spread, hyg, usd),
    ]

    indicators = {i.key: i for i in indicators_list}
    trigger_on, reasons = hedge_fund_trigger(indicators, hyg)
    crash_setup = crash_setup_detector(indicators)

    total_score = weighted_total_score(indicators_list)
    prob = calculate_probability(indicators_list, trigger_on)
    regime = regime_from_probability(prob)

    market_phase, market_phase_explanation, market_phase_drivers = determine_market_phase(
        indicators,
        prob,
        crash_setup.level,
    )

    bond_regime, bond_regime_explanation, bond_regime_drivers = determine_bond_regime(indicators)

    # trend_score, risk_trend, trend_explanation = calculate_trend_from_csv(csv_path, prob)

    macro_regime, macro_regime_explanation, macro_regime_drivers = determine_macro_regime(indicators)

    structural_vulnerability_status, structural_vulnerability_explanation, structural_vulnerability_drivers = determine_structural_vulnerability(indicators)

    system_risk_interpretation = interpret_stress_vs_vulnerability(
        prob,
        structural_vulnerability_status,
    )

    calm_before_storm, calm_before_storm_explanation = detect_calm_before_the_storm(
        indicators,
        prob,
        structural_vulnerability_status,
    )

    top_drivers = top_risk_drivers(indicators_list, top_n=3)

    macro_cycle_status = indicators["macro_global_liquidity"].status
    macro_cycle_explanation = (
        "Macro cycle indicators provide context only. They help explain whether stress is being driven by liquidity, inflation or energy conditions."
    )
    macro_cycle_drivers = []
    if indicators["macro_global_liquidity"].status in ("ELEVATED", "WARNING", "ALARM"):
        macro_cycle_drivers.append("global liquidity conditions")
    if indicators["inflation_trend"].status in ("ELEVATED", "WARNING", "ALARM"):
        macro_cycle_drivers.append("inflation trend")
    if indicators["oil_shock"].status in ("ELEVATED", "WARNING", "ALARM"):
        macro_cycle_drivers.append("oil shock")
    if indicators["usd_index"].status in ("ELEVATED", "WARNING", "ALARM"):
        macro_cycle_drivers.append("dollar strength")

    # =========================
    # Build comparison snapshots
    # =========================

    weights = indicator_weights()

    # Only include indicators that actually drive the score
    indicator_values = {
        ind.key: ind.value
        for ind in indicators_list
        if weights.get(ind.key, 0.0) > 0
    }

    indicator_risk_scores = {
        ind.key: float(ind.score)
        for ind in indicators_list
        if weights.get(ind.key, 0.0) > 0
    }

    current_snapshot = build_history_snapshot(
        crash_score=prob,   # compare on crash probability, because that is your top number
        indicator_values=indicator_values,
        indicator_risk_scores=indicator_risk_scores,
        stage_label=regime,
    )

    history = load_history(HISTORY_FILE)

    previous_run_snapshot = get_previous_run_snapshot(history)
    yesterday_snapshot = get_yesterday_snapshot(history, current_snapshot["timestamp"])

    compare_previous = calculate_driver_contributions(
        current_snapshot=current_snapshot,
        baseline_snapshot=previous_run_snapshot,
        indicator_meta=INDICATOR_META,
    )

    compare_yesterday = calculate_driver_contributions(
        current_snapshot=current_snapshot,
        baseline_snapshot=yesterday_snapshot,
        indicator_meta=INDICATOR_META,
    )

    summary_previous = format_summary_text(compare_previous, "previous run")
    summary_yesterday = format_summary_text(compare_yesterday, "yesterday")    

    output_obj = RadarOutput(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        total_score=total_score,
        crash_probability_pct=prob,
        regime=regime,
        hedge_fund_trigger=trigger_on,
        trigger_reasons=reasons,
        narrative=narrative(indicators_list, regime, trigger_on, reasons),
        indicators=indicators_list,
        crash_setup_level=crash_setup.level,
        crash_setup_reasons=crash_setup.reasons,
        crash_setup_missing=crash_setup.missing,
        crash_setup_explanation=crash_setup.explanation,
        market_phase=market_phase,
        market_phase_explanation=market_phase_explanation,
        market_phase_drivers=market_phase_drivers,
        bond_regime=bond_regime,
        bond_regime_explanation=bond_regime_explanation,
        bond_regime_drivers=bond_regime_drivers,
        trend_score=trend_score,
        risk_trend=risk_trend,
        trend_explanation=trend_explanation,
        macro_cycle_status=macro_cycle_status,
        macro_cycle_explanation=macro_cycle_explanation,
        macro_cycle_drivers=macro_cycle_drivers,
        macro_regime=macro_regime,
        macro_regime_explanation=macro_regime_explanation,
        macro_regime_drivers=macro_regime_drivers,
        structural_vulnerability_status=structural_vulnerability_status,
        structural_vulnerability_explanation=structural_vulnerability_explanation,
        structural_vulnerability_drivers=structural_vulnerability_drivers,
        system_risk_interpretation=system_risk_interpretation,
        calm_before_storm=calm_before_storm,
        calm_before_storm_explanation=calm_before_storm_explanation,
        top_risk_drivers=top_drivers,
        summary_line="",
        dashboard_explanation=build_dashboard_explanation(),
        comparison={
            "previous_run": {
                "mode": "previous_run",
                "label": "Previous run",
                "summary": summary_previous,
                "result": compare_previous,
            },
            "yesterday": {
                "mode": "yesterday",
                "label": "Yesterday",
                "summary": summary_yesterday,
                "result": compare_yesterday,
            },
            "default_mode": "previous_run",
        },
    )

    output_obj.summary_line = build_clean_summary_line(
        output_obj.comparison["previous_run"]["result"],
        "previous run"
    )

    append_history_snapshot(HISTORY_FILE, current_snapshot)

    return output_obj

def main() -> int:
    parser = argparse.ArgumentParser(description="Macro Crash Radar V3")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of terminal report")
    parser.add_argument("--csv", type=str, default="", help="Append one snapshot row to CSV")
    parser.add_argument("--debug", action="store_true", help="Print latest datapoints for debugging")

    parser.add_argument(
        "--report",
        type=str,
        default="",
        help="Write formatted report to this file path"
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory for auto-generated report files"
    )

    parser.add_argument(
        "--auto-report",
        action="store_true",
        help="Automatically create a timestamped report file"
    )

    parser.add_argument(
        "--html-report",
        type=str,
        default="",
        help="Write HTML report to this file path"
    )

    parser.add_argument(
        "--auto-html-report",
        action="store_true",
        help="Automatically create a timestamped HTML report file"
    )

    parser.add_argument(
        "--tee-report",
        action="store_true",
        help="When writing a report file, also print the same formatted report to console"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for runtime log files"
    )
    args = parser.parse_args()

    try:
        logfile = setup_logging(args.log_dir)
        LOGGER.info("Starting macro crash radar run")
        LOGGER.info(
            "Arguments | json=%s csv=%s debug=%s report=%s auto_report=%s html_report=%s auto_html_report=%s tee_report=%s",
            args.json,
            args.csv,
            args.debug,
            args.report,
            args.auto_report,
            args.html_report,
            args.auto_html_report,
            args.tee_report,
        )

        output = run_radar(debug=args.debug, csv_path=args.csv)
        LOGGER.info("Radar run completed | probability=%s regime=%s total_score=%s",
                    output.crash_probability_pct, output.regime, output.total_score)

        if args.json:
            print(json.dumps(asdict(output), indent=2, default=str))
            LOGGER.info("JSON output printed to console")
        else:
            report_path: Path | None = None
            html_report_path: Path | None = None

            if args.report:
                report_path = Path(args.report)
                report_path.parent.mkdir(parents=True, exist_ok=True)
            elif args.auto_report:
                report_path = build_report_filename(report_dir=args.report_dir)

            if args.html_report:
                html_report_path = Path(args.html_report)
                html_report_path.parent.mkdir(parents=True, exist_ok=True)
            elif args.auto_html_report:
                html_report_path = build_html_report_filename(report_dir=args.report_dir)

            if report_path is not None:
                write_report_to_file(
                    output,
                    report_path=report_path,
                    also_print_to_console=args.tee_report,
                )
                LOGGER.info("Formatted text report written to %s", report_path)

                if not args.tee_report:
                    print(f"Text report written to {report_path}")

            elif not html_report_path:
                print_report(output)

            if html_report_path is not None:
                write_html_report(output, html_report_path)
                LOGGER.info("HTML report written to %s", html_report_path)
                print(f"HTML report written to {html_report_path}")

            print(f"Runtime log written to {logfile}")

        if args.csv:
            save_csv_row(output, args.csv)
            LOGGER.info("CSV snapshot appended to %s", args.csv)

        LOGGER.info("Run finished successfully")
        return 0

    except Exception as e:
        LOGGER.exception("Run failed")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
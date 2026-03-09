from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import yfinance as yf


# =========================
# Config
# =========================

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED series
FRED_SERIES = {
    "credit_spread": "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS
    "us10y": "DGS10",                 # 10Y Treasury constant maturity
    "us2y": "DGS2",                   # 2Y Treasury constant maturity
    "fed_balance": "WALCL",           # Fed total assets
    "vix": "VIXCLS",                  # CBOE Volatility Index via FRED
    "m2": "WM2NS",
}

# Yahoo Finance tickers
YF_TICKERS = {
    "hyg": "HYG",
    "nasdaq": "^IXIC",
    "move": "^MOVE",   # sometimes available, sometimes not
    "usd": "DX-Y.NYB",
    "spy": "SPY",
    "rsp": "RSP",
}

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "cyan": "\033[36m",
}


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

    # sort back to old -> new for normal time series logic
    rows.sort(key=lambda x: x[0])

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
    elif cur >= 25 or chg_pct >= 25:
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
    prev_12w = value_days_ago(series, 12)  # weekly-ish for WALCL
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


def analyze_hyg(series: pd.Series) -> Indicator:
    s = series.dropna()
    cur = float(s.iloc[-1])
    ma20 = float(s.tail(20).mean()) if len(s) >= 20 else float(s.mean())
    ma50 = float(s.tail(50).mean()) if len(s) >= 50 else float(s.mean())
    ma200 = float(s.tail(200).mean()) if len(s) >= 200 else float(s.mean())

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


def analyze_nasdaq(series: pd.Series) -> Indicator:
    s = series.dropna()
    cur = float(s.iloc[-1])
    ma20 = float(s.tail(20).mean()) if len(s) >= 20 else float(s.mean())
    ma50 = float(s.tail(50).mean()) if len(s) >= 50 else float(s.mean())
    ma200 = float(s.tail(200).mean()) if len(s) >= 200 else float(s.mean())

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

    def analyze_usd_index(series: pd.Series) -> Indicator:
    s = series.dropna()
    cur = float(s.iloc[-1])
    ma20 = float(s.tail(20).mean())
    ma50 = float(s.tail(50).mean())
    ma200 = float(s.tail(200).mean()) if len(s) >= 200 else float(s.mean())

    if cur > ma50 and cur > ma200:
        status, score = "WARNING", 2
    elif cur > ma20 or cur > ma50:
        status, score = "ELEVATED", 1
    else:
        status, score = "OK", 0

    return Indicator(
        key="usd_index",
        name="USD Index",
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
    ma20 = float(ratio.tail(20).mean())
    ma50 = float(ratio.tail(50).mean())
    ma200 = float(ratio.tail(200).mean()) if len(ratio) >= 200 else float(ratio.mean())

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

    def analyze_vol_regime(vix_series: pd.Series) -> Indicator:
    s = vix_series.dropna()
    cur = float(s.iloc[-1])
    ma20 = float(s.tail(20).mean())
    ma50 = float(s.tail(50).mean())

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
# =========================
# Trigger / scoring logic
# =========================

def hedge_fund_trigger(indicators: dict[str, Indicator], hyg_series: pd.Series) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if indicators["vix"].value is not None and indicators["vix"].value >= 30:
        reasons.append("VIX spike > 30")

    if indicators["credit_spread"].value is not None and indicators["credit_spread"].value >= 6.0:
        reasons.append("Credit spread > 6")

    s = hyg_series.dropna()
    cur = float(s.iloc[-1])
    ma200 = float(s.tail(200).mean()) if len(s) >= 200 else float(s.mean())
    if cur < ma200:
        reasons.append("HYG below 200d MA")

    return len(reasons) >= 2, reasons


def calculate_probability(indicators: list[Indicator], trigger_on: bool) -> int:
    total_score = sum(i.score for i in indicators)

    # Heuristic mapping, not a trained prediction model
    base = 100 / (1 + math.exp(-(total_score - 6) / 2.2))
    if trigger_on:
        base += 15

    return int(round(clamp(base, 0, 100)))


def regime_from_probability(prob: int) -> str:
    if prob >= 75:
        return "CRASH WARNING"
    if prob >= 50:
        return "HIGH RISK"
    if prob >= 25:
        return "ELEVATED RISK"
    return "NORMAL"


def narrative(indicators: list[Indicator], regime: str, trigger_on: bool, reasons: list[str]) -> str:
    bad = [i.name for i in indicators if i.status in ("ALARM", "WARNING")]
    elevated = [i.name for i in indicators if i.status == "ELEVATED"]

    parts: list[str] = []

    if regime == "CRASH WARNING":
        parts.append("Meerdere kernmarkten tonen tegelijk stress. Dit is een uitgesproken risk-off regime.")
    elif regime == "HIGH RISK":
        parts.append("Er zijn meerdere betekenisvolle stress-signalen. De markt is kwetsbaar voor een scherpe correctie.")
    elif regime == "ELEVATED RISK":
        parts.append("Er zijn vroege waarschuwingssignalen, maar nog geen breed crisisbeeld.")
    else:
        parts.append("De belangrijkste macro-indicatoren ogen op dit moment relatief rustig.")

    if bad:
        parts.append("Belangrijkste stresspunten: " + ", ".join(bad) + ".")
    if elevated:
        parts.append("Verhoogde aandacht voor: " + ", ".join(elevated) + ".")
    if trigger_on:
        parts.append("Hedge-fund crash trigger actief: " + "; ".join(reasons) + ".")

    return " ".join(parts)


# =========================
# Output helpers
# =========================

def color_for_status(status: str) -> str:
    if status == "ALARM":
        return ANSI["red"]
    if status in ("WARNING", "ELEVATED", "SOFT"):
        return ANSI["yellow"]
    return ANSI["green"]


def print_report(output: RadarOutput) -> None:
    print()
    print(f"{ANSI['bold']}{ANSI['cyan']}MACRO CRASH RADAR{ANSI['reset']}")
    print(f"UTC: {output.timestamp_utc}")
    print("-" * 96)

    for ind in output.indicators:
        color = color_for_status(ind.status)
        value_txt = f"{ind.value:.2f}" if isinstance(ind.value, (int, float)) and ind.value is not None else "n/a"
        print(f"{ind.name:<24} {value_txt:>10}   {color}{ind.status:<10}{ANSI['reset']}   {ind.detail}")

    print("-" * 96)
    print(f"{ANSI['bold']}Total score:{ANSI['reset']} {output.total_score}")
    print(f"{ANSI['bold']}Crash probability:{ANSI['reset']} {output.crash_probability_pct}%")
    print(f"{ANSI['bold']}Risk regime:{ANSI['reset']} {output.regime}")
    print(f"{ANSI['bold']}Hedge-fund trigger:{ANSI['reset']} {'YES' if output.hedge_fund_trigger else 'NO'}")

    if output.trigger_reasons:
        print(f"{ANSI['bold']}Trigger reasons:{ANSI['reset']} " + "; ".join(output.trigger_reasons))

    print("-" * 96)
    print(output.narrative)
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
    }

    for ind in output.indicators:
        row[f"{ind.key}_value"] = ind.value
        row[f"{ind.key}_status"] = ind.status
        row[f"{ind.key}_score"] = ind.score
        row[f"{ind.key}_detail"] = ind.detail

    df = pd.DataFrame([row])

    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# =========================
# Main radar
# =========================

def run_radar() -> RadarOutput:
    # FRED data
    credit_spread = fred_series(FRED_SERIES["credit_spread"])
    us10y = fred_series(FRED_SERIES["us10y"])
    us2y = fred_series(FRED_SERIES["us2y"])
    fed_balance = fred_series(FRED_SERIES["fed_balance"])
    vix = fred_series(FRED_SERIES["vix"])
    m2 = fred_series(FRED_SERIES["m2"])

    print("DEBUG DGS10 latest:", us10y.tail(3))
    print("DEBUG DGS2 latest:", us2y.tail(3))
    print("DEBUG WALCL latest:", fed_balance.tail(3))

    # Yahoo data
    hyg = yf_close_series(YF_TICKERS["hyg"])
    nasdaq = yf_close_series(YF_TICKERS["nasdaq"])
    usd = yf_close_series(YF_TICKERS["usd"])
    spy = yf_close_series(YF_TICKERS["spy"])
    rsp = yf_close_series(YF_TICKERS["rsp"])



    # MOVE may fail, so handle gracefully
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

    indicators_list = [
        analyze_vix(vix),
        analyze_vol_regime(vix),
        analyze_credit_spread(credit_spread),
        move_indicator,
        analyze_yield_curve(us10y, us2y),
        analyze_fed_balance(fed_balance),
        analyze_global_liquidity(fed_balance, m2),
        analyze_usd_index(usd),
        analyze_hyg(hyg),
        analyze_sp_breadth(rsp, spy),
        analyze_nasdaq(nasdaq),
]

    indicators = {i.key: i for i in indicators_list}
    trigger_on, reasons = hedge_fund_trigger(indicators, hyg)
    total_score = sum(i.score for i in indicators_list)
    prob = calculate_probability(indicators_list, trigger_on)
    regime = regime_from_probability(prob)

    return RadarOutput(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        total_score=total_score,
        crash_probability_pct=prob,
        regime=regime,
        hedge_fund_trigger=trigger_on,
        trigger_reasons=reasons,
        narrative=narrative(indicators_list, regime, trigger_on, reasons),
        indicators=indicators_list,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Macro Crash Radar")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of terminal report")
    parser.add_argument("--csv", type=str, default="", help="Append one snapshot row to CSV")
    args = parser.parse_args()

    try:
        output = run_radar()

        if args.json:
            print(json.dumps(asdict(output), indent=2, default=str))
        else:
            print_report(output)

        if args.csv:
            save_csv_row(output, args.csv)

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
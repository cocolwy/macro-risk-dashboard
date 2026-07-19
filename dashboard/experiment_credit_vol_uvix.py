"""
F002 · Credit Spread Shock → UVIX

Pre-registered protocol (cases/F002_credit_vol_uvix.json):
  - Signal: BAA10Y Δ5 >= +0.10 (10bps widening)
  - Action: long UVIX close(t) → close(t+5)
  - Baseline: non-signal anchors, stride=5, same hold
  - Primary: hit rate + mean return, Bonferroni α=0.025
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiment_event_uvix import pct_return, proportion_test, trading_offset
from experiment_uvix_event import fetch_uvix_history, uvix_close_series
from fetch_macro_data import fetch_fred_series, sync_public_data

DATA_DIR = Path(__file__).parent / "data"

DELTA_WINDOW = 5
TRIGGER_DELTA = 0.10
HOLD_DAYS = 5
BASELINE_STRIDE = 5
N_PRIMARY_TESTS = 2
BONFERRONI_ALPHA = round(0.05 / N_PRIMARY_TESTS, 5)
MIN_SIGNALS = 20
BASE_COST_BPS = 104.02
STRESS_COST_BPS = 260.05
RECENT_YEARS = 2


def fetch_baa_spread() -> pd.Series:
    s = fetch_fred_series("BAA10Y", start="2000-01-01")
    s.index = pd.to_datetime(s.index).normalize()
    s.name = "baa_spread"
    return s.sort_index()


def spread_delta_on_calendar(spread: pd.Series, trading_index: pd.DatetimeIndex) -> pd.Series:
    aligned = spread.reindex(trading_index).ffill()
    delta = aligned - aligned.shift(DELTA_WINDOW)
    delta.name = "delta5"
    return delta


def hold_return(uvix: pd.Series, entry_day: pd.Timestamp, hold_days: int = HOLD_DAYS) -> float | None:
    idx = uvix.index
    if entry_day not in idx:
        return None
    exit_day = trading_offset(idx, entry_day, hold_days)
    return pct_return(uvix, entry_day, exit_day)


def collect_signal_trades(
    uvix: pd.Series,
    delta5: pd.Series,
) -> tuple[list[dict], set]:
    """Returns signal trades and excluded anchor days (entry + hold window)."""
    idx = uvix.index
    trades: list[dict] = []
    excluded: set = set()
    active_exit: pd.Timestamp | None = None

    for i in range(DELTA_WINDOW, len(idx)):
        t = idx[i]
        d = delta5.loc[t]
        if pd.isna(d):
            continue
        if active_exit is not None and t <= active_exit:
            continue
        if d < TRIGGER_DELTA:
            continue
        ret = hold_return(uvix, t)
        if ret is None:
            continue
        exit_day = trading_offset(idx, t, HOLD_DAYS)
        trades.append({
            "entry_date": t.strftime("%Y-%m-%d"),
            "exit_date": exit_day.strftime("%Y-%m-%d") if exit_day else None,
            "delta5": round(float(d), 4),
            "return_pct": ret,
        })
        active_exit = exit_day
        pos_entry = idx.get_loc(t)
        for j in range(pos_entry, pos_entry + HOLD_DAYS + 1):
            if j < len(idx):
                excluded.add(idx[j])

    return trades, excluded


def collect_baseline_returns(
    uvix: pd.Series,
    delta5: pd.Series,
    signal_excluded: set,
) -> list[float]:
    idx = uvix.index
    out: list[float] = []
    for i, t in enumerate(idx):
        if i < HOLD_DAYS:
            continue
        if i % BASELINE_STRIDE != 0:
            continue
        if t in signal_excluded:
            continue
        d = delta5.loc[t]
        if pd.isna(d) or d >= TRIGGER_DELTA:
            continue
        r = hold_return(uvix, t)
        if r is not None:
            out.append(r)
    return out


def welch_mean_test(signal_vals: list, base_vals: list, alpha: float) -> dict:
    if len(signal_vals) < 5 or len(base_vals) < 5:
        return {"significant": False, "p_value": None, "event_mean_pct": None, "baseline_mean_pct": None,
                "excess_mean_pct": None, "event_n": len(signal_vals), "baseline_n": len(base_vals)}
    t_stat, p_two = stats.ttest_ind(signal_vals, base_vals, equal_var=False)
    p_one = float(p_two / 2) if t_stat > 0 else 1.0
    ev_m, ba_m = float(np.mean(signal_vals)), float(np.mean(base_vals))
    return {
        "event_mean_pct": round(ev_m, 3),
        "baseline_mean_pct": round(ba_m, 3),
        "excess_mean_pct": round(ev_m - ba_m, 3),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(p_one, 4),
        "significant": p_one < alpha,
        "alpha_used": alpha,
        "event_n": len(signal_vals),
        "baseline_n": len(base_vals),
    }


def hit_rate_test(signal_vals: list, base_vals: list, alpha: float) -> dict:
    if not signal_vals or not base_vals:
        return {"significant": False, "p_value": None}
    ev_succ = sum(1 for r in signal_vals if r > 0)
    ba_succ = sum(1 for r in base_vals if r > 0)
    test = proportion_test(ev_succ, len(signal_vals), ba_succ, len(base_vals), one_sided=True)
    ev_rate = ev_succ / len(signal_vals)
    ba_rate = ba_succ / len(base_vals)
    excess = ev_rate - ba_rate
    sig = test["p_value"] is not None and test["p_value"] < alpha and excess > 0
    return {
        "event_hit_rate": round(ev_rate, 3),
        "baseline_hit_rate": round(ba_rate, 3),
        "excess_hit_rate": round(excess, 3),
        "z_stat": test["z_stat"],
        "p_value": test["p_value"],
        "significant": sig,
        "alpha_used": alpha,
        "event_n": len(signal_vals),
        "baseline_n": len(base_vals),
    }


def recent_subsample(trades: list[dict], uvix: pd.Series, years: int = RECENT_YEARS) -> dict:
    cutoff = uvix.index[-1] - pd.DateOffset(years=years)
    rets = [t["return_pct"] for t in trades if pd.Timestamp(t["entry_date"]) >= cutoff]
    if len(rets) < 3:
        return {"error": "近 2y 信号不足", "n_signals": len(rets), "gate_pass": False}
    mean_ret = float(np.mean(rets))
    t_stat, p_two = stats.ttest_1samp(rets, 0.0)
    p_one_neg = float(p_two / 2) if t_stat < 0 else 1.0
    return {
        "years": years,
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "n_signals": len(rets),
        "mean_return_pct": round(mean_ret, 3),
        "hit_rate_up": round(sum(1 for r in rets if r > 0) / len(rets), 3),
        "p_one_sided_negative": round(p_one_neg, 4),
        "gate_pass": mean_ret > 0 or p_one_neg >= 0.05,
    }


def evaluate_gate(h1_mean: dict, h1_hit: dict, recent: dict) -> dict:
    ex = h1_mean.get("excess_mean_pct")
    cost_adj = round(ex - BASE_COST_BPS / 100, 3) if ex is not None else None
    checks = {
        "h1_mean_bonferroni": {
            "pass": bool(h1_mean.get("significant") and (ex or 0) > 0),
            "p_value": h1_mean.get("p_value"),
            "alpha": BONFERRONI_ALPHA,
        },
        "h1_hit_rate_bonferroni": {
            "pass": bool(h1_hit.get("significant")),
            "p_value": h1_hit.get("p_value"),
            "alpha": BONFERRONI_ALPHA,
        },
        "cost_adjusted_excess_positive": {
            "pass": cost_adj is not None and cost_adj > 0,
            "cost_adjusted_excess_pct": cost_adj,
            "base_total_bps": BASE_COST_BPS,
        },
        "recent_2y_not_significantly_negative": {
            "pass": recent.get("gate_pass", False),
            "detail": recent,
        },
    }
    # F002 gate per preregister: mean significant + cost + recent 2y
    all_pass = (
        checks["h1_mean_bonferroni"]["pass"]
        and checks["cost_adjusted_excess_positive"]["pass"]
        and checks["recent_2y_not_significantly_negative"]["pass"]
    )
    return {
        "all_pass": all_pass,
        "verdict": "pass" if all_pass else "fail — dead per F002 preregister",
        "checks": checks,
    }


def build_output() -> dict:
    print("Fetching BAA10Y spread...")
    spread = fetch_baa_spread()
    print(f"  Spread: {len(spread)} days, {spread.index[0].date()} ~ {spread.index[-1].date()}")

    print("Fetching UVIX...")
    uvix_df = fetch_uvix_history()
    uvix = uvix_close_series(uvix_df)
    print(f"  UVIX: {len(uvix)} days, {uvix.index[0].date()} ~ {uvix.index[-1].date()}")

    delta5 = spread_delta_on_calendar(spread, uvix.index)
    trades, excluded = collect_signal_trades(uvix, delta5)
    baseline_rets = collect_baseline_returns(uvix, delta5, excluded)
    signal_rets = [t["return_pct"] for t in trades]

    print(f"  Signals: {len(trades)} · Baseline anchors: {len(baseline_rets)}")

    insufficient = len(trades) < MIN_SIGNALS
    h1_hit = hit_rate_test(signal_rets, baseline_rets, BONFERRONI_ALPHA)
    h1_mean = welch_mean_test(signal_rets, baseline_rets, BONFERRONI_ALPHA)
    recent = recent_subsample(trades, uvix)
    gate = evaluate_gate(h1_mean, h1_hit, recent)

    if h1_mean.get("excess_mean_pct") is not None:
        h1_mean["cost_adjusted_excess_pct"] = {
            "base": round(h1_mean["excess_mean_pct"] - BASE_COST_BPS / 100, 3),
            "stress": round(h1_mean["excess_mean_pct"] - STRESS_COST_BPS / 100, 3),
        }

    conclusion_parts = []
    if insufficient:
        conclusion_parts.append(f"有效信号 n={len(trades)} < {MIN_SIGNALS}，主检验标 insufficient")
    elif h1_mean.get("significant"):
        conclusion_parts.append(
            f"信号后 UVIX 5 日平均收益 {h1_mean['event_mean_pct']}% vs 基准 {h1_mean['baseline_mean_pct']}%"
            f"（Bonferroni p={h1_mean['p_value']}）"
        )
    else:
        conclusion_parts.append(
            f"主检验不显著：信号均值 {h1_mean.get('event_mean_pct')}% vs 基准 {h1_mean.get('baseline_mean_pct')}%"
            f"（p={h1_mean.get('p_value')}）"
        )
    conclusion_parts.append(f"S6 门禁 preview: {'PASS' if gate['all_pass'] else 'FAIL'}")

    return {
        "title": "Credit Spread Shock × UVIX (F002)",
        "subtitle": "BAA10Y Δ5≥10bps → long UVIX 5d",
        "case_id": "F002",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "signal": "FRED BAA10Y",
            "trigger": f"Δ{DELTA_WINDOW} >= +{TRIGGER_DELTA} (10bps)",
            "hold": f"T+0 close → T+{HOLD_DAYS} close",
            "data_range": f"{uvix.index[0].strftime('%Y-%m-%d')} ~ {uvix.index[-1].strftime('%Y-%m-%d')}",
            "n_signals": len(trades),
            "n_baseline": len(baseline_rets),
            "insufficient_sample": insufficient,
        },
        "primary_analysis": {
            "method": f"Bonferroni α={BONFERRONI_ALPHA} ({N_PRIMARY_TESTS} tests) · baseline stride={BASELINE_STRIDE}",
            "h1_hit_rate": h1_hit,
            "h1_mean_return": h1_mean,
            "conclusion_summary": "；".join(conclusion_parts) + "。",
        },
        "signal_trades": trades[-30:],
        "signal_trades_total": len(trades),
        "recent_subsample": recent,
        "s6_gate_preview": gate,
        "costs": {"base_total_bps": BASE_COST_BPS, "stress_total_bps": STRESS_COST_BPS},
    }


def main():
    output = build_output()
    out_path = DATA_DIR / "credit_vol_uvix_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"  S6 gate: {'PASS' if output['s6_gate_preview']['all_pass'] else 'FAIL'}")
    sync_public_data()


if __name__ == "__main__":
    main()

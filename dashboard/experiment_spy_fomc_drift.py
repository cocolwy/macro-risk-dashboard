"""
F003 · Pre-FOMC SPX Drift → SPY

Pre-registered protocol (cases/F003_spx_fomc_drift.json):
  - FOMC calendar only
  - H1: T-5 close → T-1 close long SPY vs sparse non-FOMC baseline
  - Bonferroni α=0.025 (2 tests)
  - Post-2015 subsample gate
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiment_event_uvix import (
    BASELINE_STRIDE,
    PRIMARY_PRE_WINDOW,
    baseline_pre_returns,
    collect_event_trading_days,
    event_pre_returns,
    filter_event_dates_in_range,
    get_event_dates,
    get_historical_fomc_dates,
    proportion_test,
    test_window,
    window_label_pre,
)
from fetch_macro_data import sync_public_data

DATA_DIR = Path(__file__).parent / "data"

N_PRIMARY_TESTS = 2
BONFERRONI_ALPHA = round(0.05 / N_PRIMARY_TESTS, 5)
MIN_FOMC_EVENTS = 40
BASE_COST_BPS = 12.0
STRESS_COST_BPS = 30.0
POST_2015_SPLIT = pd.Timestamp("2015-01-01")


def fetch_spy_close() -> pd.Series:
    """Daily adjusted close; Yahoo range=max is weekly — use period1/period2."""
    start = pd.Timestamp("1993-01-29")
    end = pd.Timestamp.utcnow().normalize()
    p1 = int(start.timestamp())
    p2 = int(end.timestamp())
    url = (
        "https://query2.finance.yahoo.com/v8/finance/chart/SPY"
        f"?interval=1d&period1={p1}&period2={p2}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    rows = []
    for i, t in enumerate(ts):
        price = (adj[i] if adj and adj[i] is not None else quote["close"][i])
        if price is None:
            continue
        rows.append((pd.Timestamp.utcfromtimestamp(t).tz_localize(None).normalize(), float(price)))
    if not rows:
        raise RuntimeError("SPY chart API returned no rows")
    s = pd.Series(dict(rows)).sort_index()
    s.name = "SPY"
    return s


def welch_mean_test(event_vals: list, base_vals: list, alpha: float) -> dict:
    if len(event_vals) < 5 or len(base_vals) < 5:
        return {"significant": False, "p_value": None}
    t_stat, p_two = stats.ttest_ind(event_vals, base_vals, equal_var=False)
    p_one = float(p_two / 2) if t_stat > 0 else 1.0
    ev_m, ba_m = float(np.mean(event_vals)), float(np.mean(base_vals))
    return {
        "event_mean_pct": round(ev_m, 3),
        "baseline_mean_pct": round(ba_m, 3),
        "excess_mean_pct": round(ev_m - ba_m, 3),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(p_one, 4),
        "significant": p_one < alpha,
        "alpha_used": alpha,
        "event_n": len(event_vals),
        "baseline_n": len(base_vals),
    }


def subsample_returns(spy: pd.Series, fomc_dates: list, cutoff: pd.Timestamp | None, before: bool = False) -> list:
    if cutoff is not None:
        if before:
            dates = [d for d in fomc_dates if pd.Timestamp(d) < cutoff]
        else:
            dates = [d for d in fomc_dates if pd.Timestamp(d) >= cutoff]
    else:
        dates = fomc_dates
    pre_start, pre_end = PRIMARY_PRE_WINDOW
    return event_pre_returns(spy, dates, pre_start, pre_end)


def post_2015_gate(spy: pd.Series, fomc_dates: list) -> dict:
    rets = subsample_returns(spy, fomc_dates, POST_2015_SPLIT, before=False)
    if len(rets) < 5:
        return {"error": "Post-2015 FOMC 样本不足", "gate_pass": False}
    mean_ret = float(np.mean(rets))
    t_stat, p_two = stats.ttest_1samp(rets, 0.0)
    p_one_neg = float(p_two / 2) if t_stat < 0 else 1.0
    return {
        "split_date": "2015-01-01",
        "n_events": len(rets),
        "mean_return_pct": round(mean_ret, 3),
        "hit_rate_up": round(sum(1 for r in rets if r > 0) / len(rets), 3),
        "p_one_sided_negative": round(p_one_neg, 4),
        "gate_pass": mean_ret > 0 or p_one_neg >= 0.05,
    }


def evaluate_gate(h1_mean: dict, h1_hit: dict, post2015: dict) -> dict:
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
        "post_2015_not_significantly_negative": {
            "pass": post2015.get("gate_pass", False),
            "detail": post2015,
        },
    }
    all_pass = (
        checks["h1_mean_bonferroni"]["pass"]
        and checks["cost_adjusted_excess_positive"]["pass"]
        and checks["post_2015_not_significantly_negative"]["pass"]
    )
    return {
        "all_pass": all_pass,
        "verdict": "pass" if all_pass else "fail — dead per F003 preregister",
        "checks": checks,
    }


def build_output() -> dict:
    print("Fetching SPY daily (1993+)...")
    spy = fetch_spy_close()
    print(f"  SPY: {len(spy)} days, {spy.index[0].date()} ~ {spy.index[-1].date()}")

    hist_fomc = get_historical_fomc_dates()
    curr_fomc = get_event_dates()["fomc"]
    all_fomc = sorted(set(hist_fomc + curr_fomc))
    fomc_in_range = filter_event_dates_in_range(spy, all_fomc, max_pre=5, max_post=1)
    print(f"  FOMC in range: {len(fomc_in_range)}")

    fomc_days = collect_event_trading_days(spy, fomc_in_range)
    pre_start, pre_end = PRIMARY_PRE_WINDOW

    ev_pre = event_pre_returns(spy, fomc_in_range, pre_start, pre_end)
    ba_pre = baseline_pre_returns(spy, fomc_days, pre_start, pre_end, stride=BASELINE_STRIDE)

    h1_hit = test_window(ev_pre, ba_pre, "up", one_sided=True, alpha=BONFERRONI_ALPHA)
    h1_mean = welch_mean_test(ev_pre, ba_pre, BONFERRONI_ALPHA)
    if h1_mean.get("excess_mean_pct") is not None:
        h1_mean["cost_adjusted_excess_pct"] = {
            "base": round(h1_mean["excess_mean_pct"] - BASE_COST_BPS / 100, 3),
            "stress": round(h1_mean["excess_mean_pct"] - STRESS_COST_BPS / 100, 3),
        }

    post2015 = post_2015_gate(spy, fomc_in_range)
    gate = evaluate_gate(h1_mean, h1_hit, post2015)

    pre_rets = subsample_returns(spy, fomc_in_range, POST_2015_SPLIT, before=True)
    post_rets = subsample_returns(spy, fomc_in_range, POST_2015_SPLIT, before=False)

    insufficient = len(ev_pre) < MIN_FOMC_EVENTS

    def period_stats(rets: list) -> dict:
        if not rets:
            return {"n": 0}
        return {
            "n": len(rets),
            "mean_pct": round(float(np.mean(rets)), 3),
            "hit_rate_up": round(sum(1 for r in rets if r > 0) / len(rets), 3),
        }

    exploratory_split = {
        "description": "Lucca & Moench (2015) 发表后衰减 · 探索性（非主门禁 family）",
        "split_date": "2015-01-01",
        "pre_2015": period_stats(pre_rets),
        "post_2015": period_stats(post_rets),
    }

    win = window_label_pre(*PRIMARY_PRE_WINDOW)
    if insufficient:
        conclusion = f"有效 FOMC n={len(ev_pre)} < {MIN_FOMC_EVENTS}，主检验 insufficient"
    elif h1_mean.get("significant"):
        conclusion = (
            f"FOMC 前 {win} SPY 平均 {h1_mean['event_mean_pct']}% vs 基准 {h1_mean['baseline_mean_pct']}%"
            f"（Bonferroni p={h1_mean['p_value']}）"
        )
    else:
        conclusion = (
            f"FOMC 前 {win} 主检验不显著：SPY {h1_mean.get('event_mean_pct')}% vs 基准"
            f" {h1_mean.get('baseline_mean_pct')}%（p={h1_mean.get('p_value')}）"
        )

    return {
        "title": "Pre-FOMC SPX Drift × SPY (F003)",
        "subtitle": f"FOMC {win} · long SPY",
        "case_id": "F003",
        "reference": "Lucca & Moench (2015) Journal of Finance",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "instrument": "SPY",
            "window": win,
            "data_range": f"{spy.index[0].strftime('%Y-%m-%d')} ~ {spy.index[-1].strftime('%Y-%m-%d')}",
            "n_fomc_events": len(ev_pre),
            "n_baseline": len(ba_pre),
            "insufficient_sample": insufficient,
        },
        "primary_analysis": {
            "method": f"Bonferroni α={BONFERRONI_ALPHA} ({N_PRIMARY_TESTS} tests) · baseline stride={BASELINE_STRIDE}",
            "h1_pre_rise_hit_rate": h1_hit,
            "h1_pre_rise_mean_return": h1_mean,
            "conclusion_summary": conclusion + f"；S6 门禁: {'PASS' if gate['all_pass'] else 'FAIL'}。",
        },
        "exploratory_pre_post_2015": exploratory_split,
        "post_2015_gate": post2015,
        "s6_gate_preview": gate,
        "costs": {"base_total_bps": BASE_COST_BPS, "stress_total_bps": STRESS_COST_BPS},
    }


def main():
    output = build_output()
    out_path = DATA_DIR / "spy_fomc_drift_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"  S6 gate: {'PASS' if output['s6_gate_preview']['all_pass'] else 'FAIL'}")
    sync_public_data()


if __name__ == "__main__":
    main()

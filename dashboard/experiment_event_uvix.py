"""
Event Calendar × VIX Analysis

Per-event window sweep: FOMC / CPI / NFP each try multiple pre/post windows,
compare hit rates vs non-event baseline, pick best window per hypothesis.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from predict_model import build_event_features, get_event_dates
from fetch_macro_data import fetch_yfinance_history, sync_public_data

DATA_DIR = Path(__file__).parent / "data"
ANALYSIS_YEARS = 10
VIX_PERIOD = f"{ANALYSIS_YEARS}y"
EVENT_START_YEAR = 2016
EVENT_END_YEAR = 2027

# Pre-registered primary windows (no post-hoc selection)
PRIMARY_PRE_WINDOW = (5, 1)   # T-5 ~ T-1
PRIMARY_POST_WINDOW = 1       # T+1 close-to-close
BASELINE_STRIDE = 5           # decorrelate overlapping baseline anchors
N_PRIMARY_TESTS = 6           # 3 events × 2 hypotheses
BONFERRONI_ALPHA = round(0.05 / N_PRIMARY_TESTS, 5)
COVID_EXCLUDE_START = pd.Timestamp("2020-03-01")
COVID_EXCLUDE_END = pd.Timestamp("2020-06-30")

EVENT_FEATURES = [
    "fomc_days_to", "fomc_days_since", "fomc_within_3d", "fomc_within_7d",
    "cpi_days_to", "cpi_days_since", "cpi_within_3d", "cpi_within_7d",
    "nfp_within_3d",
]

# Per-event candidate windows (trading days)
EVENT_WINDOW_CONFIG = {
    "fomc": {
        "label": "FOMC",
        "pre": [(10, 1), (7, 1), (5, 1), (3, 1)],   # T-N ~ T-1
        "post": [1, 2, 3, 5],                         # T+0 ~ T+N
    },
    "cpi": {
        "label": "CPI",
        "pre": [(10, 1), (7, 1), (5, 1), (3, 1)],
        "post": [1, 2, 3, 5],
    },
    "nfp": {
        "label": "NFP",
        "pre": [(7, 1), (5, 1), (3, 1), (2, 1)],
        "post": [1, 3, 5],
    },
}


def get_nfp_dates(start_year=EVENT_START_YEAR, end_year=EVENT_END_YEAR):
    dates = set()
    for year in range(start_year, end_year):
        for month in range(1, 13):
            first = pd.Timestamp(year, month, 1)
            offset = (4 - first.weekday()) % 7
            dates.add(first + pd.Timedelta(days=offset))
    return sorted(dates)


def add_nfp_days_to(features: pd.DataFrame, dates_index: pd.DatetimeIndex) -> pd.DataFrame:
    nfp_list = get_nfp_dates()
    days_to, days_since = [], []
    for d in dates_index:
        future = [e for e in nfp_list if e >= d]
        past = [e for e in nfp_list if e <= d]
        days_to.append((future[0] - d).days if future else 30)
        days_since.append((d - past[-1]).days if past else 30)
    features["nfp_days_to"] = days_to
    features["nfp_days_since"] = days_since
    return features


def fetch_vix_series() -> pd.Series:
    df = fetch_yfinance_history("^VIX", period=VIX_PERIOD)
    if df.empty:
        raise RuntimeError("VIX data unavailable")
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.dropna()


def event_trading_day(idx: pd.DatetimeIndex, event_date: pd.Timestamp):
    if event_date in idx:
        return event_date
    future = idx[idx >= event_date]
    return future[0] if len(future) else None


def trading_offset(idx: pd.DatetimeIndex, anchor, offset: int):
    pos = idx.get_loc(anchor)
    target = pos + offset
    return idx[target] if 0 <= target < len(idx) else None


def pct_return(series: pd.Series, start, end) -> Optional[float]:
    if start is None or end is None or start >= end:
        return None
    p0, p1 = series.loc[start], series.loc[end]
    if p0 <= 0:
        return None
    return round((p1 / p0 - 1) * 100, 3)


def window_label_pre(start_off: int, end_off: int = 1) -> str:
    return f"T-{start_off}~T-{end_off}"


def window_label_post(post_off: int) -> str:
    return f"T+0~T+{post_off}" if post_off > 1 else "T+1"


def proportion_test(success_a: int, n_a: int, success_b: int, n_b: int, one_sided: bool = False) -> dict:
    if n_a < 5 or n_b < 5:
        return {"z_stat": None, "p_value": None}
    p1, p2 = success_a / n_a, success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)
    if p_pool in (0, 1):
        return {"z_stat": None, "p_value": None}
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p1 - p2) / se
    if one_sided:
        p_val = float(1 - stats.norm.cdf(z)) if z > 0 else 1.0
    else:
        p_val = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"z_stat": round(float(z), 3), "p_value": round(p_val, 4)}


def phi_correlation(flags: list, outcomes: list) -> dict:
    if len(flags) < 30 or len(set(flags)) < 2 or len(set(outcomes)) < 2:
        return {"phi": None, "p_value": None}
    phi, p = stats.pearsonr(np.array(flags), np.array(outcomes))
    return {"phi": round(float(phi), 4), "p_value": round(float(p), 4)}


def _verdict(sig: bool, event_rate, base_rate, want_higher: bool) -> str:
    if event_rate is None or base_rate is None:
        return "数据不足"
    diff = event_rate - base_rate
    if not sig:
        return "无显著相关" if abs(diff) < 0.05 else "有差异但不显著"
    return "显著正相关" if (diff > 0) == want_higher else "显著负相关"


def filter_event_dates_in_range(vix: pd.Series, event_dates: list, max_pre: int = 10, max_post: int = 5) -> list:
    """Keep events with enough trading days before/after for the widest tested window."""
    idx = vix.index
    out = []
    for ed in event_dates:
        t0 = event_trading_day(idx, pd.Timestamp(ed))
        if t0 is None:
            continue
        pos = idx.get_loc(t0)
        if pos >= max_pre and pos + max_post < len(idx):
            out.append(pd.Timestamp(ed))
    return out


def collect_event_trading_days(vix: pd.Series, event_dates: list) -> set:
    return {t for d in event_dates if (t := event_trading_day(vix.index, pd.Timestamp(d))) is not None}


# ── Level-based measurement helpers ──────────────────────────────────────────

def event_vix_window_avg(vix: pd.Series, event_dates: list, start_off: int, end_off: int = 1) -> list:
    """Average VIX level across T-start_off … T-end_off before each event."""
    idx = vix.index
    out = []
    for ed in event_dates:
        t0 = event_trading_day(idx, pd.Timestamp(ed))
        if t0 is None:
            continue
        d_start = trading_offset(idx, t0, -start_off)
        d_end = trading_offset(idx, t0, -end_off)
        if d_start is None or d_end is None or d_start >= d_end:
            continue
        vals = vix.loc[d_start:d_end]
        if len(vals) >= 2:
            out.append(float(vals.mean()))
    return out


def event_vix_t1_level(vix: pd.Series, event_dates: list) -> list:
    """VIX level at T-1 (one trading day before event)."""
    idx = vix.index
    out = []
    for ed in event_dates:
        t0 = event_trading_day(idx, pd.Timestamp(ed))
        if t0 is None:
            continue
        t1 = trading_offset(idx, t0, -1)
        if t1 is not None:
            out.append(float(vix.loc[t1]))
    return out


def baseline_vix_window_avg(vix: pd.Series, exclude: set, start_off: int, end_off: int = 1, stride: int = 1) -> list:
    """Average VIX level in window for non-event anchor days."""
    idx = vix.index
    out = []
    for i, t0 in enumerate(idx):
        if t0 in exclude:
            continue
        if stride > 1 and i % stride != 0:
            continue
        d_start = trading_offset(idx, t0, -start_off)
        d_end = trading_offset(idx, t0, -end_off)
        if d_start is None or d_end is None or d_start >= d_end:
            continue
        vals = vix.loc[d_start:d_end]
        if len(vals) >= 2:
            out.append(float(vals.mean()))
    return out


def welch_ttest_greater(event_vals: list, base_vals: list, alpha: float = 0.05) -> dict:
    """One-sided Welch t-test: H1 event_mean > base_mean."""
    if len(event_vals) < 5 or len(base_vals) < 5:
        return {"t_stat": None, "p_value": None, "significant": False}
    t_stat, _ = stats.ttest_ind(event_vals, base_vals, equal_var=False)
    p_one = float(1 - stats.norm.cdf(t_stat)) if t_stat > 0 else 1.0
    sig = p_one < alpha
    return {
        "t_stat": round(float(t_stat), 3),
        "p_value": round(p_one, 4),
        "event_mean": round(float(np.mean(event_vals)), 2),
        "baseline_mean": round(float(np.mean(base_vals)), 2),
        "excess": round(float(np.mean(event_vals)) - float(np.mean(base_vals)), 2),
        "event_n": len(event_vals),
        "baseline_n": len(base_vals),
        "significant": sig,
        "alpha_used": alpha,
        "verdict": ("VIX 显著偏高" if sig else ("VIX 偏高但不显著" if float(np.mean(event_vals)) > float(np.mean(base_vals)) else "VIX 无差异")),
    }


def build_level_analysis(vix: pd.Series, date_map: dict, all_event_days: set) -> dict:
    """
    Method A: compare avg VIX level in T-5~T-1 window vs non-event baseline.
    Method B: compare VIX at T-1 vs full-sample unconditional mean.
    """
    pre_start, pre_end = PRIMARY_PRE_WINDOW
    full_mean = round(float(vix.mean()), 2)

    method_a_rows = []
    method_b_rows = []
    ba_window = baseline_vix_window_avg(vix, all_event_days, pre_start, pre_end, stride=BASELINE_STRIDE)

    for et, label in [("fomc", "FOMC"), ("cpi", "CPI"), ("nfp", "NFP")]:
        dates = date_map[et]

        # Method A
        ev_window = event_vix_window_avg(vix, dates, pre_start, pre_end)
        res_a = welch_ttest_greater(ev_window, ba_window, alpha=BONFERRONI_ALPHA)
        method_a_rows.append({"event": label, **res_a})

        # Method B
        ev_t1 = event_vix_t1_level(vix, dates)
        res_b = welch_ttest_greater(ev_t1, list(vix), alpha=BONFERRONI_ALPHA)
        method_b_rows.append({"event": label, **res_b})

    return {
        "description_a": (
            f"Method A · 窗口均值比较：T-{pre_start}~T-{pre_end} 期间 VIX 平均水平 "
            f"vs 非事件日同窗口基准（稀疏 stride={BASELINE_STRIDE}）"
        ),
        "description_b": (
            f"Method B · 绝对水平比较：T-1 日 VIX 水平 vs 全样本 VIX 均值 {full_mean}"
        ),
        "full_sample_mean_vix": full_mean,
        "method_a": method_a_rows,
        "method_b": method_b_rows,
        "note": "统计检验为单侧 Welch t-test，Bonferroni α 同主结论",
    }


def event_pre_returns(vix: pd.Series, event_dates: list, start_off: int, end_off: int = 1) -> list:
    idx = vix.index
    out = []
    for ed in event_dates:
        t0 = event_trading_day(idx, pd.Timestamp(ed))
        if t0 is None:
            continue
        r = pct_return(vix, trading_offset(idx, t0, -start_off), trading_offset(idx, t0, -end_off))
        if r is not None:
            out.append(r)
    return out


def event_post_returns(vix: pd.Series, event_dates: list, post_off: int) -> list:
    idx = vix.index
    out = []
    for ed in event_dates:
        t0 = event_trading_day(idx, pd.Timestamp(ed))
        if t0 is None:
            continue
        r = pct_return(vix, t0, trading_offset(idx, t0, post_off))
        if r is not None:
            out.append(r)
    return out


def baseline_pre_returns(vix: pd.Series, exclude: set, start_off: int, end_off: int = 1, stride: int = 1) -> list:
    idx = vix.index
    out = []
    for i, t0 in enumerate(idx):
        if t0 in exclude:
            continue
        if stride > 1 and i % stride != 0:
            continue
        r = pct_return(vix, trading_offset(idx, t0, -start_off), trading_offset(idx, t0, -end_off))
        if r is not None:
            out.append(r)
    return out


def baseline_post_returns(vix: pd.Series, exclude: set, post_off: int, stride: int = 1) -> list:
    idx = vix.index
    out = []
    for i, t0 in enumerate(idx):
        if t0 in exclude:
            continue
        if stride > 1 and i % stride != 0:
            continue
        r = pct_return(vix, t0, trading_offset(idx, t0, post_off))
        if r is not None:
            out.append(r)
    return out


def test_window(
    event_vals: list,
    base_vals: list,
    direction: str = "up",
    one_sided: bool = False,
    alpha: float = 0.05,
) -> dict:
    if direction == "up":
        ev_succ = sum(1 for r in event_vals if r > 0)
        ba_succ = sum(1 for r in base_vals if r > 0)
    else:
        ev_succ = sum(1 for r in event_vals if r < 0)
        ba_succ = sum(1 for r in base_vals if r < 0)
    ev_n, ba_n = len(event_vals), len(base_vals)
    ev_rate = round(ev_succ / ev_n, 3) if ev_n else None
    ba_rate = round(ba_succ / ba_n, 3) if ba_n else None
    test = (
        proportion_test(ev_succ, ev_n, ba_succ, ba_n, one_sided=one_sided)
        if ev_n and ba_n else {"z_stat": None, "p_value": None}
    )
    excess = round(ev_rate - ba_rate, 3) if ev_rate is not None and ba_rate is not None else None
    sig = test["p_value"] is not None and test["p_value"] < alpha and excess is not None and excess > 0
    phi = phi_correlation(
        [1] * ev_n + [0] * ba_n,
        ([1 if r > 0 else 0 for r in event_vals] if direction == "up"
         else [1 if r < 0 else 0 for r in event_vals])
        + ([1 if r > 0 else 0 for r in base_vals] if direction == "up"
           else [1 if r < 0 else 0 for r in base_vals]),
    )
    return {
        "event_hit_rate": ev_rate,
        "baseline_hit_rate": ba_rate,
        "excess_hit_rate": excess,
        "event_mean_pct": round(float(np.mean(event_vals)), 3) if event_vals else None,
        "baseline_mean_pct": round(float(np.mean(base_vals)), 3) if base_vals else None,
        "event_n": ev_n,
        "baseline_n": ba_n,
        "z_stat": test["z_stat"],
        "p_value": test["p_value"],
        "phi": phi["phi"],
        "significant": sig,
        "alpha_used": alpha,
        "one_sided": one_sided,
        "verdict": _verdict(sig, ev_rate, ba_rate, want_higher=True),
    }


def pick_best(candidates: list, hypothesis: str) -> dict:
    """Pick window with lowest p-value among those with positive excess; fallback to lowest p."""
    valid = [c for c in candidates if c["p_value"] is not None]
    if not valid:
        return candidates[0] if candidates else {}
    positive = [c for c in valid if (c.get("excess_hit_rate") or 0) > 0]
    pool = positive if positive else valid
    return min(pool, key=lambda c: c["p_value"])


def run_primary_event_test(vix: pd.Series, event_type: str, event_dates: list, all_event_days: set) -> dict:
    cfg = EVENT_WINDOW_CONFIG[event_type]
    pre_start, pre_end = PRIMARY_PRE_WINDOW
    post_off = PRIMARY_POST_WINDOW

    ev_pre = event_pre_returns(vix, event_dates, pre_start, pre_end)
    ba_pre = baseline_pre_returns(vix, all_event_days, pre_start, pre_end, stride=BASELINE_STRIDE)
    h1 = test_window(ev_pre, ba_pre, "up", one_sided=True, alpha=BONFERRONI_ALPHA)
    h1["window"] = window_label_pre(pre_start, pre_end)
    h1["start_off"] = pre_start
    h1["end_off"] = pre_end

    ev_post = event_post_returns(vix, event_dates, post_off)
    ba_post = baseline_post_returns(vix, all_event_days, post_off, stride=BASELINE_STRIDE)
    h2 = test_window(ev_post, ba_post, "down", one_sided=True, alpha=BONFERRONI_ALPHA)
    h2["window"] = window_label_post(post_off)
    h2["post_off"] = post_off

    return {
        "event": cfg["label"],
        "event_type": event_type,
        "pre_window": h1["window"],
        "post_window": h2["window"],
        "h1_pre_rise": h1,
        "h2_post_fall": h2,
    }


def build_conclusion_summary(primary_rows: list) -> str:
    parts = []
    for row in primary_rows:
        h1 = row["h1_pre_rise"]
        ev = row["event"]
        win = row["pre_window"]
        if h1.get("significant"):
            parts.append(
                f"{ev} 前 {win} 经 Bonferroni 校正后显著"
                f"（上涨率 {h1['event_hit_rate']*100:.0f}% vs 基准 {h1['baseline_hit_rate']*100:.0f}%，"
                f"p={h1['p_value']:.4f}）"
            )
        elif (h1.get("excess_hit_rate") or 0) > 0:
            parts.append(f"{ev} 前 {win} 有正向差异但不显著（p={h1['p_value']:.3f}）")
        else:
            parts.append(f"{ev} 前 {win} 无显著相关")
    h2_confirmed = [row["event"] for row in primary_rows if row["h2_post_fall"].get("significant")]
    if h2_confirmed:
        parts.append(f"发布后下跌显著：{', '.join(h2_confirmed)}")
    else:
        parts.append("三类事件发布后 VIX 下跌率经 Bonferroni 校正后均未显著高于基准")
    return "；".join(parts) + "。"


def build_primary_analysis(vix: pd.Series, date_map: dict, all_event_days: set) -> dict:
    rows = [run_primary_event_test(vix, et, date_map[et], all_event_days) for et in EVENT_WINDOW_CONFIG]
    pre_start, pre_end = PRIMARY_PRE_WINDOW
    base_pre = baseline_pre_returns(vix, all_event_days, pre_start, pre_end, stride=BASELINE_STRIDE)
    base_post = baseline_post_returns(vix, all_event_days, PRIMARY_POST_WINDOW, stride=BASELINE_STRIDE)
    return {
        "method": (
            f"预注册固定窗口 · 稀疏基准(每{BASELINE_STRIDE}交易日取anchor) · "
            f"单侧z检验 · Bonferroni α={BONFERRONI_ALPHA} ({N_PRIMARY_TESTS} tests)"
        ),
        "windows": {
            "h1_pre": window_label_pre(*PRIMARY_PRE_WINDOW),
            "h2_post": window_label_post(PRIMARY_POST_WINDOW),
        },
        "baseline_stride": BASELINE_STRIDE,
        "bonferroni_tests": N_PRIMARY_TESTS,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "baseline_reference": {
            "pre": {
                "hit_rate_up": round(sum(1 for r in base_pre if r > 0) / len(base_pre), 3) if base_pre else None,
                "mean_pct": round(float(np.mean(base_pre)), 3) if base_pre else None,
                "n": len(base_pre),
                "description": f"稀疏基准：非事件日 {window_label_pre(*PRIMARY_PRE_WINDOW)}",
            },
            "post": {
                "hit_rate_down": round(sum(1 for r in base_post if r < 0) / len(base_post), 3) if base_post else None,
                "mean_pct": round(float(np.mean(base_post)), 3) if base_post else None,
                "n": len(base_post),
                "description": f"稀疏基准：非事件日 {window_label_post(PRIMARY_POST_WINDOW)}",
            },
        },
        "by_event": rows,
        "conclusion_summary": build_conclusion_summary(rows),
    }


def build_subsample_analysis(vix: pd.Series, date_map: dict, all_event_days: set) -> dict:
    """Re-run primary + level analysis on 1y and 2y trailing sub-samples."""
    today = vix.index[-1]
    results = {}
    for label, months in [("1y", 12), ("2y", 24)]:
        cutoff = today - pd.DateOffset(months=months)
        sub_date_map = {
            et: [d for d in dates if pd.Timestamp(d) >= cutoff]
            for et, dates in date_map.items()
        }
        sub_event_days = (
            collect_event_trading_days(vix, sub_date_map["fomc"])
            | collect_event_trading_days(vix, sub_date_map["cpi"])
            | collect_event_trading_days(vix, sub_date_map["nfp"])
        )
        sub_vix = vix[vix.index >= cutoff]

        rows = []
        pre_start, pre_end = PRIMARY_PRE_WINDOW
        ba_ret = baseline_pre_returns(sub_vix, sub_event_days, pre_start, pre_end, stride=1)
        ba_lvl = baseline_vix_window_avg(sub_vix, sub_event_days, pre_start, pre_end, stride=1)

        for et, ev_label in [("fomc", "FOMC"), ("cpi", "CPI"), ("nfp", "NFP")]:
            dates = sub_date_map[et]

            # return-based (original method)
            ev_ret = event_pre_returns(sub_vix, dates, pre_start, pre_end)
            ret_result = test_window(ev_ret, ba_ret, "up", one_sided=True, alpha=0.05)

            # level-based Method A
            ev_lvl = event_vix_window_avg(sub_vix, dates, pre_start, pre_end)
            lvl_result = welch_ttest_greater(ev_lvl, ba_lvl, alpha=0.05)

            rows.append({
                "event": ev_label,
                "n_events": len(dates),
                "return_based": {
                    "event_hit_rate": ret_result.get("event_hit_rate"),
                    "baseline_hit_rate": ret_result.get("baseline_hit_rate"),
                    "excess_hit_rate": ret_result.get("excess_hit_rate"),
                    "p_value": ret_result.get("p_value"),
                    "verdict": ret_result.get("verdict"),
                },
                "level_based": {
                    "event_mean_vix": lvl_result.get("event_mean"),
                    "baseline_mean_vix": lvl_result.get("baseline_mean"),
                    "excess_vix": lvl_result.get("excess"),
                    "p_value": lvl_result.get("p_value"),
                    "verdict": lvl_result.get("verdict"),
                },
            })

        results[label] = {
            "label": f"近 {label}",
            "date_range": f"{cutoff.strftime('%Y-%m-%d')} ~ {today.strftime('%Y-%m-%d')}",
            "note": "小样本：n<15 时 p 值仅供参考",
            "by_event": rows,
        }
    return results


def build_covid_sensitivity(vix: pd.Series, date_map: dict, all_event_days: set) -> dict:
    fomc_dates = [
        d for d in date_map["fomc"]
        if not (COVID_EXCLUDE_START <= pd.Timestamp(d) <= COVID_EXCLUDE_END)
    ]
    row = run_primary_event_test(vix, "fomc", fomc_dates, all_event_days)
    h1 = row["h1_pre_rise"]
    return {
        "description": f"排除 {COVID_EXCLUDE_START.date()} ~ {COVID_EXCLUDE_END.date()} 的 FOMC 事件",
        "events_remaining": len(fomc_dates),
        "pre_window": row["pre_window"],
        "hit_rate_up": h1.get("event_hit_rate"),
        "baseline_hit_rate_up": h1.get("baseline_hit_rate"),
        "excess_hit_rate": h1.get("excess_hit_rate"),
        "p_value": h1.get("p_value"),
        "significant": h1.get("significant", False),
        "verdict": h1.get("verdict"),
    }


def sweep_event_windows(vix: pd.Series, event_type: str, event_dates: list, all_event_days: set) -> dict:
    cfg = EVENT_WINDOW_CONFIG[event_type]
    pre_candidates, post_candidates = [], []

    for start_off, end_off in cfg["pre"]:
        ev = event_pre_returns(vix, event_dates, start_off, end_off)
        ba = baseline_pre_returns(vix, all_event_days, start_off, end_off)
        result = test_window(ev, ba, "up")
        result["window"] = window_label_pre(start_off, end_off)
        result["start_off"] = start_off
        result["end_off"] = end_off
        pre_candidates.append(result)

    for post_off in cfg["post"]:
        ev = event_post_returns(vix, event_dates, post_off)
        ba = baseline_post_returns(vix, all_event_days, post_off)
        result = test_window(ev, ba, "down")
        result["window"] = window_label_post(post_off)
        result["post_off"] = post_off
        post_candidates.append(result)

    best_pre = pick_best(pre_candidates, "h1")
    best_post = pick_best(post_candidates, "h2")

    return {
        "event": cfg["label"],
        "event_type": event_type,
        "pre_candidates": pre_candidates,
        "post_candidates": post_candidates,
        "best_pre": best_pre,
        "best_post": best_post,
    }


def build_event_study_detail(vix: pd.Series, event_dates: list, event_type: str, primary_row: dict) -> dict:
    """Per-event table using pre-registered primary windows."""
    h1, h2 = primary_row["h1_pre_rise"], primary_row["h2_post_fall"]
    pre_start = h1.get("start_off", PRIMARY_PRE_WINDOW[0])
    pre_end = h1.get("end_off", PRIMARY_PRE_WINDOW[1])
    post_off = h2.get("post_off", PRIMARY_POST_WINDOW)

    idx = vix.index
    events = []
    for ed in event_dates:
        t0 = event_trading_day(idx, pd.Timestamp(ed))
        if t0 is None or t0 < idx[0] or t0 > idx[-1]:
            continue
        pre_r = pct_return(vix, trading_offset(idx, t0, -pre_start), trading_offset(idx, t0, -pre_end))
        row = {
            "date": pd.Timestamp(ed).strftime("%Y-%m-%d"),
            "trading_day": t0.strftime("%Y-%m-%d"),
            "vix_at_event": round(float(vix.loc[t0]), 2),
            "pre_return_pct": pre_r,
            "pre_window": primary_row["pre_window"],
        }
        for po in EVENT_WINDOW_CONFIG[event_type]["post"]:
            row[f"post_{po}d_return_pct"] = pct_return(vix, t0, trading_offset(idx, t0, po))
        row["best_post_window"] = primary_row["post_window"]
        events.append(row)

    pre_vals = [e["pre_return_pct"] for e in events if e["pre_return_pct"] is not None]
    post_key = f"post_{post_off}d_return_pct"
    post_vals = [e[post_key] for e in events if e.get(post_key) is not None]

    def summarize(vals, direction):
        if not vals:
            return {"mean": None, "hit_rate": None, "count": 0}
        arr = np.array(vals)
        hit = (arr > 0).mean() if direction == "up" else (arr < 0).mean()
        return {"mean": round(float(arr.mean()), 3), "hit_rate": round(float(hit), 3), "count": len(vals)}

    return {
        "event_type": event_type,
        "best_pre_window": primary_row["pre_window"],
        "best_post_window": primary_row["post_window"],
        "events": events[-12:],
        "summary": {
            "count": len(events),
            "pre": summarize(pre_vals, "up"),
            "post": summarize(post_vals, "down"),
        },
    }


def build_hypothesis_verdict(primary: dict) -> dict:
    by_event = []
    for row in primary["by_event"]:
        h1, h2 = row["h1_pre_rise"], row["h2_post_fall"]
        by_event.append({
            "event": row["event"],
            "best_pre_window": row["pre_window"],
            "best_post_window": row["post_window"],
            "h1_pre_rise": {
                "confirmed": h1.get("significant", False),
                "mean_pct": h1.get("event_mean_pct"),
                "hit_rate_up": h1.get("event_hit_rate"),
                "baseline_hit_rate_up": h1.get("baseline_hit_rate"),
                "excess_hit_rate": h1.get("excess_hit_rate"),
                "p_value": h1.get("p_value"),
                "verdict": h1.get("verdict"),
            },
            "h2_post_fall": {
                "confirmed": h2.get("significant", False),
                "mean_pct": h2.get("event_mean_pct"),
                "hit_rate_down": h2.get("event_hit_rate"),
                "baseline_hit_rate_down": h2.get("baseline_hit_rate"),
                "excess_hit_rate": h2.get("excess_hit_rate"),
                "p_value": h2.get("p_value"),
                "verdict": h2.get("verdict"),
            },
        })
    win = primary["windows"]
    return {
        "h1": f"发布前 VIX 上涨率是否显著高于非事件日（固定窗口 {win['h1_pre']}，Bonferroni 校正）",
        "h2": f"发布后 VIX 下跌率是否显著高于非事件日（固定窗口 {win['h2_post']}，Bonferroni 校正）",
        "by_event": by_event,
    }


def build_correlation_analysis(primary: dict) -> dict:
    ref = primary["baseline_reference"]
    by_event = []
    for row in primary["by_event"]:
        h1, h2 = row["h1_pre_rise"], row["h2_post_fall"]
        by_event.append({
            "event": row["event"],
            "best_pre_window": row["pre_window"],
            "best_post_window": row["post_window"],
            "h1_pre_rise": {
                "event_hit_rate_up": h1.get("event_hit_rate"),
                "baseline_hit_rate_up": h1.get("baseline_hit_rate"),
                "excess_hit_rate": h1.get("excess_hit_rate"),
                "event_mean_pct": h1.get("event_mean_pct"),
                "baseline_mean_pct": h1.get("baseline_mean_pct"),
                "event_n": h1.get("event_n"),
                "baseline_n": h1.get("baseline_n"),
                "z_stat": h1.get("z_stat"),
                "p_value": h1.get("p_value"),
                "phi": h1.get("phi"),
                "significant": h1.get("significant", False),
                "verdict": h1.get("verdict"),
            },
            "h2_post_fall": {
                "event_hit_rate_down": h2.get("event_hit_rate"),
                "baseline_hit_rate_down": h2.get("baseline_hit_rate"),
                "excess_hit_rate": h2.get("excess_hit_rate"),
                "event_mean_pct": h2.get("event_mean_pct"),
                "baseline_mean_pct": h2.get("baseline_mean_pct"),
                "event_n": h2.get("event_n"),
                "baseline_n": h2.get("baseline_n"),
                "z_stat": h2.get("z_stat"),
                "p_value": h2.get("p_value"),
                "phi": h2.get("phi"),
                "significant": h2.get("significant", False),
                "verdict": h2.get("verdict"),
            },
        })
    return {
        "method": primary["method"],
        "baseline_reference": {
            "pre_7d": {
                "hit_rate_up": ref["pre"]["hit_rate_up"],
                "mean_pct": ref["pre"]["mean_pct"],
                "n": ref["pre"]["n"],
                "description": ref["pre"]["description"],
            },
            "post_1d": {
                "hit_rate_down": ref["post"]["hit_rate_down"],
                "mean_pct": ref["post"]["mean_pct"],
                "n": ref["post"]["n"],
                "description": ref["post"]["description"],
            },
        },
        "by_event": by_event,
    }


def get_upcoming_events() -> list:
    today = pd.Timestamp.today().normalize()
    events = get_event_dates()
    nfp = get_nfp_dates()
    upcoming = []
    for name, dates in [("FOMC", events["fomc"]), ("CPI", events["cpi"]), ("NFP", nfp)]:
        for d in dates:
            if d >= today:
                upcoming.append({"event": name, "date": d.strftime("%Y-%m-%d")})
    upcoming.sort(key=lambda x: x["date"])
    return upcoming[:6]


def build_output() -> dict:
    print("Fetching VIX...")
    vix = fetch_vix_series()
    print(f"  VIX: {len(vix)} days, {vix.index[0].date()} ~ {vix.index[-1].date()}")

    event_dates = get_event_dates()
    nfp_dates = get_nfp_dates()
    date_map = {
        "fomc": filter_event_dates_in_range(vix, event_dates["fomc"]),
        "cpi": filter_event_dates_in_range(vix, event_dates["cpi"]),
        "nfp": filter_event_dates_in_range(vix, nfp_dates),
    }
    print(
        "  Events in range:",
        {k: len(v) for k, v in date_map.items()},
    )

    all_event_days = (
        collect_event_trading_days(vix, date_map["fomc"])
        | collect_event_trading_days(vix, date_map["cpi"])
        | collect_event_trading_days(vix, date_map["nfp"])
    )

    sweeps = [sweep_event_windows(vix, et, date_map[et], all_event_days) for et in EVENT_WINDOW_CONFIG]
    primary = build_primary_analysis(vix, date_map, all_event_days)
    studies = [
        build_event_study_detail(vix, date_map[et], et, primary["by_event"][i])
        for i, et in enumerate(EVENT_WINDOW_CONFIG)
    ]

    corr_analysis = build_correlation_analysis(primary)
    verdict = build_hypothesis_verdict(primary)
    covid_sensitivity = build_covid_sensitivity(vix, date_map, all_event_days)
    level_analysis = build_level_analysis(vix, date_map, all_event_days)
    subsample = build_subsample_analysis(vix, date_map, all_event_days)
    print("  Subsample event counts:")
    for period, r in subsample.items():
        counts = {row["event"]: row["n_events"] for row in r["by_event"]}
        print(f"    {period}: {counts}")

    n_exploratory = sum(
        len(EVENT_WINDOW_CONFIG[et]["pre"]) + len(EVENT_WINDOW_CONFIG[et]["post"])
        for et in EVENT_WINDOW_CONFIG
    )

    # Feature correlations (unchanged)
    df = pd.DataFrame({"vix": vix, "daily_return": vix.pct_change(), "return_7d": vix.pct_change(7)})
    merged = df.join(add_nfp_days_to(build_event_features(df), df.index), how="inner")

    timeline = [{
        "date": d.strftime("%Y-%m-%d"),
        "vix": round(float(row["vix"]), 2),
        "daily_return_pct": round(float(row["daily_return"]) * 100, 3) if pd.notna(row["daily_return"]) else None,
        "event_flags": [],
    } for d, row in merged.iterrows()]

    window_sweep_summary = [{
        "event": s["event"],
        "pre_windows_tested": [c["window"] for c in s["pre_candidates"]],
        "post_windows_tested": [c["window"] for c in s["post_candidates"]],
        "best_pre": s["best_pre"],
        "best_post": s["best_post"],
        "all_pre": s["pre_candidates"],
        "all_post": s["post_candidates"],
    } for s in sweeps]

    return {
        "title": "VIX × Event Calendar Analysis",
        "subtitle": "FOMC / CPI / NFP · 预注册固定窗口 + 探索性窗口扫描",
        "primary_analysis": primary,
        "conclusion_summary": primary["conclusion_summary"],
        "hypothesis": verdict,
        "correlation_analysis": corr_analysis,
        "window_sweep": window_sweep_summary,
        "sensitivity": {"covid_fomc": covid_sensitivity},
        "level_analysis": level_analysis,
        "subsample_analysis": subsample,
        "summary": {
            "data_range": f"{merged.index[0].strftime('%Y-%m-%d')} ~ {merged.index[-1].strftime('%Y-%m-%d')}",
            "analysis_years": ANALYSIS_YEARS,
            "instrument": "^VIX (CBOE Volatility Index)",
            "total_trading_days": len(merged),
            "event_counts": {k: len(v) for k, v in date_map.items()},
            "method": "Pre-registered T-5~T-1 / T+1 with sparse baseline + Bonferroni",
        },
        "event_studies": studies,
        "upcoming_events": get_upcoming_events(),
        "vix_timeline": timeline,
        "methodology": {
            "instrument": "Yahoo Finance ^VIX",
            "primary_windows": {
                "h1_pre": window_label_pre(*PRIMARY_PRE_WINDOW),
                "h2_post": window_label_post(PRIMARY_POST_WINDOW),
            },
            "baseline_stride": BASELINE_STRIDE,
            "bonferroni_primary": {"tests": N_PRIMARY_TESTS, "alpha": BONFERRONI_ALPHA},
            "exploratory_sweep": {
                "window_config": EVENT_WINDOW_CONFIG,
                "tests": n_exploratory,
                "bonferroni_alpha": round(0.05 / n_exploratory, 5),
                "warning": "探索性扫描存在多重检验风险，不作为主结论依据",
            },
            "baseline": "Sparse non-event anchors; one-sided two-proportion z-test",
        },
    }


def main():
    output = build_output()
    out_path = DATA_DIR / "event_vix_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")
    sync_public_data()
    print("Done.")


if __name__ == "__main__":
    main()

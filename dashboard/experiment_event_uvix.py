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


def get_nfp_dates(start_year=2022, end_year=2027):
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
    df = fetch_yfinance_history("^VIX", period="5y")
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


def proportion_test(success_a: int, n_a: int, success_b: int, n_b: int) -> dict:
    if n_a < 5 or n_b < 5:
        return {"z_stat": None, "p_value": None}
    p1, p2 = success_a / n_a, success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)
    if p_pool in (0, 1):
        return {"z_stat": None, "p_value": None}
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p1 - p2) / se
    return {"z_stat": round(float(z), 3), "p_value": round(float(2 * (1 - stats.norm.cdf(abs(z)))), 4)}


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


def collect_event_trading_days(vix: pd.Series, event_dates: list) -> set:
    return {t for d in event_dates if (t := event_trading_day(vix.index, pd.Timestamp(d))) is not None}


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


def baseline_pre_returns(vix: pd.Series, exclude: set, start_off: int, end_off: int = 1) -> list:
    idx = vix.index
    out = []
    for t0 in idx:
        if t0 in exclude:
            continue
        r = pct_return(vix, trading_offset(idx, t0, -start_off), trading_offset(idx, t0, -end_off))
        if r is not None:
            out.append(r)
    return out


def baseline_post_returns(vix: pd.Series, exclude: set, post_off: int) -> list:
    idx = vix.index
    out = []
    for t0 in idx:
        if t0 in exclude:
            continue
        r = pct_return(vix, t0, trading_offset(idx, t0, post_off))
        if r is not None:
            out.append(r)
    return out


def test_window(event_vals: list, base_vals: list, direction: str = "up") -> dict:
    if direction == "up":
        ev_succ = sum(1 for r in event_vals if r > 0)
        ba_succ = sum(1 for r in base_vals if r > 0)
    else:
        ev_succ = sum(1 for r in event_vals if r < 0)
        ba_succ = sum(1 for r in base_vals if r < 0)
    ev_n, ba_n = len(event_vals), len(base_vals)
    ev_rate = round(ev_succ / ev_n, 3) if ev_n else None
    ba_rate = round(ba_succ / ba_n, 3) if ba_n else None
    test = proportion_test(ev_succ, ev_n, ba_succ, ba_n) if ev_n and ba_n else {"z_stat": None, "p_value": None}
    excess = round(ev_rate - ba_rate, 3) if ev_rate is not None and ba_rate is not None else None
    sig = test["p_value"] is not None and test["p_value"] < 0.05
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
        "significant": sig and excess is not None and excess > 0,
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


def build_event_study_detail(vix: pd.Series, event_dates: list, event_type: str, sweep: dict) -> dict:
    """Per-event table using best windows."""
    best_pre = sweep["best_pre"]
    best_post = sweep["best_post"]
    pre_start = best_pre.get("start_off", 7)
    pre_end = best_pre.get("end_off", 1)
    post_off = best_post.get("post_off", 1)

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
            "pre_window": best_pre.get("window", ""),
        }
        for po in EVENT_WINDOW_CONFIG[event_type]["post"]:
            row[f"post_{po}d_return_pct"] = pct_return(vix, t0, trading_offset(idx, t0, po))
        row["best_post_window"] = best_post.get("window", "")
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
        "best_pre_window": best_pre.get("window"),
        "best_post_window": best_post.get("window"),
        "events": events[-12:],
        "summary": {
            "count": len(events),
            "pre": summarize(pre_vals, "up"),
            "post": summarize(post_vals, "down"),
        },
    }


def build_hypothesis_verdict(sweeps: list) -> dict:
    by_event = []
    for s in sweeps:
        bp, bpo = s["best_pre"], s["best_post"]
        by_event.append({
            "event": s["event"],
            "best_pre_window": bp.get("window"),
            "best_post_window": bpo.get("window"),
            "h1_pre_rise": {
                "confirmed": bp.get("significant", False),
                "mean_pct": bp.get("event_mean_pct"),
                "hit_rate_up": bp.get("event_hit_rate"),
                "baseline_hit_rate_up": bp.get("baseline_hit_rate"),
                "excess_hit_rate": bp.get("excess_hit_rate"),
                "p_value": bp.get("p_value"),
                "verdict": bp.get("verdict"),
            },
            "h2_post_fall": {
                "confirmed": bpo.get("significant", False),
                "mean_pct": bpo.get("event_mean_pct"),
                "hit_rate_down": bpo.get("event_hit_rate"),
                "baseline_hit_rate_down": bpo.get("baseline_hit_rate"),
                "excess_hit_rate": bpo.get("excess_hit_rate"),
                "p_value": bpo.get("p_value"),
                "verdict": bpo.get("verdict"),
            },
        })
    return {
        "h1": "发布前 VIX 上涨率是否显著高于非事件日（各事件独立选最优窗口）",
        "h2": "发布后 VIX 下跌率是否显著高于非事件日（各事件独立选最优窗口）",
        "by_event": by_event,
    }


def build_correlation_analysis(sweeps: list, vix: pd.Series, all_event_days: set) -> dict:
    # Shared baseline for default T-7~T-1 / T+1 reference
    base_pre = baseline_pre_returns(vix, all_event_days, 7, 1)
    base_post = baseline_post_returns(vix, all_event_days, 1)
    by_event = []
    for s in sweeps:
        bp, bpo = s["best_pre"], s["best_post"]
        by_event.append({
            "event": s["event"],
            "best_pre_window": bp.get("window"),
            "best_post_window": bpo.get("window"),
            "h1_pre_rise": {
                "event_hit_rate_up": bp.get("event_hit_rate"),
                "baseline_hit_rate_up": bp.get("baseline_hit_rate"),
                "excess_hit_rate": bp.get("excess_hit_rate"),
                "event_mean_pct": bp.get("event_mean_pct"),
                "baseline_mean_pct": bp.get("baseline_mean_pct"),
                "event_n": bp.get("event_n"),
                "baseline_n": bp.get("baseline_n"),
                "z_stat": bp.get("z_stat"),
                "p_value": bp.get("p_value"),
                "phi": bp.get("phi"),
                "significant": bp.get("significant", False),
                "verdict": bp.get("verdict"),
            },
            "h2_post_fall": {
                "event_hit_rate_down": bpo.get("event_hit_rate"),
                "baseline_hit_rate_down": bpo.get("baseline_hit_rate"),
                "excess_hit_rate": bpo.get("excess_hit_rate"),
                "event_mean_pct": bpo.get("event_mean_pct"),
                "baseline_mean_pct": bpo.get("baseline_mean_pct"),
                "event_n": bpo.get("event_n"),
                "baseline_n": bpo.get("baseline_n"),
                "z_stat": bpo.get("z_stat"),
                "p_value": bpo.get("p_value"),
                "phi": bpo.get("phi"),
                "significant": bpo.get("significant", False),
                "verdict": bpo.get("verdict"),
            },
        })
    return {
        "method": "各事件独立扫描多个时间窗口 · 命中率 vs 同窗口非事件日基准 · 双比例 z 检验",
        "baseline_reference": {
            "pre_7d": {
                "hit_rate_up": round(sum(1 for r in base_pre if r > 0) / len(base_pre), 3),
                "mean_pct": round(float(np.mean(base_pre)), 3),
                "n": len(base_pre),
                "description": "参考基准：非事件日 T-7~T-1",
            },
            "post_1d": {
                "hit_rate_down": round(sum(1 for r in base_post if r < 0) / len(base_post), 3),
                "mean_pct": round(float(np.mean(base_post)), 3),
                "n": len(base_post),
                "description": "参考基准：非事件日 T+1",
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
    all_event_days = (
        collect_event_trading_days(vix, event_dates["fomc"])
        | collect_event_trading_days(vix, event_dates["cpi"])
        | collect_event_trading_days(vix, nfp_dates)
    )

    date_map = {"fomc": event_dates["fomc"], "cpi": event_dates["cpi"], "nfp": nfp_dates}
    sweeps = [sweep_event_windows(vix, et, date_map[et], all_event_days) for et in EVENT_WINDOW_CONFIG]
    studies = [build_event_study_detail(vix, date_map[et], et, sw) for et, sw in zip(EVENT_WINDOW_CONFIG, sweeps)]

    corr_analysis = build_correlation_analysis(sweeps, vix, all_event_days)
    verdict = build_hypothesis_verdict(sweeps)

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
        "subtitle": "FOMC / CPI / NFP · 各事件独立扫描时间窗口 · 命中率 vs 基准",
        "hypothesis": verdict,
        "correlation_analysis": corr_analysis,
        "window_sweep": window_sweep_summary,
        "summary": {
            "data_range": f"{merged.index[0].strftime('%Y-%m-%d')} ~ {merged.index[-1].strftime('%Y-%m-%d')}",
            "instrument": "^VIX (CBOE Volatility Index)",
            "total_trading_days": len(merged),
            "method": "Per-event window sweep with matched baseline",
        },
        "event_studies": studies,
        "upcoming_events": get_upcoming_events(),
        "vix_timeline": timeline,
        "methodology": {
            "instrument": "Yahoo Finance ^VIX",
            "window_config": EVENT_WINDOW_CONFIG,
            "baseline": "Same window on non-event trading days; two-proportion z-test",
            "best_window_rule": "Lowest p-value among candidates with positive excess hit rate",
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

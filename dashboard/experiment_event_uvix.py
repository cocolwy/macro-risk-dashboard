"""
Event Calendar × UVIX Analysis

Reuses the 9 event calendar features from predict_model.build_event_features():
  FOMC: days_to, days_since, within_3d, within_7d  (4)
  CPI:  days_to, days_since, within_3d, within_7d  (4)
  NFP:  within_3d                                   (1)

Analyzes correlation between these features and UVIX price moves,
plus pre-release event-study returns around FOMC / CPI / NFP dates.
"""

import json
from pathlib import Path

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

FEATURE_META = {
    "fomc_days_to": {"event": "FOMC", "label": "距下次 FOMC 天数"},
    "fomc_days_since": {"event": "FOMC", "label": "距上次 FOMC 天数"},
    "fomc_within_3d": {"event": "FOMC", "label": "FOMC 前 3 天内"},
    "fomc_within_7d": {"event": "FOMC", "label": "FOMC 前 7 天内"},
    "cpi_days_to": {"event": "CPI", "label": "距下次 CPI 天数"},
    "cpi_days_since": {"event": "CPI", "label": "距上次 CPI 天数"},
    "cpi_within_3d": {"event": "CPI", "label": "CPI 前 3 天内"},
    "cpi_within_7d": {"event": "CPI", "label": "CPI 前 7 天内"},
    "nfp_within_3d": {"event": "NFP", "label": "NFP 前 3 天内"},
}

WINDOW_FEATURES = [
    ("fomc_within_3d", "FOMC 前 3 天"),
    ("fomc_within_7d", "FOMC 前 7 天"),
    ("cpi_within_3d", "CPI 前 3 天"),
    ("cpi_within_7d", "CPI 前 7 天"),
    ("nfp_within_3d", "NFP 前 3 天"),
]


def get_nfp_dates(start_year=2022, end_year=2027):
    dates = set()
    for year in range(start_year, end_year):
        for month in range(1, 13):
            first = pd.Timestamp(year, month, 1)
            offset = (4 - first.weekday()) % 7
            dates.add(first + pd.Timedelta(days=offset))
    return sorted(dates)


def fetch_uvix_series() -> pd.Series:
    df = fetch_yfinance_history("UVIX", period="max")
    if df.empty:
        raise RuntimeError("UVIX data unavailable from Yahoo Finance")
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.name = "uvix"
    return s.dropna()


def nearest_trading_day(idx: pd.DatetimeIndex, target: pd.Timestamp, direction="before"):
    """Find nearest trading day on or before/after target."""
    if direction == "before":
        candidates = idx[idx <= target]
        return candidates[-1] if len(candidates) else None
    candidates = idx[idx >= target]
    return candidates[0] if len(candidates) else None


def compute_returns(uvix: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"uvix": uvix})
    df["daily_return"] = df["uvix"].pct_change()
    df["return_5d"] = df["uvix"].pct_change(5)
    return df.dropna(subset=["uvix"])


def spearman_corr(x: pd.Series, y: pd.Series) -> dict:
    mask = x.notna() & y.notna()
    if mask.sum() < 30:
        return {"rho": None, "p_value": None, "n": int(mask.sum())}
    rho, p = stats.spearmanr(x[mask], y[mask])
    return {"rho": round(float(rho), 4), "p_value": round(float(p), 4), "n": int(mask.sum())}


def analyze_feature_correlations(merged: pd.DataFrame) -> list:
    results = []
    for feat in EVENT_FEATURES:
        meta = FEATURE_META[feat]
        daily = spearman_corr(merged[feat], merged["daily_return"])
        ret5d = spearman_corr(merged[feat], merged["return_5d"])
        results.append({
            "feature": feat,
            "event": meta["event"],
            "label": meta["label"],
            "daily_return": daily,
            "return_5d": ret5d,
        })
    return results


def analyze_window_comparison(merged: pd.DataFrame) -> list:
    baseline_daily = merged["daily_return"].mean() * 100
    baseline_5d = merged["return_5d"].mean() * 100
    results = []

    for feat, label in WINDOW_FEATURES:
        mask = merged[feat] == 1.0
        subset = merged[mask]
        if len(subset) < 5:
            continue
        mean_daily = subset["daily_return"].mean() * 100
        mean_5d = subset["return_5d"].mean() * 100
        hit_rate = (subset["daily_return"] > 0).mean()
        results.append({
            "feature": feat,
            "label": label,
            "days_count": len(subset),
            "mean_daily_return_pct": round(mean_daily, 3),
            "mean_5d_return_pct": round(mean_5d, 3),
            "baseline_daily_return_pct": round(baseline_daily, 3),
            "baseline_5d_return_pct": round(baseline_5d, 3),
            "excess_daily_return_pct": round(mean_daily - baseline_daily, 3),
            "excess_5d_return_pct": round(mean_5d - baseline_5d, 3),
            "hit_rate_up": round(hit_rate, 3),
        })
    return results


def event_study(uvix: pd.Series, event_dates: list, event_type: str) -> dict:
    idx = uvix.index
    events = []

    for event_date in event_dates:
        ed = pd.Timestamp(event_date)
        if ed < idx[0] or ed > idx[-1]:
            continue

        t0 = nearest_trading_day(idx, ed, "before")
        if t0 is None:
            continue

        def window_return(days_back_start, days_back_end):
            start = nearest_trading_day(idx, ed - pd.Timedelta(days=days_back_start), "before")
            end = nearest_trading_day(idx, ed - pd.Timedelta(days=days_back_end), "before")
            if start is None or end is None or start >= end:
                return None
            p0, p1 = uvix.loc[start], uvix.loc[end]
            if p0 <= 0:
                return None
            return round((p1 / p0 - 1) * 100, 3)

        pre_5d = window_return(7, 1)
        pre_3d = window_return(4, 1)
        pre_1d = window_return(2, 1)

        post_start = nearest_trading_day(idx, ed, "after")
        post_end = nearest_trading_day(idx, ed + pd.Timedelta(days=2), "after")
        post_1d = None
        if post_start and post_end and post_start < post_end:
            p0, p1 = uvix.loc[post_start], uvix.loc[post_end]
            if p0 > 0:
                post_1d = round((p1 / p0 - 1) * 100, 3)

        events.append({
            "date": ed.strftime("%Y-%m-%d"),
            "uvix_at_event": round(float(uvix.loc[t0]), 2),
            "pre_5d_return_pct": pre_5d,
            "pre_3d_return_pct": pre_3d,
            "pre_1d_return_pct": pre_1d,
            "post_1d_return_pct": post_1d,
        })

    pre5_vals = [e["pre_5d_return_pct"] for e in events if e["pre_5d_return_pct"] is not None]
    pre3_vals = [e["pre_3d_return_pct"] for e in events if e["pre_3d_return_pct"] is not None]
    pre1_vals = [e["pre_1d_return_pct"] for e in events if e["pre_1d_return_pct"] is not None]

    def summarize(vals):
        if not vals:
            return {"mean": None, "median": None, "hit_rate": None}
        arr = np.array(vals)
        return {
            "mean": round(float(arr.mean()), 3),
            "median": round(float(np.median(arr)), 3),
            "hit_rate": round(float((arr > 0).mean()), 3),
        }

    return {
        "event_type": event_type,
        "events": events[-12:],  # recent 12 for dashboard table
        "summary": {
            "count": len(events),
            "pre_5d": summarize(pre5_vals),
            "pre_3d": summarize(pre3_vals),
            "pre_1d": summarize(pre1_vals),
        },
    }


def get_upcoming_events(uvix_start: pd.Timestamp) -> list:
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
    print("Fetching UVIX...")
    uvix = fetch_uvix_series()
    print(f"  UVIX: {len(uvix)} days, {uvix.index[0].date()} ~ {uvix.index[-1].date()}")

    returns = compute_returns(uvix)
    event_feats = build_event_features(returns)
    merged = returns.join(event_feats, how="inner")

    print(f"  Merged dataset: {len(merged)} trading days")

    correlations = analyze_feature_correlations(merged)
    window_comparison = analyze_window_comparison(merged)

    event_dates = get_event_dates()
    studies = [
        event_study(uvix, event_dates["fomc"], "fomc"),
        event_study(uvix, event_dates["cpi"], "cpi"),
        event_study(uvix, get_nfp_dates(), "nfp"),
    ]

    # UVIX timeline with event flags for chart overlay
    timeline = []
    for date, row in merged.iterrows():
        flags = []
        if row["fomc_within_3d"] == 1:
            flags.append("FOMC")
        if row["cpi_within_3d"] == 1:
            flags.append("CPI")
        if row["nfp_within_3d"] == 1:
            flags.append("NFP")
        timeline.append({
            "date": date.strftime("%Y-%m-%d"),
            "uvix": round(float(row["uvix"]), 2),
            "daily_return_pct": round(float(row["daily_return"]) * 100, 3) if pd.notna(row["daily_return"]) else None,
            "event_flags": flags,
        })

    strongest = sorted(
        [c for c in correlations if c["daily_return"]["rho"] is not None],
        key=lambda x: abs(x["daily_return"]["rho"]),
        reverse=True,
    )[:3]

    return {
        "title": "UVIX × Event Calendar Analysis",
        "subtitle": "FOMC / CPI / NFP 发布前 UVIX 上涨相关性 · 复用 +Events 9 特征",
        "summary": {
            "data_range": f"{merged.index[0].strftime('%Y-%m-%d')} ~ {merged.index[-1].strftime('%Y-%m-%d')}",
            "uvix_inception": uvix.index[0].strftime("%Y-%m-%d"),
            "total_trading_days": len(merged),
            "event_features": EVENT_FEATURES,
            "strongest_correlations": [
                {
                    "feature": c["feature"],
                    "label": c["label"],
                    "rho": c["daily_return"]["rho"],
                    "p_value": c["daily_return"]["p_value"],
                }
                for c in strongest
            ],
        },
        "feature_correlations": correlations,
        "window_comparison": window_comparison,
        "event_studies": studies,
        "upcoming_events": get_upcoming_events(uvix.index[0]),
        "uvix_timeline": timeline,
        "methodology": {
            "event_features_source": "predict_model.build_event_features() — same 9 features as LR Slim+Events",
            "uvix_source": "Yahoo Finance UVIX (1.5x Long VIX Futures ETF)",
            "pre_window_definition": "T-Nd return = UVIX close at T-1 divided by close at T-N trading days before event",
            "correlation_method": "Spearman rank correlation",
        },
    }


def main():
    output = build_output()
    out_path = DATA_DIR / "event_uvix_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")
    sync_public_data()
    print("Done.")


if __name__ == "__main__":
    main()

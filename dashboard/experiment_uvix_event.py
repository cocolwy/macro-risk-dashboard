"""
F001b · UVIX Product Layer Test

Pre-registered protocol (cases/F001b_uvix_preregister.json):
  - Instrument: UVIX
  - H1: T-5 close → T-1 close long return vs sparse non-event baseline
  - H2: T+0 → T+1 post-event fall
  - Bonferroni: 6 primary hit-rate tests (3 events × H1/H2 family aligned with F001)
  - H1 mean return: Welch one-sided t-test (sensitivity alpha = 0.05/3)
  - Costs: broker fee @ $6k + spread proxy from (High-Low)/Close
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
    BONFERRONI_ALPHA,
    EVENT_WINDOW_CONFIG,
    N_PRIMARY_TESTS,
    PRIMARY_POST_WINDOW,
    PRIMARY_PRE_WINDOW,
    baseline_post_returns,
    baseline_pre_returns,
    build_conclusion_summary,
    collect_event_trading_days,
    event_post_returns,
    event_pre_returns,
    event_trading_day,
    filter_event_dates_in_range,
    get_event_dates,
    get_nfp_dates,
    pct_return,
    proportion_test,
    test_window,
    trading_offset,
    window_label_post,
    window_label_pre,
)
from fetch_macro_data import sync_public_data

DATA_DIR = Path(__file__).parent / "data"
TYPICAL_NOTIONAL_USD = 6000
BROKER = {
    "platform_fee_per_share": 0.009,
    "platform_fee_min_usd": 1.88,
    "settlement_fee_per_share": 0.003,
    "settlement_fee_min_usd": 0.01,
}
STRESS_MULTIPLIER = 2.5
H1_MEAN_ALPHA = round(0.05 / 3, 5)  # F001b sensitivity family for mean return
RECENT_YEARS = 2


def _apply_reverse_splits(df: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """Deprecated: prefer Yahoo adjclose in fetch_uvix_history."""
    if not splits:
        return df
    out = df.copy()
    split_rows = sorted(
        (
            pd.Timestamp.utcfromtimestamp(s["date"]).tz_localize(None).normalize(),
            s["numerator"] / s["denominator"],
        )
        for s in splits.values()
    )
    for split_date, ratio in reversed(split_rows):
        mask = out.index < split_date
        for col in ("Open", "High", "Low", "Close"):
            if col in out.columns:
                out.loc[mask, col] = out.loc[mask, col] * ratio
    return out


def parse_uvix_split_dates(splits: dict) -> list[pd.Timestamp]:
    if not splits:
        return []
    return sorted(
        pd.Timestamp.utcfromtimestamp(s["date"]).tz_localize(None).normalize()
        for s in splits.values()
    )


UVIX_SPLIT_DATES: list[pd.Timestamp] = []


def fetch_uvix_history() -> pd.DataFrame:
    """UVIX daily OHLC; Close uses Yahoo adjclose (consistent split-adjusted scale)."""
    global UVIX_SPLIT_DATES
    url = (
        "https://query2.finance.yahoo.com/v8/finance/chart/UVIX"
        "?interval=1d&range=5y&events=split"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    splits = (result.get("events") or {}).get("splits") or {}
    UVIX_SPLIT_DATES = parse_uvix_split_dates(splits)
    rows = []
    for i, t in enumerate(ts):
        close = quote["close"][i]
        if close is None:
            continue
        adj_close = adj[i] if adj and adj[i] is not None else close
        scale = adj_close / close if close else 1.0
        rows.append({
            "Date": pd.Timestamp.utcfromtimestamp(t).tz_localize(None).normalize(),
            "Open": quote["open"][i] * scale if quote["open"][i] is not None else None,
            "High": quote["high"][i] * scale if quote["high"][i] is not None else None,
            "Low": quote["low"][i] * scale if quote["low"][i] is not None else None,
            "Close": adj_close,
            "Volume": quote["volume"][i],
        })
    if not rows:
        raise RuntimeError("UVIX chart API returned no rows")
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df.dropna(subset=["Close"])


def uvix_close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Close"].astype(float)
    s.name = "UVIX"
    return s


def broker_fee_usd(notional_usd: float, price: float) -> float:
    if price <= 0:
        return 0.0
    shares = notional_usd / price
    platform = max(BROKER["platform_fee_min_usd"], BROKER["platform_fee_per_share"] * shares)
    settlement = max(BROKER["settlement_fee_min_usd"], BROKER["settlement_fee_per_share"] * shares)
    return round(platform + settlement, 4)


def broker_round_trip_bps(notional_usd: float, price: float) -> float:
    if notional_usd <= 0:
        return 0.0
    return round(broker_fee_usd(notional_usd, price) * 2 / notional_usd * 10000, 2)


def estimate_spread_bps(df: pd.DataFrame) -> dict:
    """Conservative spread proxy: median daily range / close (not true bid-ask)."""
    hl = (df["High"] - df["Low"]) / df["Close"]
    hl = hl.replace([np.inf, -np.inf], np.nan).dropna()
    if hl.empty:
        return {"median_half_spread_bps": None, "method": "0.5 × median((High-Low)/Close) × 10000"}
    half_spread = float(hl.median()) * 0.5
    return {
        "method": "round_trip ≈ median((High-Low)/Close) × 10000; half as one-way spread proxy",
        "median_round_trip_bps": round(float(hl.median()) * 10000, 2),
        "median_half_spread_bps": round(half_spread * 10000, 2),
        "p75_round_trip_bps": round(float(hl.quantile(0.75)) * 10000, 2),
        "note": "Daily range overstates true bid-ask; use broker Bid/Ask when available.",
    }


def welch_mean_test(event_vals: list, base_vals: list, alpha: float) -> dict:
    if len(event_vals) < 5 or len(base_vals) < 5:
        return {
            "event_mean_pct": None,
            "baseline_mean_pct": None,
            "excess_mean_pct": None,
            "t_stat": None,
            "p_value": None,
            "significant": False,
            "alpha_used": alpha,
        }
    t_stat, p_two = stats.ttest_ind(event_vals, base_vals, equal_var=False)
    p_one = float(p_two / 2) if t_stat > 0 else 1.0
    ev_m = float(np.mean(event_vals))
    ba_m = float(np.mean(base_vals))
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


def cost_adjusted_excess(mean_excess_pct: float | None, round_trip_bps: float) -> float | None:
    if mean_excess_pct is None:
        return None
    return round(mean_excess_pct - round_trip_bps / 100, 3)


def run_uvix_primary(
    uvix: pd.Series,
    event_type: str,
    event_dates: list,
    all_event_days: set,
) -> dict:
    cfg = EVENT_WINDOW_CONFIG[event_type]
    pre_start, pre_end = PRIMARY_PRE_WINDOW
    post_off = PRIMARY_POST_WINDOW

    ev_pre = event_pre_returns(uvix, event_dates, pre_start, pre_end, skip_action_dates=UVIX_SPLIT_DATES)
    ba_pre = baseline_pre_returns(uvix, all_event_days, pre_start, pre_end, stride=BASELINE_STRIDE, skip_action_dates=UVIX_SPLIT_DATES)
    h1_hit = test_window(ev_pre, ba_pre, "up", one_sided=True, alpha=BONFERRONI_ALPHA)
    h1_hit["window"] = window_label_pre(pre_start, pre_end)
    h1_mean = welch_mean_test(ev_pre, ba_pre, alpha=H1_MEAN_ALPHA)
    h1_mean["window"] = h1_hit["window"]

    ev_post = event_post_returns(uvix, event_dates, post_off, skip_action_dates=UVIX_SPLIT_DATES)
    ba_post = baseline_post_returns(uvix, all_event_days, post_off, stride=BASELINE_STRIDE, skip_action_dates=UVIX_SPLIT_DATES)
    h2_hit = test_window(ev_post, ba_post, "down", one_sided=True, alpha=BONFERRONI_ALPHA)
    h2_hit["window"] = window_label_post(post_off)

    return {
        "event": cfg["label"],
        "event_type": event_type,
        "n_events": len(event_dates),
        "pre_window": h1_hit["window"],
        "post_window": h2_hit["window"],
        "h1_pre_rise_hit_rate": h1_hit,
        "h1_pre_rise_mean_return": h1_mean,
        "h2_post_fall_hit_rate": h2_hit,
    }


def recent_fomc_subsample(uvix: pd.Series, fomc_dates: list, years: int = RECENT_YEARS) -> dict:
    cutoff = uvix.index[-1] - pd.DateOffset(years=years)
    recent = [d for d in fomc_dates if pd.Timestamp(d) >= cutoff]
    if not recent:
        return {"error": "近 2y 无 FOMC 样本", "n_events": 0}

    pre_start, pre_end = PRIMARY_PRE_WINDOW
    ev_pre = event_pre_returns(uvix, recent, pre_start, pre_end, skip_action_dates=UVIX_SPLIT_DATES)
    if len(ev_pre) < 3:
        return {"error": "近 2y FOMC 有效样本不足", "n_events": len(recent)}

    mean_ret = float(np.mean(ev_pre))
    # one-sample t vs 0 (one-sided negative check for gate)
    t_stat, p_two = stats.ttest_1samp(ev_pre, 0.0)
    p_one_neg = float(p_two / 2) if t_stat < 0 else 1.0
    return {
        "years": years,
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "n_events": len(ev_pre),
        "mean_return_pct": round(mean_ret, 3),
        "hit_rate_up": round(sum(1 for r in ev_pre if r > 0) / len(ev_pre), 3),
        "t_stat_vs_zero": round(float(t_stat), 3),
        "p_one_sided_negative": round(p_one_neg, 4),
        "significantly_negative": p_one_neg < 0.05 and mean_ret < 0,
        "gate_pass": mean_ret > 0 or p_one_neg >= 0.05,
    }


def evaluate_s6_gate(primary_rows: list, costs: dict, recent_fomc: dict) -> dict:
    fomc = next(r for r in primary_rows if r["event_type"] == "fomc")
    h1_mean = fomc["h1_pre_rise_mean_return"]
    base_total_bps = costs["base_total_bps"]
    cost_adj = cost_adjusted_excess(h1_mean.get("excess_mean_pct"), base_total_bps)

    checks = {
        "fomc_h1_mean_bonferroni": {
            "pass": h1_mean.get("significant") and (h1_mean.get("excess_mean_pct") or 0) > 0,
            "p_value": h1_mean.get("p_value"),
            "alpha": BONFERRONI_ALPHA,
            "note": "F001b 门禁用 mean return；Bonferroni α 同 6-test family",
        },
        "fomc_cost_adjusted_excess_positive": {
            "pass": cost_adj is not None and cost_adj > 0,
            "cost_adjusted_excess_pct": cost_adj,
            "base_total_bps": base_total_bps,
        },
        "recent_2y_fomc_not_significantly_negative": {
            "pass": recent_fomc.get("gate_pass", False),
            "detail": recent_fomc,
        },
    }
    all_pass = all(c["pass"] for c in checks.values())
    return {
        "all_pass": all_pass,
        "verdict": "conditional_or_confirmed_pending" if all_pass else "fail — trading layer dead per F001b",
        "checks": checks,
    }


def fetch_vix_chart_series(start: pd.Timestamp) -> pd.Series:
    url = "https://query2.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=5y"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    rows = []
    for i, t in enumerate(ts):
        if closes[i] is None:
            continue
        d = pd.Timestamp.utcfromtimestamp(t).tz_localize(None).normalize()
        if d >= start:
            rows.append((d, closes[i]))
    if not rows:
        raise RuntimeError("VIX chart API returned no rows")
    return pd.Series(dict(rows)).sort_index()


def build_vix_uvix_mapping(uvix: pd.Series, date_map: dict) -> dict:
    """S6-3: same-window FOMC/CPI/NFP returns — VIX vs UVIX scatter + hit rates."""
    vix = fetch_vix_chart_series(uvix.index[0])
    common = uvix.index.intersection(vix.index)
    uvix_a = uvix.loc[common]
    vix_a = vix.loc[common]
    pre_start, pre_end = PRIMARY_PRE_WINDOW

    def paired_returns(series: pd.Series, event_dates: list) -> list[dict]:
        out = []
        idx = series.index
        for ed in event_dates:
            t0 = event_trading_day(idx, pd.Timestamp(ed))
            if t0 is None:
                continue
            r = pct_return(
                series,
                trading_offset(idx, t0, -pre_start),
                trading_offset(idx, t0, -pre_end),
                skip_action_dates=UVIX_SPLIT_DATES if series.name == "UVIX" else None,
            )
            if r is not None:
                out.append({"date": pd.Timestamp(ed).strftime("%Y-%m-%d"), "return_pct": r})
        return out

    by_event = []
    all_pairs = []
    for et in EVENT_WINDOW_CONFIG:
        dates = date_map[et]
        u = {r["date"]: r["return_pct"] for r in paired_returns(uvix_a, dates)}
        v = {r["date"]: r["return_pct"] for r in paired_returns(vix_a, dates)}
        shared = sorted(set(u) & set(v))
        pairs = [{"date": d, "uvix_pct": u[d], "vix_pct": v[d]} for d in shared]
        if len(pairs) >= 5:
            uu = [p["uvix_pct"] for p in pairs]
            vv = [p["vix_pct"] for p in pairs]
            corr = float(np.corrcoef(uu, vv)[0, 1])
            slope = float(np.polyfit(vv, uu, 1)[0]) if np.std(vv) > 0 else None
        else:
            corr, slope = None, None
        u_hit = round(sum(1 for p in pairs if p["uvix_pct"] > 0) / len(pairs), 3) if pairs else None
        v_hit = round(sum(1 for p in pairs if p["vix_pct"] > 0) / len(pairs), 3) if pairs else None
        by_event.append({
            "event": EVENT_WINDOW_CONFIG[et]["label"],
            "event_type": et,
            "window": window_label_pre(pre_start, pre_end),
            "n_pairs": len(pairs),
            "correlation": round(corr, 4) if corr is not None else None,
            "regression_uvix_on_vix_slope": round(slope, 3) if slope is not None else None,
            "hit_rate_up_uvix": u_hit,
            "hit_rate_up_vix": v_hit,
            "hit_rate_gap_uvix_minus_vix": round(u_hit - v_hit, 3) if u_hit is not None and v_hit is not None else None,
            "mean_return_uvix_pct": round(float(np.mean([p["uvix_pct"] for p in pairs])), 3) if pairs else None,
            "mean_return_vix_pct": round(float(np.mean([p["vix_pct"] for p in pairs])), 3) if pairs else None,
            "pairs": pairs,
        })
        all_pairs.extend(pairs)

    if len(all_pairs) >= 5:
        all_u = [p["uvix_pct"] for p in all_pairs]
        all_v = [p["vix_pct"] for p in all_pairs]
        pooled_corr = round(float(np.corrcoef(all_u, all_v)[0, 1]), 4)
    else:
        pooled_corr = None

    return {
        "description": "同窗口 T-5→T-1 逐事件配对：^VIX vs UVIX 收益与命中率",
        "period": f"{uvix.index[0].strftime('%Y-%m-%d')} ~ {uvix.index[-1].strftime('%Y-%m-%d')}",
        "pooled_correlation": pooled_corr,
        "by_event": by_event,
        "interpretation_note": (
            "高相关 = 方向同涨同跌；斜率>2 或 UVIX 均值更负 = 杠杆+衰减放大损失。"
            "命中率差 = UVIX 涨的比例与 VIX 不一致。"
        ),
    }


def build_output() -> dict:
    print("Fetching UVIX (max history)...")
    uvix_df = fetch_uvix_history()
    uvix = uvix_close_series(uvix_df)
    print(f"  UVIX: {len(uvix)} days, {uvix.index[0].date()} ~ {uvix.index[-1].date()}")

    event_dates = get_event_dates()
    uvix_start_year = uvix.index[0].year
    nfp_dates = get_nfp_dates(start_year=uvix_start_year, end_year=uvix.index[-1].year + 1)

    date_map = {
        "fomc": filter_event_dates_in_range(uvix, event_dates["fomc"]),
        "cpi": filter_event_dates_in_range(uvix, event_dates["cpi"]),
        "nfp": filter_event_dates_in_range(uvix, nfp_dates),
    }
    print("  Events in UVIX range:", {k: len(v) for k, v in date_map.items()})

    all_event_days = (
        collect_event_trading_days(uvix, date_map["fomc"])
        | collect_event_trading_days(uvix, date_map["cpi"])
        | collect_event_trading_days(uvix, date_map["nfp"])
    )

    primary_rows = [
        run_uvix_primary(uvix, et, date_map[et], all_event_days)
        for et in EVENT_WINDOW_CONFIG
    ]

    median_price = float(uvix.median())
    spread = estimate_spread_bps(uvix_df)
    broker_bps = broker_round_trip_bps(TYPICAL_NOTIONAL_USD, median_price)
    spread_rt_bps = spread.get("median_half_spread_bps")
    if spread_rt_bps is not None:
        spread_rt_bps = min(round(spread_rt_bps * 2, 2), 80.0)  # cap range proxy
    else:
        spread_rt_bps = 40.0  # conservative placeholder until broker bid/ask
    base_total_bps = round(broker_bps + spread_rt_bps, 2)
    stress_total_bps = round(base_total_bps * STRESS_MULTIPLIER, 2)

    costs = {
        "typical_notional_usd": TYPICAL_NOTIONAL_USD,
        "median_uvix_price_usd": round(median_price, 2),
        "broker_round_trip_bps": broker_bps,
        "spread_proxy": spread,
        "spread_round_trip_bps": spread_rt_bps,
        "base_total_bps": base_total_bps,
        "stress_total_bps": stress_total_bps,
        "stress_multiplier": STRESS_MULTIPLIER,
        "execution_live": "MOC recommended (see F001b)",
    }

    for row in primary_rows:
        ex = row["h1_pre_rise_mean_return"].get("excess_mean_pct")
        row["h1_pre_rise_mean_return"]["cost_adjusted_excess_pct"] = {
            "base": cost_adjusted_excess(ex, base_total_bps),
            "stress": cost_adjusted_excess(ex, stress_total_bps),
        }

    recent_fomc = recent_fomc_subsample(uvix, date_map["fomc"])
    gate = evaluate_s6_gate(primary_rows, costs, recent_fomc)

    # Hit-rate oriented summary (same shape as VIX primary for dashboard reuse)
    hit_summary_rows = [
        {
            "event": r["event"],
            "pre_window": r["pre_window"],
            "h1_pre_rise": r["h1_pre_rise_hit_rate"],
            "h2_post_fall": r["h2_post_fall_hit_rate"],
        }
        for r in primary_rows
    ]

    pre_start, pre_end = PRIMARY_PRE_WINDOW
    base_pre = baseline_pre_returns(
        uvix, all_event_days, pre_start, pre_end, stride=BASELINE_STRIDE, skip_action_dates=UVIX_SPLIT_DATES
    )

    mapping = build_vix_uvix_mapping(uvix, date_map)

    return {
        "title": "UVIX × Event Calendar Analysis (F001b)",
        "subtitle": "Pre-registered product layer · T-5~T-1 close → close",
        "case_id": "F001b",
        "parent_case": "F001",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "instrument": "UVIX (Yahoo Finance)",
            "data_range": f"{uvix.index[0].strftime('%Y-%m-%d')} ~ {uvix.index[-1].strftime('%Y-%m-%d')}",
            "trading_days": len(uvix),
            "event_counts": {k: len(v) for k, v in date_map.items()},
            "median_uvix_close": round(median_price, 2),
        },
        "costs": costs,
        "primary_analysis": {
            "method": (
                f"F001b 预注册 · stride={BASELINE_STRIDE} · "
                f"hit-rate Bonferroni α={BONFERRONI_ALPHA} ({N_PRIMARY_TESTS} tests) · "
                f"mean-return sensitivity α={H1_MEAN_ALPHA}"
            ),
            "windows": {
                "h1_pre": window_label_pre(*PRIMARY_PRE_WINDOW),
                "h2_post": window_label_post(PRIMARY_POST_WINDOW),
            },
            "baseline_reference": {
                "pre_mean_pct": round(float(np.mean(base_pre)), 3) if base_pre else None,
                "pre_hit_rate_up": round(sum(1 for r in base_pre if r > 0) / len(base_pre), 3) if base_pre else None,
                "pre_n": len(base_pre),
            },
            "by_event": primary_rows,
            "conclusion_summary": build_conclusion_summary(hit_summary_rows),
        },
        "recent_fomc_subsample": recent_fomc,
        "s6_gate_preview": gate,
        "vix_uvix_mapping": mapping,
        "methodology": {
            "execution_spec": "T-5 close buy → T-1 close sell (long UVIX)",
            "baseline_stride": BASELINE_STRIDE,
            "bonferroni_primary": {"tests": N_PRIMARY_TESTS, "alpha": BONFERRONI_ALPHA},
            "h1_mean_alpha": H1_MEAN_ALPHA,
            "not_included_in_bps": "contango / daily rebalance decay (in UVIX returns)",
        },
    }


def main():
    output = build_output()
    out_path = DATA_DIR / "event_uvix_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    map_path = DATA_DIR / "event_uvix_vix_mapping.json"
    with open(map_path, "w") as f:
        json.dump(output["vix_uvix_mapping"], f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"Saved {map_path}")
    gate = output["s6_gate_preview"]
    print(f"  S6 gate preview: {'PASS' if gate['all_pass'] else 'FAIL'}")
    sync_public_data()


if __name__ == "__main__":
    main()

"""
Composite Risk Score Engine

Converts raw indicator values into a single 0-100 risk score using:
  1. Sigmoid mapping per indicator (raw → individual 0-100 score)
  2. Weighted aggregation across indicators
  3. Acceleration bonus for rapid deterioration
"""

import json
import math
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def sigmoid_score(value: float, midpoint: float, steepness: float, invert: bool = False) -> float:
    """
    Map a raw value to 0-100 using a sigmoid curve.
    - midpoint: value at which score = 50
    - steepness: how fast the transition is (higher = sharper)
    - invert: if True, lower values are more dangerous
    """
    x = steepness * (value - midpoint)
    if invert:
        x = -x
    score = 100 / (1 + math.exp(-x))
    return max(0, min(100, score))


def term_spread_score(value: float) -> float:
    """Term spread: negative (inverted) = dangerous."""
    # midpoint at +0.2%, steep transition, invert so lower = more dangerous
    return sigmoid_score(value, midpoint=0.2, steepness=3.0, invert=True)


def credit_spread_hy_score(value: float) -> float:
    """HY credit spread: higher = more dangerous."""
    # midpoint at 4.5%, moderate transition
    return sigmoid_score(value, midpoint=4.5, steepness=1.2, invert=False)


def vix_score(value: float) -> float:
    """VIX: higher = more dangerous."""
    # midpoint at 22, moderate steepness
    return sigmoid_score(value, midpoint=22, steepness=0.25, invert=False)


def absorption_ratio_score(value: float, mean: float, std: float) -> float:
    """AR: z-score based, higher z = more dangerous."""
    z = (value - mean) / std if std > 0 else 0
    return sigmoid_score(z, midpoint=1.2, steepness=1.5, invert=False)


def turbulence_score(value: float, p50: float, p90: float, p99: float) -> float:
    """Turbulence: percentile-based scoring."""
    if value <= p50:
        return (value / p50) * 15  # 0-15 range for below-median
    elif value <= p90:
        return 15 + ((value - p50) / (p90 - p50)) * 35  # 15-50 range
    elif value <= p99:
        return 50 + ((value - p90) / (p99 - p90)) * 35  # 50-85 range
    else:
        return min(100, 85 + ((value - p99) / p99) * 15)  # 85-100 range


def breadth_score(value: float) -> float:
    """Market breadth: lower = more dangerous."""
    # midpoint at 50%, invert so lower breadth = higher score
    return sigmoid_score(value, midpoint=50, steepness=0.08, invert=True)


# Weights by causal layer
WEIGHTS = {
    "term_spread": 0.15,
    "credit_spread": 0.25,
    "vix": 0.15,
    "absorption_ratio": 0.15,
    "turbulence": 0.15,
    "breadth": 0.15,
}


def compute_composite_score(data: dict) -> dict:
    """
    Compute the composite risk score from all indicator JSON data.
    Returns a time series of daily composite scores + component breakdown.
    """
    term_spread = {d["date"]: d["term_spread_10y2y"] for d in data.get("term_spread", [])}
    credit_spread = {d["date"]: d.get("high_yield_spread") for d in data.get("credit_spread", [])}
    vix = {d["date"]: d["vix"] for d in data.get("vix", [])}
    sp500 = {d["date"]: d["sp500"] for d in data.get("sp500", [])}
    breadth = {d["date"]: d["pct_above_200ma"] for d in data.get("breadth", [])}
    ar = {d["date"]: d["absorption_ratio"] for d in data.get("absorption_ratio", [])}
    turb = {d["date"]: d["turbulence"] for d in data.get("turbulence", [])}

    # Pre-compute AR statistics (rolling 252-day)
    ar_values = list(ar.values())
    ar_mean = np.mean(ar_values) if ar_values else 0.9
    ar_std = np.std(ar_values) if ar_values else 0.03

    # Pre-compute Turbulence percentiles
    turb_values = list(turb.values())
    turb_p50 = np.percentile(turb_values, 50) if turb_values else 10
    turb_p90 = np.percentile(turb_values, 90) if turb_values else 30
    turb_p99 = np.percentile(turb_values, 99) if turb_values else 80

    # Get all dates where we have at least turbulence + vix (most restrictive)
    all_dates = sorted(set(turb.keys()) & set(vix.keys()))

    results = []
    prev_scores = []

    for date in all_dates:
        components = {}
        weighted_sum = 0
        total_weight = 0

        # Term Spread
        if date in term_spread:
            s = term_spread_score(term_spread[date])
            components["term_spread"] = round(s, 1)
            weighted_sum += s * WEIGHTS["term_spread"]
            total_weight += WEIGHTS["term_spread"]

        # Credit Spread
        if date in credit_spread and credit_spread[date] is not None:
            s = credit_spread_hy_score(credit_spread[date])
            components["credit_spread"] = round(s, 1)
            weighted_sum += s * WEIGHTS["credit_spread"]
            total_weight += WEIGHTS["credit_spread"]

        # VIX
        if date in vix:
            s = vix_score(vix[date])
            components["vix"] = round(s, 1)
            weighted_sum += s * WEIGHTS["vix"]
            total_weight += WEIGHTS["vix"]

        # Absorption Ratio
        if date in ar:
            s = absorption_ratio_score(ar[date], ar_mean, ar_std)
            components["absorption_ratio"] = round(s, 1)
            weighted_sum += s * WEIGHTS["absorption_ratio"]
            total_weight += WEIGHTS["absorption_ratio"]

        # Turbulence
        if date in turb:
            s = turbulence_score(turb[date], turb_p50, turb_p90, turb_p99)
            components["turbulence"] = round(s, 1)
            weighted_sum += s * WEIGHTS["turbulence"]
            total_weight += WEIGHTS["turbulence"]

        # Breadth
        if date in breadth:
            s = breadth_score(breadth[date])
            components["breadth"] = round(s, 1)
            weighted_sum += s * WEIGHTS["breadth"]
            total_weight += WEIGHTS["breadth"]

        # Normalize if not all indicators available
        base_score = (weighted_sum / total_weight) if total_weight > 0 else 0

        # Acceleration bonus: if score jumped >15 points in 5 days
        acceleration = 0
        if len(prev_scores) >= 5:
            score_5d_ago = prev_scores[-5]
            if base_score - score_5d_ago > 15:
                acceleration = min(15, (base_score - score_5d_ago) - 15)

        # Persistence filter: require 3-day confirmation for high scores
        # (reduces false positives from single-day spikes like elections)
        persistence_penalty = 0
        if base_score > 60 and len(prev_scores) >= 3:
            recent_avg = np.mean(prev_scores[-3:])
            if recent_avg < 40:
                persistence_penalty = (base_score - recent_avg) * 0.3

        final_score = max(0, min(100, base_score + acceleration - persistence_penalty))

        results.append({
            "date": date,
            "composite_score": round(final_score, 1),
            "base_score": round(base_score, 1),
            "acceleration": round(acceleration, 1),
            "persistence_penalty": round(persistence_penalty, 1),
            "components": components,
        })

        prev_scores.append(base_score)

    return results


def get_score_label(score: float) -> dict:
    """Get human-readable label and color for a score."""
    if score <= 20:
        return {"level": "low", "label": "Low Risk", "color": "#34d399", "action": "正常持仓"}
    elif score <= 40:
        return {"level": "moderate", "label": "Moderate", "color": "#86efac", "action": "保持关注"}
    elif score <= 60:
        return {"level": "elevated", "label": "Elevated", "color": "#fbbf24", "action": "审视仓位，准备防御"}
    elif score <= 80:
        return {"level": "high", "label": "High Risk", "color": "#fb923c", "action": "减仓/增加对冲"}
    else:
        return {"level": "extreme", "label": "Extreme", "color": "#f87171", "action": "大幅减仓/现金为王"}


def main():
    """Compute composite score and save to JSON."""
    # Load all indicator data
    all_data = {}
    for name in ["term_spread", "credit_spread", "vix", "sp500", "breadth", "absorption_ratio", "turbulence"]:
        filepath = DATA_DIR / f"{name}.json"
        if filepath.exists():
            with open(filepath) as f:
                all_data[name] = json.load(f)

    # Compute scores
    scores = compute_composite_score(all_data)

    if not scores:
        print("No scores computed!")
        return

    latest = scores[-1]
    label = get_score_label(latest["composite_score"])

    print(f"Composite Risk Score: {latest['composite_score']:.0f}/100 [{label['label']}]")
    print(f"  Action: {label['action']}")
    print(f"  Base: {latest['base_score']:.1f} | Accel: +{latest['acceleration']:.1f} | Persist: -{latest['persistence_penalty']:.1f}")
    print(f"  Components:")
    for k, v in latest["components"].items():
        print(f"    {k}: {v:.0f}/100")

    # Save full time series
    with open(DATA_DIR / "composite_score.json", "w") as f:
        json.dump(scores, f)

    # Save latest summary
    summary_path = DATA_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    summary["composite_score"] = {
        "score": latest["composite_score"],
        "label": label["label"],
        "level": label["level"],
        "color": label["color"],
        "action": label["action"],
        "components": latest["components"],
        "date": latest["date"],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(scores)} daily scores to composite_score.json")


if __name__ == "__main__":
    main()

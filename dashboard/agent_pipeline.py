"""
Macro risk agent pipeline.

Reads news_headlines.json + model_metrics.json, calls Claude for a structured
risk report, and writes dashboard/data/agent_report.json.

Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
NEWS_PATH = DATA_DIR / "news_headlines.json"
METRICS_PATH = DATA_DIR / "model_metrics.json"
OUTPUT_PATH = DATA_DIR / "agent_report.json"

SYSTEM_PROMPT = """You are a senior macro risk analyst specializing in equity market crash risk.

You receive:
1) Today's finance news headlines (Reuters, CNBC, MarketWatch)
2) An ML model's current probability of a >5% S&P 500 drawdown within the next 20 trading days

Your job is to synthesize the news context with the ML signal into a concise risk assessment.

Respond with ONLY valid JSON (no markdown fences) matching this schema:
{
  "timestamp": "<ISO-8601 UTC>",
  "risk_score": <integer 0-100>,
  "risk_level": "<low|moderate|elevated|high|critical>",
  "ml_probability": <float 0-1>,
  "key_risks": ["<short risk phrase>", "..."],
  "news_analysis": "<2-3 sentences analyzing the news backdrop>",
  "recommendation": "<actionable recommendation for a risk manager>",
  "reasoning": "<brief chain of reasoning linking ML signal and news>"
}

Guidelines:
- risk_score should reflect both the ML probability and news tone (geopolitics, rates, credit, liquidity, earnings shocks).
- risk_level bands: low 0-20, moderate 21-40, elevated 41-60, high 61-80, critical 81-100.
- key_risks: 3-6 concrete items grounded in the headlines.
- Be calibrated: do not scream "critical" on routine volatility news.
- Write news_analysis, recommendation, and reasoning in Chinese (简体中文).
- key_risks may be Chinese short phrases.
"""


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _extract_ml_probability(metrics: dict[str, Any]) -> float:
    current = metrics.get("current_prediction") or {}
    if "probability" in current:
        return float(current["probability"])
    experiments = metrics.get("experiments") or []
    if experiments and "current_probability" in experiments[0]:
        return float(experiments[0]["current_probability"])
    raise KeyError("No current ML probability found in model_metrics.json")


def _build_user_prompt(headlines: list[dict[str, str]], ml_prob: float) -> str:
    lines = [
        f"ML crash probability (next 20d, >5% drawdown): {ml_prob:.4f} ({ml_prob * 100:.1f}%)",
        "",
        "Today's headlines:",
    ]
    if not headlines:
        lines.append("(no headlines available)")
    else:
        for i, h in enumerate(headlines, 1):
            lines.append(
                f"{i}. [{h.get('source', '?')}] {h.get('title', '')} "
                f"({h.get('published', 'n/a')})"
            )
    lines.append("")
    lines.append(f"Current UTC time: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("Return the JSON risk report now.")
    return "\n".join(lines)


def _score_to_level(score: int) -> str:
    if score <= 20:
        return "low"
    if score <= 40:
        return "moderate"
    if score <= 60:
        return "elevated"
    if score <= 80:
        return "high"
    return "critical"


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def _normalize_report(raw: dict[str, Any], ml_prob: float) -> dict[str, Any]:
    score = int(round(float(raw.get("risk_score", ml_prob * 100))))
    score = max(0, min(100, score))
    level = str(raw.get("risk_level") or _score_to_level(score)).lower().strip()
    if level not in {"low", "moderate", "elevated", "high", "critical"}:
        level = _score_to_level(score)

    key_risks = raw.get("key_risks") or []
    if not isinstance(key_risks, list):
        key_risks = [str(key_risks)]
    key_risks = [str(r).strip() for r in key_risks if str(r).strip()][:8]

    return {
        "timestamp": raw.get("timestamp") or dt.datetime.now(dt.timezone.utc).isoformat(),
        "risk_score": score,
        "risk_level": level,
        "ml_probability": round(float(raw.get("ml_probability", ml_prob)), 4),
        "key_risks": key_risks,
        "news_analysis": str(raw.get("news_analysis") or "").strip(),
        "recommendation": str(raw.get("recommendation") or "").strip(),
        "reasoning": str(raw.get("reasoning") or "").strip(),
    }


def call_claude(user_prompt: str) -> dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError("Empty response from Claude API")
    return _parse_json_response(text)


def main() -> int:
    if not NEWS_PATH.exists():
        print(f"ERROR: missing {NEWS_PATH} — run news_fetcher.py first", file=sys.stderr)
        return 1
    if not METRICS_PATH.exists():
        print(f"ERROR: missing {METRICS_PATH}", file=sys.stderr)
        return 1

    news = _load_json(NEWS_PATH)
    metrics = _load_json(METRICS_PATH)
    headlines = news.get("headlines") or []
    ml_prob = _extract_ml_probability(metrics)

    print(f"  Headlines: {len(headlines)}")
    print(f"  ML probability: {ml_prob:.4f}")

    user_prompt = _build_user_prompt(headlines, ml_prob)
    print("Calling Claude API…")
    raw = call_claude(user_prompt)
    report = _normalize_report(raw, ml_prob)

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Saved agent report → {OUTPUT_PATH}")
    print(f"  risk_score={report['risk_score']} level={report['risk_level']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: agent_pipeline failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

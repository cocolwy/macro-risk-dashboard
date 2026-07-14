"""
Macro risk agent pipeline.

Reads news_headlines.json + model_metrics.json, synthesizes a structured risk
report, and writes dashboard/data/agent_report.json.

Providers (env AGENT_LLM, default: local):
  local     — keyword/theme synthesizer + ML probability (no API, CI-safe)
  ollama    — local Ollama OpenAI-compatible /api/chat (OLLAMA_HOST, OLLAMA_MODEL)
  anthropic — Claude API (ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.request
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

# Theme lexicon: (theme_id, chinese_label, risk_delta, keywords)
THEME_LEXICON: list[tuple[str, str, int, tuple[str, ...]]] = [
    ("geopolitics", "地缘政治/能源冲击", 12, (
        "war", "iran", "israel", "hormuz", "oil", "sanction", "military",
        "strike", "invasion", "taiwan", "ukraine", "missile", "troops",
    )),
    ("rates", "利率/央行政策不确定性", 10, (
        "fed", "rate", "rates", "powell", "fomc", "yield", "treasury",
        "hawkish", "cut", "hike", "inflation", "cpi", "pce", "ecb",
    )),
    ("credit", "信用/银行体系压力", 14, (
        "credit", "default", "bank", "bailout", "insolvency", "downgrade",
        "spread", "junk", "debt ceiling", "bankruptcy", "liquidity crisis",
    )),
    ("volatility", "波动率/市场抛售", 11, (
        "crash", "selloff", "sell-off", "plunge", "rout", "volatility",
        "vix", "panic", "turmoil", "correction", "bear",
    )),
    ("growth", "增长放缓/衰退担忧", 9, (
        "recession", "slowdown", "gdp", "unemployment", "layoff", "weak",
        "contraction", "soft landing", "hard landing", "demand",
    )),
    ("earnings", "盈利下修/财报冲击", 7, (
        "earnings", "profit warning", "miss", "guidance", "downgrade",
        "revenue", "outlook cut",
    )),
    ("china", "中国/新兴市场外溢", 6, (
        "china", "yuan", "property", "evergrande", "emerging", "em ",
    )),
    ("ai_tech", "科技/AI 叙事波动", 3, (
        "ai ", "nvidia", "tech", "megacap", "bubble", "Magnificent",
    )),
]

POSITIVE_KEYWORDS = (
    "rally", "surge", "record high", "soft landing", "ceasefire", "deal",
    "beat", "strong jobs", "cooling inflation", "risk-on", "rebound",
)


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
    # Some local models wrap JSON with prose — grab first {...} block
    if not text.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
    return json.loads(text)


def _normalize_report(raw: dict[str, Any], ml_prob: float, provider: str) -> dict[str, Any]:
    score = int(round(float(raw.get("risk_score", ml_prob * 100))))
    score = max(0, min(100, score))
    level = str(raw.get("risk_level") or _score_to_level(score)).lower().strip()
    if level not in {"low", "moderate", "elevated", "high", "critical"}:
        level = _score_to_level(score)

    key_risks = raw.get("key_risks") or []
    if not isinstance(key_risks, list):
        key_risks = [str(key_risks)]
    key_risks = [str(r).strip() for r in key_risks if str(r).strip()][:8]

    report = {
        "timestamp": raw.get("timestamp") or dt.datetime.now(dt.timezone.utc).isoformat(),
        "risk_score": score,
        "risk_level": level,
        "ml_probability": round(float(raw.get("ml_probability", ml_prob)), 4),
        "key_risks": key_risks,
        "news_analysis": str(raw.get("news_analysis") or "").strip(),
        "recommendation": str(raw.get("recommendation") or "").strip(),
        "reasoning": str(raw.get("reasoning") or "").strip(),
        "provider": provider,
    }
    return report


def _match_themes(headlines: list[dict[str, str]]) -> list[tuple[str, str, int, int]]:
    """Return list of (theme_id, label, delta, hit_count) sorted by hits."""
    corpus = " ".join((h.get("title") or "") for h in headlines).lower()
    hits: list[tuple[str, str, int, int]] = []
    for theme_id, label, delta, keywords in THEME_LEXICON:
        count = sum(1 for kw in keywords if kw.lower() in corpus)
        if count > 0:
            hits.append((theme_id, label, delta, count))
    hits.sort(key=lambda x: (-x[3], -x[2]))
    return hits


def _positive_tilt(headlines: list[dict[str, str]]) -> int:
    corpus = " ".join((h.get("title") or "") for h in headlines).lower()
    return sum(1 for kw in POSITIVE_KEYWORDS if kw in corpus)


def synthesize_local(headlines: list[dict[str, str]], ml_prob: float) -> dict[str, Any]:
    """Deterministic local synthesizer — no external LLM required."""
    themes = _match_themes(headlines)
    pos = _positive_tilt(headlines)

    # Base score from ML, then nudge by news themes (capped)
    base = ml_prob * 100
    news_adj = 0
    for _, _, delta, count in themes[:4]:
        news_adj += min(delta, delta * 0.5 + count * 1.5)
    news_adj -= min(8, pos * 3)
    news_adj = max(-12, min(18, news_adj))
    score = int(round(max(0, min(100, base + news_adj))))

    top_themes = themes[:4]
    key_risks: list[str] = []
    for _, label, _, count in top_themes:
        key_risks.append(f"{label}（标题命中 {count}）")
    if not key_risks:
        key_risks.append("当日标题未触发显著宏观风险主题")
    key_risks.append(f"ML 未来20日>5%回撤概率 {ml_prob * 100:.1f}%")
    if ml_prob >= 0.5:
        key_risks.append("模型信号处于 elevated/high 区间，需提高仓位审视频率")

    theme_names = "、".join(label for _, label, _, _ in top_themes) or "无明显主导主题"
    sample_titles = [h.get("title", "") for h in headlines[:3] if h.get("title")]
    sample_txt = "；".join(sample_titles) if sample_titles else "（无可用标题）"

    if news_adj >= 6:
        tone = "新闻面偏风险偏好压制，对模型高概率形成共振。"
    elif news_adj <= -4:
        tone = "新闻面出现一定缓和信号，对模型风险读数有所对冲。"
    else:
        tone = "新闻主题与模型信号大体同向，未出现明显背离。"

    news_analysis = (
        f"今日共纳入 {len(headlines)} 条财经标题，主导主题为：{theme_names}。"
        f"代表性标题：{sample_txt}。"
        f"{tone}"
    )

    level = _score_to_level(score)
    if level in {"high", "critical"}:
        recommendation = (
            "建议降低新增风险暴露、检查对冲覆盖与杠杆；对高 beta 与流动性较差资产提高减仓优先级，"
            "并跟踪利率决议/地缘进展的日内冲击。"
        )
    elif level == "elevated":
        recommendation = (
            "维持防守偏中性仓位，避免追涨高估值成长；用期权或波动率工具做有限保护，"
            "等待 ML 概率回落或新闻主题降温后再恢复进攻。"
        )
    else:
        recommendation = (
            "风险读数可控，可按既定再平衡执行；仍建议关注隔夜地缘与利率意外，避免集中押注单一叙事。"
        )

    reasoning = (
        f"本地规则引擎：score = ML×100 ({base:.1f}) + 新闻主题调整 ({news_adj:+.1f}) → {score}。"
        f"主题命中：{theme_names}；缓和词命中 {pos}。"
        f"provider=local（无需 API）。"
    )

    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "risk_score": score,
        "risk_level": level,
        "ml_probability": round(ml_prob, 4),
        "key_risks": key_risks[:6],
        "news_analysis": news_analysis,
        "recommendation": recommendation,
        "reasoning": reasoning,
    }


def call_ollama(user_prompt: str) -> dict[str, Any]:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama unreachable at {host}: {exc}") from exc

    text = (body.get("message") or {}).get("content") or ""
    if not text:
        raise RuntimeError("Empty response from Ollama")
    return _parse_json_response(text)


def call_anthropic(user_prompt: str) -> dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
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


def resolve_provider() -> str:
    explicit = (os.environ.get("AGENT_LLM") or "").strip().lower()
    if explicit in {"local", "ollama", "anthropic"}:
        return explicit
    # Auto: prefer Ollama if reachable, else local
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1.5) as resp:
            if resp.status == 200:
                return "ollama"
    except Exception:  # noqa: BLE001
        pass
    return "local"


def generate_report(headlines: list[dict[str, str]], ml_prob: float, provider: str) -> dict[str, Any]:
    user_prompt = _build_user_prompt(headlines, ml_prob)
    if provider == "local":
        print("Using local rule synthesizer…")
        raw = synthesize_local(headlines, ml_prob)
    elif provider == "ollama":
        print(f"Calling Ollama ({os.environ.get('OLLAMA_MODEL', 'llama3.2')})…")
        raw = call_ollama(user_prompt)
    elif provider == "anthropic":
        print("Calling Claude API…")
        raw = call_anthropic(user_prompt)
    else:
        raise RuntimeError(f"Unknown AGENT_LLM provider: {provider}")
    return _normalize_report(raw, ml_prob, provider)


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
    provider = resolve_provider()

    print(f"  Headlines: {len(headlines)}")
    print(f"  ML probability: {ml_prob:.4f}")
    print(f"  Provider: {provider}")

    report = generate_report(headlines, ml_prob, provider)

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

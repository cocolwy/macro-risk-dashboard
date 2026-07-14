"""
Fetch today's finance headlines from free RSS feeds (Reuters, CNBC, MarketWatch).

Output: dashboard/data/news_headlines.json
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import re

import feedparser

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "news_headlines.json"

FEEDS: dict[str, str] = {
    # Official Reuters RSS endpoints are frequently blocked; Google News mirror works reliably.
    "Reuters": (
        "https://news.google.com/rss/search?"
        "q=site:reuters.com+(markets+OR+stocks+OR+economy+OR+fed)&hl=en-US&gl=US&ceid=US:en"
    ),
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    # FT direct RSS is paywalled; Google News mirror surfaces headlines reliably.
    "FT": (
        "https://news.google.com/rss/search?"
        "q=site:ft.com+(markets+OR+economy+OR+rates+OR+fed+OR+inflation)"
        "&hl=en-US&gl=US&ceid=US:en"
    ),
    # Fed official feeds — always macro-relevant, no noise filtering needed.
    "Fed FOMC": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    "Fed Speech": "https://www.federalreserve.gov/feeds/speeches.xml",
}

# Sources that are always macro-relevant — skip keyword allowlist check
TRUSTED_SOURCES: frozenset[str] = frozenset(["Fed FOMC", "Fed Speech"])

# Cap per source so the agent context stays focused
MAX_PER_SOURCE = 8
MAX_TOTAL = 30  # increased to accommodate 5 sources

# ── Relevance filter ────────────────────────────────────────────────────────

# Regex patterns that reliably indicate personal-finance / lifestyle / clickbait
# content with no macro relevance.
_BLOCKLIST_PATTERNS: list[re.Pattern[str]] = [re.compile(p, re.I) for p in [
    r"\bI('m| am)\s+\d+\s*(years? old|and)",   # "I'm 45 and..."
    r"\bturn\s+\$[\d,]+\s+into",                # "turn $1,000 into $250,000"
    r"\bhow (can|do|to|should) (I|you|we)\b",   # "How can I / How to..."
    r"\b(retirement|retire)\b.*\b(tip|trick|plan|saving|advice)\b",
    r"\bbest (credit card|savings account|CD rate|high.yield|checking)\b",
    r"\b(mortgage rate|home loan|refinanc)\b.*\b(tip|advice|how|guide)\b",
    r"\b\d+ (ways?|tips?|thing|step|reason|mistake)\b.*\b(invest|sav|budget|retire)\b",
    r"\b(avoid|beware|don.t make)\b.*\b(mistake|error)\b.*\b(invest|sav|retire|stock)\b",
    r"\bhere.s (why|how|what)\b.*\b(you|your|i |we )",
    r"\b(celebrity|divorce|wedding|house hunting|dream home)\b",
    r"\bsuze orman\b|\bdave ramsey\b",           # personal-finance TV hosts
]]

# At least ONE of these macro keywords must appear in the title (case-insensitive)
# for the headline to be considered relevant. Headlines with zero hits are dropped.
_MACRO_KEYWORDS: frozenset[str] = frozenset([
    "fed", "fomc", "powell", "rate", "rates", "yield", "treasury", "bond",
    "inflation", "cpi", "pce", "gdp", "recession", "economic", "economy",
    "market", "stock", "stocks", "s&p", "nasdaq", "dow", "equity", "equities",
    "earnings", "profit", "revenue", "guidance", "outlook",
    "oil", "crude", "energy", "commodity", "commodities", "gold",
    "dollar", "euro", "yen", "currency", "forex",
    "china", "trade", "tariff", "sanction", "geopolit",
    "bank", "credit", "debt", "deficit", "fiscal", "budget",
    "war", "iran", "israel", "ukraine", "russia", "taiwan",
    "tech", "nvidia", "ai", "semiconductor",
    "job", "jobs", "unemployment", "payroll", "nonfarm",
    "ipo", "merger", "acquisition", "bankruptcy", "default",
    "hedge fund", "private equity", "wall street",
])


def _is_macro_relevant(title: str) -> bool:
    """Return False if the headline is personal-finance/lifestyle content."""
    lower = title.lower()
    # Block if matches any personal-finance pattern
    for pat in _BLOCKLIST_PATTERNS:
        if pat.search(title):
            return False
    # Require at least one macro keyword
    words = re.split(r"\W+", lower)
    word_set = set(words)
    for kw in _MACRO_KEYWORDS:
        if " " in kw:  # multi-word phrase
            if kw in lower:
                return True
        else:
            if kw in word_set:
                return True
    return False


def _parse_published(entry: Any) -> str | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                return dt.datetime(*parsed[:6], tzinfo=dt.timezone.utc).isoformat()
            except (TypeError, ValueError):
                pass
    for key in ("published", "updated"):
        raw = entry.get(key)
        if raw:
            return str(raw)
    return None


def _is_today_or_recent(published: str | None, now: dt.datetime, max_age_hours: int = 36) -> bool:
    """Keep headlines from roughly the last trading day window."""
    if not published:
        return True  # keep undated items; better than dropping everything
    try:
        ts = dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    age = now - ts
    return age <= dt.timedelta(hours=max_age_hours)


def _clean_description(raw: str, max_chars: int = 300) -> str:
    """Strip HTML tags and collapse whitespace; return empty string if not useful."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    # Drop if it's just a media placeholder or too short to be useful
    if len(text) < 20:
        return ""
    return text[:max_chars]


def fetch_headlines() -> list[dict[str, str]]:
    now = dt.datetime.now(dt.timezone.utc)
    headlines: list[dict[str, str]] = []
    seen_titles: set[str] = set()

    for source, url in FEEDS.items():
        try:
            feed = feedparser.parse(url)
        except Exception as exc:  # noqa: BLE001 — network/parse resilience
            print(f"  WARN: failed to fetch {source}: {exc}")
            continue

        if getattr(feed, "bozo", False) and not feed.entries:
            print(f"  WARN: {source} feed parse error: {getattr(feed, 'bozo_exception', 'unknown')}")
            continue

        count = 0
        for entry in feed.entries:
            if count >= MAX_PER_SOURCE:
                break
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            # Google News appends " - Source Name"; strip it for cleaner titles
            for suffix in (f" - {source}", f" – {source}", f" | {source}",
                           " - Financial Times", " – Financial Times",
                           " - Reuters", " – Reuters"):
                if title.endswith(suffix):
                    title = title[: -len(suffix)].strip()
                    break

            # Trusted sources (Fed) are always relevant — skip noise filter
            if source not in TRUSTED_SOURCES and not _is_macro_relevant(title):
                continue

            title_key = title.lower()
            if title_key in seen_titles:
                continue

            published = _parse_published(entry)
            if not _is_today_or_recent(published, now):
                continue

            link = (entry.get("link") or "").strip()

            # RSS description: first paragraph, strip HTML tags, cap at 300 chars
            raw_desc = (
                entry.get("summary")
                or entry.get("description")
                or entry.get("content", [{}])[0].get("value", "")
                if isinstance(entry.get("content"), list) else
                entry.get("summary") or entry.get("description") or ""
            )
            desc = _clean_description(str(raw_desc))

            headlines.append({
                "title": title,
                "source": source,
                "published": published or now.isoformat(),
                "url": link,
                "description": desc,
            })
            seen_titles.add(title_key)
            count += 1

        print(f"  {source}: {count} headlines")

    # Prefer most recent when trimming
    def sort_key(h: dict[str, str]) -> str:
        return h.get("published") or ""

    headlines.sort(key=sort_key, reverse=True)
    return headlines[:MAX_TOTAL]


def main() -> None:
    print("Fetching finance headlines from RSS…")
    headlines = fetch_headlines()
    payload = {
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "headlines": headlines,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Saved {len(headlines)} headlines → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

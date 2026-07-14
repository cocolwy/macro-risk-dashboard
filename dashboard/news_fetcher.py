"""
Fetch today's finance headlines from free RSS feeds (Reuters, CNBC, MarketWatch).

Output: dashboard/data/news_headlines.json
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

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
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
}

# Cap per source so the agent context stays focused
MAX_PER_SOURCE = 8
MAX_TOTAL = 24


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
            # Google News often appends " - Reuters"
            for suffix in (f" - {source}", f" – {source}", f" | {source}"):
                if title.endswith(suffix):
                    title = title[: -len(suffix)].strip()
                    break
            title_key = title.lower()
            if title_key in seen_titles:
                continue

            published = _parse_published(entry)
            if not _is_today_or_recent(published, now):
                continue

            link = (entry.get("link") or "").strip()
            headlines.append({
                "title": title,
                "source": source,
                "published": published or now.isoformat(),
                "url": link,
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

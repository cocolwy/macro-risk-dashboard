"""
Fetch today's finance headlines from RSS feeds (Reuters, CNBC, FT, Fed).

Output: dashboard/data/news_headlines.json
"""

from __future__ import annotations

import datetime as dt
import json
import re
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
    # FT direct RSS is paywalled; Google News mirror surfaces headlines reliably.
    "FT": (
        "https://news.google.com/rss/search?"
        "q=site:ft.com+(markets+OR+economy+OR+rates+OR+fed+OR+inflation)"
        "&hl=en-US&gl=US&ceid=US:en"
    ),
    # Fed official feeds — FOMC statements and governor speeches.
    "Fed FOMC": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    "Fed Speech": "https://www.federalreserve.gov/feeds/speeches.xml",
}

# Cap per source so the agent context stays focused
MAX_PER_SOURCE = 8
MAX_TOTAL = 30


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

            # Google News appends " - Source Name"; strip for cleaner titles
            for suffix in (f" - {source}", f" – {source}", f" | {source}",
                           " - Financial Times", " – Financial Times",
                           " - Reuters", " – Reuters"):
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

            # RSS description: first paragraph, strip HTML, cap at 300 chars
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

    headlines.sort(key=lambda h: h.get("published") or "", reverse=True)
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

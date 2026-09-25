"""Web search helpers using DuckDuckGo."""

import time
from typing import Any

from ddgs import DDGS


def search_web(queries: list[str], max_results_per_query: int = 5) -> list[dict[str, str]]:
    """Search each planned query, retrying once when DuckDuckGo fails."""
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for query in queries:
        query_results: list[dict[str, Any]] = []
        for attempt in range(2):
            try:
                query_results = list(DDGS().text(query, max_results=max_results_per_query))
                if query_results:
                    break
            except Exception:
                pass
            if attempt == 0:
                time.sleep(1)

        for item in query_results:
            url = str(item.get("href") or item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                {
                    "title": str(item.get("title") or url),
                    "url": url,
                    "snippet": str(item.get("body") or item.get("snippet") or ""),
                }
            )

    return results

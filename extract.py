"""Fetch and extract readable text from web pages."""

from typing import Any

import requests
import trafilatura

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"


def extract_page(url: str, timeout: int = 8) -> str | None:
    """Return cleaned page text, or None when the URL cannot be read."""
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=timeout,
            allow_redirects=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if content_type and "html" not in content_type and "text/" not in content_type:
            return None
        extracted = trafilatura.extract(response.text, include_comments=False, include_tables=True)
        return extracted if extracted and len(extracted.strip()) > 50 else None
    except (requests.RequestException, ValueError, TypeError):
        return None
    except Exception:
        return None


def extract_sources(results: list[dict[str, Any]], max_pages: int = 5) -> list[dict[str, str]]:
    """Extract up to max_pages successful pages while preserving result metadata."""
    sources: list[dict[str, str]] = []
    for result in results:
        if len(sources) >= max_pages:
            break
        text = extract_page(result["url"])
        if not text:
            continue
        sources.append(
            {
                "title": result["title"],
                "url": result["url"],
                "text": text[:3500],
            }
        )
    return sources

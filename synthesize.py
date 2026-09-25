"""Groq-powered query planning and source-grounded synthesis."""

import json
from typing import Any

from groq import Groq

PLANNER_SYSTEM = """You plan web research. Given a user's question, return only valid JSON in the form {\"queries\": [\"query 1\", \"query 2\"]}. Generate 1 to 3 specific search queries that together cover the question. Do not include markdown or explanation."""
SYNTHESIS_SYSTEM = """You are a careful research assistant. Answer the user's question ONLY using the provided sources. Cite every material claim inline using [1], [2], etc., matching the source labels. If sources disagree, explicitly say so and describe the disagreement. If the sources do not answer the question, say that clearly instead of filling gaps from general knowledge. Do not invent facts or citations. Use concise paragraphs and simple markdown when useful."""
GROQ_MODEL = "qwen/qwen3.8-27b"


def _client() -> Groq:
    return Groq()


def plan_queries(question: str) -> list[str]:
    """Ask Groq for focused search queries, with a useful fallback."""
    response = _client().chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=250,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    content = response.choices[0].message.content or ""
    try:
        data = json.loads(content)
        queries = data.get("queries", [])
        if isinstance(queries, list):
            cleaned = [str(query).strip() for query in queries if str(query).strip()]
            if cleaned:
                return cleaned[:3]
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return [question]


def synthesize_answer(question: str, sources: list[dict[str, str]]) -> str:
    """Generate an answer grounded only in extracted source text."""
    source_text = "\n\n".join(
        f"Source {index}: {source['url']}\nTitle: {source['title']}\n{source['text']}"
        for index, source in enumerate(sources, start=1)
    )
    response = _client().chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": f"Question: {question}\n\n{source_text}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()

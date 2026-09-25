"""FastAPI application for the Personal Research Assistant."""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from extract import extract_sources
from search import search_web
from synthesize import plan_queries, synthesize_answer

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Personal Research Assistant")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/research")
def research(request: ResearchRequest) -> dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a valid question.")

    try:
        queries = plan_queries(question)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Failed to plan research queries. Check your Groq API key and try again.",
        ) from exc

    try:
        search_results = search_web(queries)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Web search failed. Check your internet connection and try again.",
        ) from exc

    if not search_results:
        raise HTTPException(
            status_code=404,
            detail="No search results were found. Try rephrasing or asking a different question.",
        )

    sources = extract_sources(search_results, max_pages=5)
    if not sources:
        # Fallback to search snippets if full text extraction fails on all pages
        sources = [
            {
                "title": item["title"],
                "url": item["url"],
                "text": item["snippet"],
            }
            for item in search_results[:5]
            if item.get("snippet")
        ]

    if not sources:
        raise HTTPException(
            status_code=502,
            detail="Search results were found, but none of the pages could be read.",
        )

    try:
        answer = synthesize_answer(question, sources)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The answer service failed. Verify your Groq API key and try again.",
        ) from exc

    return {
        "answer": answer,
        "queries": queries,
        "sources": [
            {"id": index, "title": source["title"], "url": source["url"]}
            for index, source in enumerate(sources, start=1)
        ],
    }

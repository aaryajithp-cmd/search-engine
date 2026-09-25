"""FastAPI application for the Personal Research Assistant."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from extract import extract_sources
from search import search_web
from synthesize import plan_queries, synthesize_answer

load_dotenv()

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
    try:
        queries = plan_queries(question)
        search_results = search_web(queries)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Research planning or web search failed. Check your Groq API key and try again.") from exc

    if not search_results:
        raise HTTPException(status_code=404, detail="No search results were found. Try a more specific question.")

    sources = extract_sources(search_results, max_pages=5)
    if not sources:
        raise HTTPException(status_code=502, detail="Search results were found, but none of the pages could be read.")

    try:
        answer = synthesize_answer(question, sources)
    except Exception as exc:
        groq_status = getattr(exc, "status_code", None)
        if groq_status in (401, 403):
            detail = "Groq rejected the API key. Check GROQ_API_KEY in your .env file."
            response_status = 502
        elif groq_status in (413, 429):
            detail = "Groq's request limit was reached. Wait a moment and try again with a shorter question."
            response_status = 429
        else:
            detail = "The answer service failed. Check your Groq API key and try again."
            response_status = 502
        raise HTTPException(status_code=response_status, detail=detail) from exc

    return {
        "answer": answer,
        "sources": [
            {"id": index, "title": source["title"], "url": source["url"]}
            for index, source in enumerate(sources, start=1)
        ],
    }

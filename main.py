"""FastAPI application for the Personal Research Assistant."""

from pathlib import Path
from typing import Literal
from io import BytesIO
from xml.sax.saxutils import escape
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from extract import extract_sources
from search import search_web
from synthesize import chat_answer, plan_queries, synthesize_answer

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Personal Research Assistant")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[ConversationMessage] = Field(default_factory=list, max_length=8)
    mode: Literal["web", "chat"] = "web"


class PdfSource(BaseModel):
    id: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2000)


class PdfExportRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=12000)
    sources: list[PdfSource] = Field(default_factory=list, max_length=20)


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/export/pdf")
def export_pdf(request: PdfExportRequest) -> Response:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.78 * inch,
        leftMargin=0.78 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.72 * inch,
        title="Fieldnote research answer",
        author="Fieldnote",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "FieldnoteTitle", parent=styles["Title"], alignment=TA_LEFT,
        textColor=colors.HexColor("#202723"), fontName="Helvetica-Bold",
        fontSize=20, leading=25, spaceAfter=16,
    )
    label_style = ParagraphStyle(
        "FieldnoteLabel", parent=styles["Heading3"], textColor=colors.HexColor("#397768"),
        fontName="Helvetica-Bold", fontSize=9, leading=12, spaceBefore=10, spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "FieldnoteBody", parent=styles["BodyText"], textColor=colors.HexColor("#202723"),
        fontName="Helvetica", fontSize=10.5, leading=16, spaceAfter=8,
    )
    source_style = ParagraphStyle(
        "FieldnoteSource", parent=body_style, fontSize=9, leading=13, spaceAfter=6,
    )

    def pdf_text(value: str) -> str:
        return value.encode("cp1252", "replace").decode("cp1252")

    story = [
        Paragraph("Fieldnote", title_style),
        Paragraph("QUESTION", label_style),
        Paragraph(escape(pdf_text(request.question)), body_style),
        Paragraph("ANSWER", label_style),
    ]
    for paragraph in request.answer.split("\n"):
        story.append(Paragraph(escape(pdf_text(paragraph)) or "&nbsp;", body_style))

    valid_sources = [
        source for source in request.sources
        if urlparse(source.url).scheme in {"http", "https"}
    ]
    if valid_sources:
        story.extend([Spacer(1, 8), Paragraph("SOURCES", label_style)])
        for source in valid_sources:
            safe_url = escape(pdf_text(source.url), {'"': "&quot;"})
            safe_title = escape(pdf_text(source.title))
            story.append(Paragraph(
                f'<font color="#397768"><b>[{source.id}]</b></font> '
                f'<link href="{safe_url}" color="#397768">{safe_title}</link><br/>'
                f'<font size="8" color="#69736d">{escape(pdf_text(source.url))}</font>',
                source_style,
            ))

    def add_page_number(canvas, _document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#69736d"))
        canvas.drawRightString(letter[0] - 0.78 * inch, 0.42 * inch, f"Fieldnote  ·  {_document.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return Response(
        content=buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="fieldnote-answer.pdf"'},
    )


@app.post("/research")
def research(request: ResearchRequest) -> dict:
    question = request.question.strip()
    history = [message.model_dump() for message in request.history]
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a valid question.")

    if request.mode == "chat":
        try:
            answer = chat_answer(question, history)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="The chat service failed. Verify your Groq API key and try again.",
            ) from exc
        return {"answer": answer, "queries": [], "sources": []}

    if len(question) < 3:
        raise HTTPException(
            status_code=422,
            detail="Web research questions must be at least 3 characters.",
        )

    try:
        queries = plan_queries(question, history)
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
        answer = synthesize_answer(question, sources, history)
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

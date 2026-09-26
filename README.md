# Personal Research Assistant

A small FastAPI app with two assistant modes: conversational answers through Groq, or web research that plans searches, finds pages through DuckDuckGo, extracts readable text with Trafilatura, and returns a source-grounded answer with inline citations.

## Setup

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Create `.env` from `.env.example` and add `GROQ_API_KEY`.

   Get a free key at [console.groq.com](https://console.groq.com/). The app uses the free tier; no payment is required.

5. Run the app:

   ```powershell
   uvicorn main:app --reload
   ```

6. Open http://127.0.0.1:8000.

## Configuration

- `GROQ_API_KEY`: required for query planning and synthesis.
- Search uses DuckDuckGo without an API key.
- Page requests use a 10-second timeout and the pipeline caps extraction at five pages per question.

## Modes

- **Web search**: researches the question online and returns cited sources.
- **Normal chat**: answers conversationally with chat history and no web retrieval.
- Answers can be downloaded as PDFs with the question and source links included.

## Project layout

- `main.py`: FastAPI routes and pipeline orchestration
- `search.py`: DuckDuckGo search and retry handling
- `extract.py`: page fetching and Trafilatura extraction
- `synthesize.py`: Groq query planning and source-grounded answer generation
- `static/index.html`: vanilla frontend

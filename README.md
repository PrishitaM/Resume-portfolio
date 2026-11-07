# Prishita Matreja: portfolio with an AI assistant

FastAPI serves the site (`static/index.html`) and two endpoints:
`GET /api/profile` (the content) and `POST /api/chat` (RAG over that same content, answered by Groq).

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then paste your key into GROQ_API_KEY
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000

## Edit your content
Everything lives in `app/knowledge.json`: socials (fill `LinkedIn`), experience, projects, skills.
The website and the chatbot both read it, so one edit updates both.

## Deploy (Render, free tier)
1. Push this folder to GitHub (`.env` is git-ignored).
2. Render > New > Web Service > pick the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Environment: add `GROQ_API_KEY` (and optionally `GROQ_MODEL`).
The key stays on the server and is never sent to the browser.

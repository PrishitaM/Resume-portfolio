"""Portfolio backend. Chat answers come in three tiers:
1. Rules: curated answers in answers.json (instant, free, no LLM).
2. RAG + Groq: for questions the rules do not cover (needs GROQ_API_KEY).
3. Fallback: a friendly message with suggested questions.
The API key is read from the GROQ_API_KEY environment variable only."""
import json, math, os, re
from collections import Counter
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()
BASE = Path(__file__).parent
PROFILE = json.loads((BASE / "knowledge.json").read_text(encoding="utf-8"))
ANSWERS = json.loads((BASE / "answers.json").read_text(encoding="utf-8"))
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ---- Knowledge base: every fact on the site becomes a searchable chunk ----
def build_chunks(p):
    chunks = [f"About {p['name']}. {p['summary']} Education: {p['education']}"]
    chunks.append("Skills. " + " ".join(f"{k}: {', '.join(v)}." for k, v in p["skills"].items()))
    for e in p["experience"]:
        chunks.append(f"Experience: {e['role']} at {e['company']}, {e['dates']}. " + " ".join(e["points"]))
    for pr in p["projects"]:
        extra = " ".join(pr.get("highlights", []))
        chunks.append(f"Project: {pr['title']} (stack: {', '.join(pr['stack'])}). {pr['description']} {extra}")
    return chunks


def tokens(text):
    return re.findall(r"[a-z0-9#+]+", text.lower())


CHUNKS = build_chunks(PROFILE)
CHUNK_TOKENS = [Counter(tokens(c)) for c in CHUNKS]
DOC_FREQ = Counter(t for c in CHUNK_TOKENS for t in c)


def retrieve(query, k=4):
    """Small TF-IDF ranking: no vector DB needed for a portfolio-sized corpus."""
    q = tokens(query)
    scored = []
    for i, c in enumerate(CHUNK_TOKENS):
        s = sum((1 + math.log(c[t])) * math.log(1 + len(CHUNKS) / DOC_FREQ[t]) for t in q if t in c)
        scored.append((s, i))
    scored.sort(reverse=True)
    top = [CHUNKS[i] for s, i in scored[:k] if s > 0]
    return top or CHUNKS[:1]


# ---- Tier 1: rule-based answers ----
def word_set(text):
    ws = set(tokens(text))
    return ws | {w.rstrip("s") for w in ws if len(w) > 3}


def match_rule(message):
    """Score each entry: strong keywords 3 points, phrases 2, single words 1."""
    text = " " + " ".join(tokens(message)) + " "
    words = word_set(message)

    def hit(kw):
        kw = " ".join(tokens(kw))
        return (" " + kw + " " in text) if " " in kw else (kw in words)

    # Specific topics (a project, a skill area) beat broad overview entries.
    best, best_score = None, 0
    for generic in (False, True):
        for e in ANSWERS["entries"]:
            if e.get("generic", False) != generic:
                continue
            score = sum(3 for k in e["strong"] if hit(k)) + sum(2 if " " in k else 1 for k in e["keywords"] if hit(k))
            if score > best_score:
                best, best_score = e, score
        if best_score >= ANSWERS["threshold"]:
            break
    return best if best_score >= ANSWERS["threshold"] else None


SYSTEM = (
    "You are the assistant on {name}'s portfolio website. Answer visitor questions about {name} "
    "using ONLY the context below. Be concise (under 120 words), friendly, and in the third person. "
    "If the context does not contain the answer, say you don't have that detail and suggest "
    "contacting {name} at {email}. Never invent employers, dates, metrics or links.\n\nContext:\n{context}"
)

app = FastAPI(title="Portfolio")


class Msg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=1500)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    history: list[Msg] = Field(default_factory=list, max_length=12)


@app.get("/api/profile")
def profile():
    return PROFILE


@app.post("/api/chat")
async def chat(body: ChatIn):
    rule = match_rule(body.message)
    if rule:
        return {"reply": rule["answer"], "source": "rules", "followups": rule["followups"]}

    fb = ANSWERS["fallback"]
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return {"reply": fb["answer"], "source": "fallback", "followups": fb["followups"]}
    context = "\n\n".join(retrieve(body.message))
    system = SYSTEM.format(name=PROFILE["name"], email=PROFILE["email"], context=context)
    messages = [{"role": "system", "content": system}]
    messages += [m.model_dump() for m in body.history[-6:]]
    messages.append({"role": "user", "content": body.message})
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 350},
            )
        r.raise_for_status()
        return {"reply": r.json()["choices"][0]["message"]["content"].strip(), "source": "ai", "followups": fb["followups"]}
    except httpx.HTTPError:
        # Never show a raw error to visitors: degrade to the friendly fallback.
        return {"reply": fb["answer"], "source": "fallback", "followups": fb["followups"]}


app.mount("/", StaticFiles(directory=BASE.parent / "static", html=True), name="static")

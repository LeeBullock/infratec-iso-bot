from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------- Config ----------
BASE_DIR = Path(__file__).parent.resolve()
CHUNKS_PATH = BASE_DIR / "outputs" / "chunks.jsonl"

# Allow your Render domain (and local dev)
ALLOWED_ORIGINS = [
    "https://infratec-iso-bot.onrender.com",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# ---------- Load chunks ----------
def load_chunks() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not CHUNKS_PATH.exists():
        return items
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                # ignore corrupt lines
                pass
    return items

# in-memory cache
CHUNKS = load_chunks()

# ---------- App ----------
app = FastAPI(title="INFRATEC ISO Coach", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS + ["*"],  # keep simple; tighten later if you prefer
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def ui_root():
    """Serve index.html"""
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>INFRATEC ISO Coach</h1><p>Frontend not found.</p>", status_code=200)
    return FileResponse(index)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/_debug/chunks")
def debug_chunks():
    sample = []
    for c in CHUNKS[:5]:
        sample.append({
            "title": c.get("title") or Path(c.get("source_path","")).name,
            "path": c.get("source_path") or c.get("path") or ""
        })
    return {
        "count": len(CHUNKS),
        "sample": sample,
        "path": str(CHUNKS_PATH)
    }

@app.post("/_refresh")
def refresh_chunks():
    """Manually reload chunks.jsonl without redeploying."""
    global CHUNKS
    CHUNKS = load_chunks()
    return {"ok": True, "count": len(CHUNKS)}

# ---------- Very simple retrieval + answer synthesis ----------
# NOTE: This is the same lightweight approach you tested. It returns
# an answer based on simple keyword match across CHUNKS with citations.
def find_relevant_chunks(question: str, top_k: int = 4) -> List[Dict[str, Any]]:
    q = question.lower().strip()
    terms = [t for t in re.split(r"[^a-z0-9]+", q) if t]
    scored = []
    for item in CHUNKS:
        text = (item.get("text") or "").lower()
        score = sum(text.count(t) for t in terms)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:top_k]]

def synthesize_answer(question: str, ctx: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not ctx:
        return {
            "answer": "No relevant INFRATEC context found. Please consult the process owner.",
            "citations": []
        }

    # naive synthesis: join snippets
    bullets = []
    citations = []
    for item in ctx:
        snippet = (item.get("text") or "").strip()
        if snippet:
            bullets.append(snippet)
        citations.append({
            "title": item.get("title") or Path(item.get("source_path","")).name,
            "path": item.get("source_path") or item.get("path") or ""
        })

    joined = "\n\n".join(bullets[:3])
    answer = f"{joined}"

    return {"answer": answer, "citations": citations}

@app.post("/ask")
async def ask(payload: Dict[str, Any]):
    question = (payload or {}).get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")
    ctx = find_relevant_chunks(question)
    result = synthesize_answer(question, ctx)
    return JSONResponse(result)

# app.py — INFRATEC ISO Coach (debug-enabled)

import os, re, json
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI

app = FastAPI(title="INFRATEC ISO Coach API", version="0.2.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax while developing
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(ROOT_DIR, "outputs", "chunks.jsonl")
PROMPT_PATH = os.path.join(ROOT_DIR, "prompts", "system_prompt.txt")

# Load system prompt with safe fallback
try:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except Exception:
    SYSTEM_PROMPT = (
        "You are INFRATEC ISO Coach.\n"
        "Answer questions strictly using the provided context chunks.\n"
        "Always cite ISO clause numbers and INFRATEC documents.\n"
        "Paraphrase requirements (do not copy ISO text).\n"
        "Provide: (1) concise answer (2) ISO clause references (3) “What to do at INFRATEC” steps (4) Evidence list.\n"
        "If context is weak or outside scope, say so and propose the next step.\n"
        "Prefer UK terminology."
    )

def load_chunks():
    chunks = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(json.loads(line))
                except Exception:
                    pass
    return chunks

def normalize_tokens(text: str):
    toks = re.findall(r"[a-z0-9\-]+", (text or "").lower())
    norm = set()
    for t in toks:
        norm.add(t)
        if t.endswith("es"):
            norm.add(t[:-2])
        if t.endswith("s"):
            norm.add(t[:-1])
    return norm

def simple_retrieve(question: str, k: int = 6):
    q_norm = normalize_tokens(question)
    hits = []
    for ch in load_chunks():
        t_norm = normalize_tokens(ch.get("text", ""))
        score = len(q_norm.intersection(t_norm))
        if score > 0:
            hits.append((score, ch))
    hits.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in hits[:k]]

# Lightweight clause hints (nice-to-have)
CLAUSE_HINTS = [
    {"standard": "ISO9001:2015", "clause": "8.4",
     "terms": ["supplier","suppliers","externally provided","procurement","approved supplier","purchase order"]},
    {"standard": "ISO9001:2015", "clause": "7.5",
     "terms": ["document control","documented information","version control","records retention"]},
    {"standard": "ISO9001:2015", "clause": "9.2",
     "terms": ["internal audit","audit plan","audit report","nonconformity","corrective action"]},
]
def clause_hints_for(question: str):
    q = (question or "").lower()
    out = []
    for c in CLAUSE_HINTS:
        if any(t in q for t in c["terms"]):
            out.append(f'{c["standard"]} {c["clause"]}')
    return ", ".join(sorted(set(out)))

@app.get("/")
def serve_frontend_or_info():
    index_path = os.path.join(ROOT_DIR, "frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        {"service":"INFRATEC ISO Coach","endpoints":["/health","/ask","/docs","/_debug/chunks"],
         "note":"Add frontend/index.html to serve a UI at /"}
    )

@app.get("/health")
def health():
    return {"ok": True}

# NEW: debug endpoint so we can see what the server actually loaded
@app.get("/_debug/chunks")
def debug_chunks():
    chunks = load_chunks()
    sample = [{"title": c.get("title"), "path": c.get("source_path")} for c in chunks[:5]]
    return {"count": len(chunks), "sample": sample, "path": CHUNKS_PATH}

@app.post("/ask")
def ask(payload: dict = Body(...)):
    question = (payload.get("question") or "").strip()
    role = (payload.get("role") or "General").strip()
    if not question:
        return {"error": "Question is required."}

    ctx = simple_retrieve(question, k=6)
    if not ctx:
        return {"answer": "No relevant INFRATEC context found. Please consult the process owner.", "citations": []}

    sources = []
    for ch in ctx:
        ref_title = ch.get("title") or ch.get("doc_id") or "Unknown"
        path = ch.get("source_path") or ""
        snippet = (ch.get("text", ""))[:1200]
        sources.append(f"[{ref_title}] — {path}\n{snippet}")
    joined = "\n\n---\n\n".join(sources)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY",""))
    likely_clauses = clause_hints_for(question)
    user_block = (
        f"Role: {role}\n"
        f"Likely ISO clauses: {likely_clauses}\n\n"
        f"Question: {question}\n\n"
        f"Use ONLY the context below.\n\n"
        f"Context:\n{joined}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": user_block},
            ],
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Generation error: {e}"

    citations = [
        {"title": ch.get("title") or ch.get("doc_id") or "Unknown", "path": ch.get("source_path") or ""}
        for ch in ctx
    ]
    return {"answer": answer, "citations": citations}
# (paste the full app.py from above here)

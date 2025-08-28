# (cat > app.py << 'EOF'
import os, json
from pathlib import Path
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# --- FastAPI app ---
app = FastAPI(title="INFRATEC ISO Coach API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Paths ---
BASE_DIR = Path(__file__).parent.resolve()
CHUNKS_PATH = BASE_DIR / "outputs" / "chunks.jsonl"
FRONTEND_INDEX = BASE_DIR / "frontend" / "index.html"

# --- Chunk cache ---
_chunks = None
def get_chunks():
    global _chunks
    if _chunks is None:
        _chunks = []
        if CHUNKS_PATH.exists():
            with CHUNKS_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        _chunks.append(json.loads(line))
                    except Exception:
                        pass
    return _chunks

def simple_retrieve(question, k=6):
    tokens = [t for t in question.lower().split() if len(t) > 2]
    hits = []
    for ch in get_chunks():
        text = (ch.get("text") or "").lower()
        score = sum(1 for t in set(tokens) if t in text)
        if score > 0:
            hits.append((score, ch))
    hits.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in hits[:k]]

@app.get("/")
def root():
    if FRONTEND_INDEX.exists():
        return HTMLResponse(FRONTEND_INDEX.read_text(encoding="utf-8"))
    return {"service":"INFRATEC ISO Coach","endpoints":["/health","/ask","/_debug/chunks"],"docs":"/docs"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/_debug/chunks")
def debug_chunks():
    chunks = get_chunks()
    sample = [{"title": c.get("title"), "path": c.get("source_path")} for c in chunks[:5]]
    return {"count": len(chunks), "sample": sample, "path": str(CHUNKS_PATH)}

@app.post("/ask")
def ask(payload: dict = Body(...)):
    question = (payload.get("question") or "").strip()
    role = (payload.get("role") or "General").strip()
    if not question:
        return JSONResponse({"error": "Question is required."}, status_code=400)

    ctx = simple_retrieve(question, k=6)
    if not ctx:
        return {"answer": "No relevant INFRATEC context found. Please consult the process owner.", "citations":[]}

    # Lightweight answer from context only (no OpenAI call needed on server)
    top = ctx[0]
    answer_lines = []
    answer_lines.append("(1) " + (top.get("text","").strip()[:900] or "Context available in sources below."))
    answer_lines.append("\n(2) ISO clause references: If present in the cited documents.")
    answer_lines.append("\n(3) What to do at INFRATEC: Follow the controls and steps described in the cited document(s).")
    answer_lines.append("\n(4) Evidence list: records / logs explicitly named in the cited document(s).")
    answer = "\n".join(answer_lines)

    cites = [{"title": c.get("title") or c.get("doc_id") or "unknown", "path": c.get("source_path")} for c in ctx]
    return {"answer": answer, "citations": cites}
EOF
paste the code above)

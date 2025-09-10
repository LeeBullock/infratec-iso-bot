import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ---- Your PRESETS and ims_search() need to be defined elsewhere in this file ----
# Example stubs (replace with your actual logic)
PRESETS = {}  # { iso: { section: [ {file, clause, question}, ... ] } }
import json
import re
from pathlib import Path

IMS_INDEX_PATH = Path("data/ims_index.json")

def ims_search(query: str, k: int = 6):
    """
    Return top-k chunks from the local IMS index whose text matches the query.
    Expects ims_index.json as a list of objects with keys:
      - file (str)  full filename
      - relpath (str)  path under ManagementSystem
      - chunk (str)  text content of the chunk
    """
    if not IMS_INDEX_PATH.exists():
        return []

    try:
        docs = json.loads(IMS_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    q = query.strip().lower()
    terms = [t for t in re.split(r"\W+", q) if t]
    if not terms:
        return []

    def score(text: str) -> int:
        t = text.lower()
        return sum(t.count(term) for term in terms)

    scored = []
    for d in docs:
        chunk = d.get("chunk", "") or ""
        s = score(chunk)
        if s > 0:
            scored.append((s, {
                "file": d.get("file", ""),
                "relpath": d.get("relpath", ""),
                "chunk": chunk
            }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:k]]


# ---- FastAPI app ----
app = FastAPI(title="INFRATEC Audit Console", version="0.1.0")


# ---- Models ----
class AskIn(BaseModel):
    question: str
    session_id: str = "cli"
    top_k: int = 6


# ---- Root ----
@app.get("/")
def root():
    return {"msg": "INFRATEC Audit Console running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---- Ask endpoint ----
@app.post("/ask")
def ask(payload: AskIn):
    """
    Answers an audit question by blending:
      1) Top matches from PRESETS (ISO audit checklist rows)
      2) Top IMS excerpts from the local index (ims_search)
    Returns: {"answer": str, "sources": [ {file, section, clause}, ... ]}
    """
    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(400, "empty")
    top_k = payload.top_k

    # ---- 1) Audit checklist hits ----
    hits = []
    tokens = set(q.lower().split())
    for iso, secmap in PRESETS.items():
        for sec, rows in secmap.items():
            for r in rows:
                txt = (r.get("clause", "") + " " + r.get("question", "")).lower()
                score = len(tokens.intersection(set(txt.split())))
                if score > 0:
                    hits.append(
                        (
                            score,
                            {
                                "file": r.get("file"),
                                "section": sec,
                                "clause": r.get("clause"),
                                "question": r.get("question"),
                            },
                        )
                    )
    hits.sort(key=lambda x: x[0], reverse=True)
    audit_top = [h[1] for h in hits[:top_k]]

    # ---- 2) IMS hits ----
    ims_top = ims_search(q, k=top_k)

    # ---- 3) Build context ----
    audit_ctx = (
        "\n".join(
            f"- [AUDIT {h['file']} — {h['section']} — {h['clause']}] {h['question']}"
            for h in audit_top
        )
        or "No audit checklist matches."
    )

    ims_ctx = (
        "\n".join(
            f"- [IMS {h['relpath']}] {h['chunk'][:600]}"
            for h in ims_top
        )
        or "No IMS excerpts found."
    )

    # ---- 4) Build answer via LLM ----
    system_msg = (
        "You are INFRATEC Audit Console. Answer clearly, use bullets, "
        "and cite IMS excerpts when available. If unsure, say so."
    )
    user_msg = f"Question: {q}\n\nContext:\n{audit_ctx}\n\n{ims_ctx}"

    answer = "No LLM configured."
    if os.getenv("OPENAI_API_KEY"):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.2,
                },
                timeout=60,
            )
            data = resp.json()
            answer = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            answer = f"[LLM error: {e}]"

    # ---- 5) Sources list ----
    sources = audit_top + [
        {"file": h["file"], "section": "IMS", "clause": h["relpath"]}
        for h in ims_top
    ]

    return {"answer": answer, "sources": sources}


# ---- Index page ----
@app.get("/index")
def index():
    with open(os.path.join("frontend", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


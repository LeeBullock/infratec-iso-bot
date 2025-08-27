import os, json, re
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI(title="INFRATEC ISO Coach API", version="0.1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SYSTEM_PROMPT = open("prompts/system_prompt.txt","r",encoding="utf-8").read()
CHUNKS_PATH = "outputs/chunks.jsonl"

def load_chunks():
    chunks = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH,"r",encoding="utf-8") as f:
            for line in f:
                try:
                    chunks.append(json.loads(line))
                except Exception:
                    pass
    return chunks

def normalize_tokens(text):
    # tokens: lowercase, alnum-only, strip plurals (very simple stemming)
    toks = re.findall(r"[a-z0-9\-]+", text.lower())
    norm = set()
    for t in toks:
        norm.add(t)
        if t.endswith("es"): norm.add(t[:-2])
        if t.endswith("s"):  norm.add(t[:-1])
    return norm

def simple_retrieve(question, k=6):
    q_norm = normalize_tokens(question)
    hits = []
    for ch in load_chunks():  # reload every call so you can re-ingest without restart
        t_norm = normalize_tokens(ch.get("text",""))
        score = len(q_norm.intersection(t_norm))
        if score > 0:
            hits.append((score, ch))
    hits.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in hits[:k]]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(payload: dict = Body(...)):
    question = (payload.get("question") or "").strip()
    role = (payload.get("role") or "General").strip()
    if not question:
        return {"error": "Question is required."}

    ctx = simple_retrieve(question, k=6)
    if not ctx:
        return {"answer": "No relevant INFRATEC context found. Please consult the process owner.", "citations":[]}

    sources = []
    for ch in ctx:
        ref = f"{ch.get('title') or ch.get('doc_id') or 'Unknown'} - {ch.get('source_path')}"
        snippet = (ch.get("text",""))[:1200]
        sources.append(f"[{ref}]\n{snippet}")
    joined = "\n\n---\n\n".join(sources)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY",""))
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "text"},
            messages=[
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": f"Role: {role}\n\nQuestion: {question}\n\nUse ONLY the context below.\n\nContext:\n{joined}"}
            ]
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Generation error: {e}"

    cites = [{"title": ch.get("title"), "path": ch.get("source_path")} for ch in ctx]
    return {"answer": answer, "citations": cites}

@app.get("/")
def root():
    return {"service":"INFRATEC ISO Coach","endpoints":["/health","/ask"],"docs":"/docs"}

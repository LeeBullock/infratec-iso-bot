
from pathlib import Path
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="InfraTec ISO Bot")

# ---- Try to import your main ASGI app + helpers from asgi.py ----
asgi_mod = None
try:
    import asgi as asgi_mod
    if hasattr(asgi_mod, "app"):
        app.mount("/core", asgi_mod.app)
        print("[server] Mounted asgi.app at /core")
except Exception as e:
    print("[server] asgi import failed:", e)

build_ims_index = getattr(asgi_mod, "build_ims_index", None) if asgi_mod else None
load_ims_index  = getattr(asgi_mod, "load_ims_index",  None) if asgi_mod else None
IMS_INDEX_PATH  = getattr(asgi_mod, "IMS_INDEX_PATH",  "data/ims_index.json") if asgi_mod else "data/ims_index.json"

ask_func = None
if asgi_mod:
    for _name in ("ask", "ask_ims", "answer_question", "answer", "run_query"):
        if hasattr(asgi_mod, _name):
            ask_func = getattr(asgi_mod, _name)
            break

# ---- Static / frontend ----
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
if Path("frontend").exists():
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

def _read_index_html() -> str:
    for p in (Path("static/index.html"), Path("frontend/index.html")):
        if p.exists():
            return p.read_text(encoding="utf-8")
    return "<h1>InfraTec ISO Bot</h1><p>Place your UI at static/index.html or frontend/index.html.</p>"

# ---- Routes ----
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(_read_index_html())

@app.on_event("startup")
async def ensure_index():
    if not (build_ims_index and load_ims_index):
        print("[startup] IMS index builder not available (using runtime search only)")
        return
    try:
        needs = (not os.path.exists(IMS_INDEX_PATH)) or (os.path.getsize(IMS_INDEX_PATH) == 0)
        if needs:
            print("[startup] No IMS index — building…")
            n = build_ims_index()
            print(f"[startup] built {n} IMS chunks")
        else:
            print("[startup] IMS index present")
        load_ims_index()
    except Exception as e:
        print("[startup] IMS index error:", e)

@app.post("/admin/reindex")
def admin_reindex():
    if not (build_ims_index and load_ims_index):
        return {"status": "error", "error": "index builder not available"}
    n = build_ims_index()
    load_ims_index()
    return {"status": "ok", "chunks": n}

# Minimal /ask passthrough
from pydantic import BaseModel
class AskBody(BaseModel):
    question: str

@app.post("/ask")
def ask_endpoint(body: AskBody):
    if not ask_func:
        return {"answer": "[server] ask() is not wired from asgi.py", "sources": []}
    try:
        res = ask_func(body.question)
        if isinstance(res, dict):
            return res
        return {"answer": str(res), "sources": []}
    except Exception as e:
        return {"answer": f"[LLM error: {e}]", "sources": []}


@app.get("/_debug/ims", response_class=JSONResponse)
async def _debug_ims():
    import os
    base = os.getenv("IMS_DIR") or os.path.join(os.path.dirname(__file__), "data", "source_docs")
    files = []
    for root, dirs, filenames in os.walk(base):
        # skip macOS artifact folders / files
        dirs[:] = [d for d in dirs if not d.startswith("__MACOSX")]
        for name in filenames:
            if name.startswith("._"):
                continue
            files.append(os.path.relpath(os.path.join(root, name), base))
    return JSONResponse({"ims_dir": base, "files_found": len(files), "sample": files[:20]})

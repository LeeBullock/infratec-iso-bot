from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# Try to use the existing ask() implementation from asgi.py if available.
# If not, we'll return a friendly error from /ask instead of crashing.
try:
    from asgi import ask as ask_impl  # type: ignore
except Exception:
    ask_impl = None  # noqa: N816

app = FastAPI(title="InfraTec ISO Bot")

# ---- Static / Frontend ------------------------------------------------------

# Serve assets from ./frontend if you have any (CSS/JS). Safe if folder missing.
if Path("frontend").exists():
    app.mount("/assets", StaticFiles(directory="frontend"), name="assets")

def _load_index_html() -> str:
    """Return index.html content from ./index.html or ./frontend/index.html, else a fallback."""
    for p in (Path("index.html"), Path("frontend/index.html")):
        if p.exists():
            return p.read_text(encoding="utf-8")
    # Fallback mini page (keeps the service usable even if file is missing)
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>InfraTec ISO Bot</title></head>
<body style="font-family:system-ui;padding:2rem;max-width:900px;margin:auto">
  <h1>InfraTec ISO Bot</h1>
  <p>Type an IMS question. Answers come from your uploaded documents.</p>
  <form id="f" onsubmit="ask();return false;">
    <input id="q" style="width:100%;padding:.6rem" placeholder="Ask an IMS question…" />
    <button style="margin-top:.75rem;padding:.5rem 1rem">Ask</button>
  </form>
  <pre id="out" style="white-space:pre-wrap;background:#111;color:#eee;padding:1rem;border-radius:.5rem;margin-top:1rem"></pre>
<script>
async function ask(){
  const out=document.getElementById('out');
  const q=document.getElementById('q').value;
  out.textContent='Asking…';
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  out.textContent=await r.text();
}
</script>
</body></html>"""

@app.get("/", response_class=HTMLResponse)
async def home(_: Request) -> HTMLResponse:
    return HTMLResponse(_load_index_html())

# ---- API --------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/ask")
async def ask_api(payload: Dict[str, Any]) -> JSONResponse | PlainTextResponse:
    question = (payload or {}).get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Missing 'question'."}, status_code=400)

    if ask_impl is None:
        # asgi.ask couldn't be imported; give a helpful message but keep the app running
        return JSONResponse(
            {
                "answer": "[server error: asgi.ask() not available]",
                "sources": [],
            },
            status_code=500,
        )

    try:
        # Expect ask_impl to return a dict with 'answer' and 'sources' (your existing shape).
        result = await ask_impl(question) if callable(getattr(ask_impl, "__call__", None)) else ask_impl(question)  # type: ignore
        return JSONResponse(result if isinstance(result, dict) else {"answer": str(result), "sources": []})
    except Exception as e:  # keep service responsive even on model errors
        return JSONResponse({"answer": f"[LLM error: {e}]", "sources": []})

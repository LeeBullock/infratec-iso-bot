from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
try:
    from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles, JSONResponse, PlainTextResponse
except Exception:
    from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse

# Try to import your real app; if it fails, serve a minimal fallback app
_real_import_error = None
try:
    from asgi import app as real_app
except Exception as e:
    _real_import_error = e
    real_app = None

app = real_app if real_app is not None else FastAPI(title="infratec-iso-bot (fallback)")

# Open CORS so you can share the link
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Always provide health
@app.get("/health")
def health():
    return {"status": "ok", "mode": ("normal" if real_app else "fallback")}

# Always provide IMS debug (counts files deployed to the container)
@app.get("/_debug/ims")
def debug_ims():
    root = Path("data/source_docs")
    files = [str(p) for p in root.rglob("*") if p.is_file()]
    return {"files_found": len(files), "sample": files[:20]}

# If we are in fallback, make that obvious at /
if real_app is None:
    @app.get("/")
    def fallback_root():
        return {
            "message": "App running in FALLBACK mode (asgi.py failed to import).",
            "import_error": str(_real_import_error),
            "next_steps": [
                "Fix indentation in asgi.py, commit, and redeploy.",
                "Once fixed, Start Command keeps working without changes."
            ]
        }


@app.get('/', response_class=HTMLResponse)
def ui_home():
    return HTMLResponse(content='''<!doctype html><html><head><meta charset="utf-8">
<title>INFRATEC ISO Bot</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;max-width:900px}
h1{margin-top:0}.muted{color:#555}textarea{width:100%;height:140px}
button{padding:10px 14px;border:0;border-radius:10px;cursor:pointer}
pre{white-space:pre-wrap;background:#f6f8fa;padding:12px;border-radius:8px}
.src{font-size:14px;color:#444}</style></head>
<body>
<h1>INFRATEC ISO Bot</h1>
<p class="muted">Ask about your IMS documents. Your OpenAI key is used server-side.</p>
<textarea id="q" placeholder="e.g., Where is Emergency preparedness & response defined?"></textarea><br><br>
<button id="ask">Ask</button>
<div id="out" style="margin-top:18px"></div>
<script>
const out=document.getElementById('out');
document.getElementById('ask').onclick=async()=>{
  const q=document.getElementById('q').value.trim();
  if(!q){ out.innerHTML='<em>Please type a question.</em>'; return; }
  out.innerHTML='Asking…';
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
    const data=await r.json();
    out.innerHTML = '<h3>Answer</h3><pre>'+ (data.answer||'') +'</pre>' +
      (Array.isArray(data.sources)? '<h3>Sources</h3><ul>' +
        data.sources.map(s=>'<li class="src">'+(s.file||JSON.stringify(s))+'</li>').join('') + '</ul>' : '');
  }catch(e){ out.innerHTML='<pre>'+e+'</pre>'; }
};
</script>
</body></html>''')

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import os

# Try to import the real app from asgi.py
try:
    import asgi as _asgi
    import requests as _requests  # make sure asgi can see 'requests'
    setattr(_asgi, "requests", _requests)  # provide global name if asgi didn't import it
    app = _asgi.app
    fallback_error = None
except Exception as e:
    app = FastAPI()
    fallback_error = e

# Always mount static files for the frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root page — serves your UI if present
@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    msg = "<h1>InfraTec ISO Bot</h1><p>Add a frontend at <code>static/index.html</code>.</p>"
    if fallback_error:
        msg += f"<p><b>Running in FALLBACK mode:</b><br><pre>{fallback_error}</pre></p>"
    return HTMLResponse(msg)

# Health endpoint
@app.get("/health", response_class=JSONResponse)
def health():
    return {"status": "ok"}

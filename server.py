from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

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

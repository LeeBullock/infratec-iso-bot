from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Import the actual FastAPI app defined in asgi.py
try:
    from asgi import app  # <-- your real app lives here
except Exception as e:
    raise RuntimeError(f"Failed to import app from asgi.py: {e}")

# Open CORS for sharing; tighten later if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/_debug/ims")
def _debug_ims():
    root = Path("data/source_docs")
    files = [str(p) for p in root.rglob("*") if p.is_file()]
    return {"files_found": len(files), "sample": files[:20]}

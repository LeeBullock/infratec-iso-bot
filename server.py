from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Import your FastAPI app object named "app" from app.py or main.py
app = None
for mod in ("app", "main"):
    try:
        m = __import__(mod)
        if hasattr(m, "app"):
            app = getattr(m, "app")
            break
    except Exception:
        pass

if app is None:
    raise RuntimeError("Could not find 'app' in app.py or main.py")

# Open CORS for quick sharing (tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Debug: show how many source docs were unzipped
@app.get("/_debug/ims")
def debug_ims():
    root = Path("data/source_docs")
    files = [str(p) for p in root.rglob("*") if p.is_file()]
    return {"files_found": len(files), "sample": files[:20]}

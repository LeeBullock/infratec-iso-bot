import os
from fastapi.middleware.cors import CORSMiddleware

# Try common entry files for "app = FastAPI(...)"
app = None
for mod in ("app", "main", "asgi"):
    try:
        m = __import__(mod)
        if hasattr(m, "app"):
            app = getattr(m, "app")
            break
    except Exception:
        pass

if app is None:
    raise RuntimeError("Could not find 'app' in app.py, main.py or asgi.py")

# Open CORS for quick sharing (tighten later if you want)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check for Render
@app.get("/health")
def health():
    return {"status": "ok"}

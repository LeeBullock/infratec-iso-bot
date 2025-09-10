from fastapi.middleware.cors import CORSMiddleware

# Import your existing FastAPI app object named "app"
# Adjust if your entry file is different (app.py, main.py, asgi.py)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

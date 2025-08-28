# asgi.py — tiny wrapper so Render/uvicorn can always import `app`
from app import app  # exposes FastAPI instance as `app`

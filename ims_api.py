
from fastapi import APIRouter
import os, threading, importlib

router = APIRouter()

_IMS_RUNNING = False
_IMS_LAST_ERROR = None
_IMS_FILES_SEEN = 0

def _extractor():
    asgi_mod = importlib.import_module("asgi")
    base = getattr(asgi_mod, "IMS_DIR", os.path.join("data", "source_docs"))
    extract = getattr(asgi_mod, "extract_text_from_file", None) or getattr(asgi_mod, "extract_text", None)
    if extract is None:
        raise RuntimeError("No text extractor available in asgi.py")
    return asgi_mod, base, extract

def _worker():
    global _IMS_RUNNING, _IMS_LAST_ERROR, _IMS_FILES_SEEN
    _IMS_RUNNING = True
    _IMS_LAST_ERROR = None
    _IMS_FILES_SEEN = 0
    try:
        asgi_mod, base, extract = _extractor()
        index = []
        for root, dirs, files in os.walk(base):
            if "__MACOSX" in root:
                continue
            for fn in files:
                _IMS_FILES_SEEN += 1
                ext = os.path.splitext(fn)[1].lower()
                if ext not in (".txt", ".md", ".pdf", ".docx", ".xlsx", ".xlsm", ".xltx", ".xltm"):
                    continue
                path = os.path.join(root, fn)
                try:
                    text = extract(path)
                except Exception:
                    text = ""
                if not text:
                    continue
                rel = os.path.relpath(path, base)
                for i in range(0, len(text), 1200):
                    chunk = text[i:i+1200]
                    if chunk.strip():
                        index.append({"relpath": rel, "text": chunk})
        setattr(asgi_mod, "IMS_INDEX", index)
    except Exception as e:
        _IMS_LAST_ERROR = str(e)
    finally:
        _IMS_RUNNING = False

@router.post("/ims/reindex")
def ims_reindex():
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return {"ok": True, "started": True}

@router.get("/ims/status")
def ims_status():
    try:
        asgi_mod = importlib.import_module("asgi")
        idx = getattr(asgi_mod, "IMS_INDEX", None)
        chunks = len(idx) if isinstance(idx, list) else 0
    except Exception:
        chunks = 0
    return {
        "running": _IMS_RUNNING,
        "last_error": _IMS_LAST_ERROR,
        "files_seen": _IMS_FILES_SEEN,
        "chunks": chunks,
    }

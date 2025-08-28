# scripts/ingest.py
# Recursively ingest PDFs, DOCX, and TXT under data/source_docs/*
# Writes JSONL chunks to outputs/chunks.jsonl
import os, sys, json, re, argparse
from pathlib import Path

# Optional deps: PyMuPDF (fitz) and python-docx
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from docx import Document
except Exception:
    Document = None

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "source_docs"
CATALOG_CSV = ROOT / "data" / "metadata" / "docs_catalog.csv"
OUT_PATH = ROOT / "outputs" / "chunks.jsonl"

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
ALLOWED_EXTS = {".docx", ".txt"}

def read_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()

    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".docx":
        if Document is None:
            print(f"[WARN] python-docx not installed; skipping {path}", file=sys.stderr)
            return ""
        try:
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            print(f"[WARN] DOCX read failed for {path}: {e}", file=sys.stderr)
            return ""

    if ext == ".pdf":
        if fitz is None:
            print(f"[WARN] PyMuPDF not installed; skipping {path}", file=sys.stderr)
            return ""
        try:
            text_parts = []
            with fitz.open(str(path)) as pdf:
                for page in pdf:
                    text_parts.append(page.get_text("text"))
            return "\n".join(text_parts)
        except Exception as e:
            print(f"[WARN] PDF read failed for {path}: {e}", file=sys.stderr)
            return ""

    return ""  # unsupported types (e.g., .xls/.ppt) are skipped

def clean_text(t: str) -> str:
    # Collapse very long whitespace, normalise newlines
    t = t.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_text(t: str, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    t = t.strip()
    if not t:
        return []
    chunks = []
    start = 0
    n = len(t)
    while start < n:
        end = min(start + size, n)
        # try to end on a sentence boundary if possible
        slice_ = t[start:end]
        if end < n:
            m = re.search(r"[.?!]\s+\S+$", slice_)
            if m:
                end = start + m.end()
                slice_ = t[start:end]
        chunks.append(slice_)
        start = max(end - overlap, 0)
        if start == end:
            start = end + 1
    return chunks

def load_catalog():
    """
    Optional metadata: maps path -> dict of {doc_id, title, ...}
    """
    if not CATALOG_CSV.exists():
        return {}
    import csv
    out = {}
    with open(CATALOG_CSV, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            # normalise path separators for matching
            p = (row.get("path_hint") or "").replace("\\", "/")
            out[p] = row
    return out

def guess_title_from_filename(path: Path) -> str:
    name = path.name
    name = re.sub(r"[_-]+", " ", name)
    return name

def main():
    parser = argparse.ArgumentParser(description="Recursive ingester for INFRATEC ISO Coach")
    parser.add_argument("--root", default=str(SOURCE_ROOT), help="Root folder to scan (default: data/source_docs)")
    args = parser.parse_args()

    scan_root = Path(args.root)
    if not scan_root.exists():
        print(f"[ERR] Missing source root: {scan_root}", file=sys.stderr)
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

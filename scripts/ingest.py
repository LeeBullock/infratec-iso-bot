#!/usr/bin/env python3
import os, sys, json, argparse, pathlib
from typing import List

# Optional DOCX support (won't crash if not installed)
try:
    from docx import Document
except Exception:
    Document = None

OUT_PATH = "outputs/chunks.jsonl"

def log(msg: str):
    print(msg, flush=True)

def read_txt(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log(f"[SKIP][TXT] {path} ({e})")
        return ""

def read_docx(path: pathlib.Path) -> str:
    if Document is None:
        log("[WARN] python-docx not installed; cannot read DOCX.")
        return ""
    try:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        log(f"[SKIP][DOCX] {path} ({e})")
        return ""

def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    """Simple word-safe chunking to avoid huge records."""
    text = text.replace("\r\n", "\n")
    parts = []
    buf = []
    size = 0
    for line in text.split("\n"):
        if size + len(line) + 1 > max_chars and buf:
            parts.append("\n".join(buf).strip())
            buf = [line]
            size = len(line) + 1
        else:
            buf.append(line)
            size += len(line) + 1
    if buf:
        parts.append("\n".join(buf).strip())
    # Remove empties
    return [p for p in parts if p.strip()]

def iter_files(root: pathlib.Path, allowed_exts: List[str]):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in allowed_exts:
            yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/source_docs", help="Folder to scan (recursive).")
    ap.add_argument("--ext", default="txt,docx", help="Comma-separated list (e.g. txt,docx,pdf)")
    ap.add_argument("--append", action="store_true", help="Append to existing outputs/chunks.jsonl")
    ap.add_argument("--max-files", type=int, default=0, help="Limit number of files (0 = no limit)")
    ap.add_argument("--min-chars", type=int, default=40, help="Skip chunks smaller than this")
    args = ap.parse_args()

    ROOT = pathlib.Path(args.root).resolve()
    allowed_exts = ["."+e.strip().lower() for e in args.ext.split(",") if e.strip()]
    log(f"[INGEST] root={ROOT} ext={allowed_exts} max_files={args.max_files or '∞'} append={args.append} min_chars={args.min_chars}")

    if not ROOT.exists():
        log(f"[ERR] Root does not exist: {ROOT}")
        sys.exit(1)

    out_mode = "a" if args.append and os.path.exists(OUT_PATH) else "w"
    pathlib.Path("outputs").mkdir(parents=True, exist_ok=True)
    out = open(OUT_PATH, out_mode, encoding="utf-8")

    count_files = 0
    count_chunks = 0

    try:
        for p in iter_files(ROOT, allowed_exts):
            # Respect max-files
            if args.max_files and count_files >= args.max_files:
                log("[STOP] max_files reached")
                break

            rel = p.relative_to(pathlib.Path.cwd()) if str(ROOT) in str(pathlib.Path.cwd()) else p
            log(f"[READ] {rel}")

            text = ""
            ext = p.suffix.lower()
            if ext == ".txt":
                text = read_txt(p)
            elif ext == ".docx":
                text = read_docx(p)
            else:
                # PDFs intentionally ignored in this script (we’ll add later with tesseract/poppler)
                continue

            if not text or len(text.strip()) < args.min_chars:
                log(f"[SKIP] too little text: {rel}")
                count_files += 1
                continue

            chunks = chunk_text(text, max_chars=900)
            for i, ch in enumerate(chunks):
                if len(ch) < args.min_chars:
                    continue
                rec = {
                    "chunk_id": f"{p.name}::chunk{i:03d}",
                    "text": ch,
                    "source_path": str(p).replace("\\", "/"),
                    "title": p.name,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count_chunks += 1

            log(f"[OK] wrote {len(chunks)} chunks from {rel}")
            count_files += 1

    finally:
        out.close()

    log(f"[OK] wrote {count_chunks} chunks -> {OUT_PATH}")

if __name__ == "__main__":
    main()

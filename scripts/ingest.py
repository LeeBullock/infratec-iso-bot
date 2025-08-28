# scripts/ingest.py (drop-in)
import os, sys, json, re, argparse
from pathlib import Path

try:
    import fitz  # PyMuPDF for PDFs
except Exception:
    fitz = None

try:
    from docx import Document
except Exception:
    Document = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "source_docs"
OUT_PATH = ROOT / "outputs" / "chunks.jsonl"

def read_txt(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def read_docx(p: Path) -> str:
    if Document is None:
        return ""
    try:
        d = Document(str(p))
        return "\n".join(par.text for par in d.paragraphs)
    except Exception:
        return ""

def read_pdf(p: Path, max_pages: int | None) -> str:
    if fitz is None:
        return ""
    text_parts = []
    try:
        with fitz.open(str(p)) as pdf:
            n = len(pdf)
            limit = min(n, max_pages) if max_pages else n
            for i in range(limit):
                text_parts.append(pdf[i].get_text("text"))
        return "\n".join(text_parts)
    except Exception:
        return ""

def clean_text(t: str) -> str:
    t = t.replace("\r","\n")
    t = re.sub(r"[ \t]+"," ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_text(t: str, size=1200, overlap=150):
    if not t: return []
    out, start, n = [], 0, len(t)
    while start < n:
        end = min(start+size, n)
        slice_ = t[start:end]
        if end < n:
            m = re.search(r"[.?!]\s+\S+$", slice_)
            if m:
                end = start + m.end()
                slice_ = t[start:end]
        out.append(slice_)
        start = max(end - overlap, 0)
        if start == end: start = end + 1
    return out

def main():
    ap = argparse.ArgumentParser("INGEST")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--ext", default="txt,docx", help="comma list of ext to ingest (e.g. txt,docx,pdf)")
    ap.add_argument("--max-pages", type=int, default=0, help="limit pages per PDF (0 = all)")
    ap.add_argument("--max-files", type=int, default=0, help="limit number of files (0 = all)")
    args = ap.parse_args()

    allowed = {"."+e.strip().lower() for e in args.ext.split(",") if e.strip()}
    max_pages = args.max_pages if args.max_pages and args.max_pages > 0 else None
    max_files = args.max_files if args.max_files and args.max_files > 0 else None

    scan_root = Path(args.root)
    if not scan_root.exists():
        print(f"[ERR] Source root not found: {scan_root}", file=sys.stderr); sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written, files_seen = 0, 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        print(f"[INGEST] root={scan_root} ext={sorted(allowed)} max_pages={max_pages} max_files={max_files}")
        for dirpath, _, files in os.walk(scan_root):
            for fname in files:
                p = Path(dirpath) / fname
                ext = p.suffix.lower()
                if ext not in allowed:
                    continue
                files_seen += 1
                if max_files and files_seen > max_files:
                    print(f"[STOP] max_files reached ({max_files})")
                    break

                rel = str(p.relative_to(ROOT)).replace("\\","/")
                try:
                    if ext == ".txt":
                        text = read_txt(p)
                    elif ext == ".docx":
                        text = read_docx(p)
                    elif ext == ".pdf":
                        text = read_pdf(p, max_pages)
                    else:
                        text = ""
                except Exception as e:
                    print(f"[SKIP] {rel}: read error {e}", file=sys.stderr)
                    continue

                text = clean_text(text)
                if not text:
                    print(f"[SKIP] {rel}: empty/unsupported")
                    continue

                chunks = chunk_text(text)
                for i, ch in enumerate(chunks):
                    rec = {
                        "chunk_id": f"{p.name}::chunk{i:03d}",
                        "text": ch,
                        "source_path": rel,
                        "title": p.name,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1

        print(f"[OK] wrote {written} chunks -> {OUT_PATH}")

if __name__ == "__main__":
    main()


from pathlib import Path

PDF_DIR = Path("data/source_docs")
INDEX_DIR = Path("data/index_store")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def build_index():
    docs = []
    for ext in (".pdf", ".docx", ".xlsx"):
        docs += list(PDF_DIR.rglob(f"*{ext}"))
    print(f"[preindex] Found {len(docs)} docs under {PDF_DIR}")
    if not docs:
        print("[preindex] WARNING: no documents found to index.")
    # TODO: call your real indexer here if needed
    # from ims.indexer import index_all
    # index_all([str(p) for p in docs], str(INDEX_DIR))

if not any(INDEX_DIR.iterdir()):
    print("[preindex] No index found — building…")
    build_index()
else:
    print("[preindex] Index exists — skipping.")

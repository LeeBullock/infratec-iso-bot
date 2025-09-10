from pathlib import Path

PDF_DIR = Path("data/source_docs")
INDEX_DIR = Path("data/index_store")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def build_index():
    # TODO: replace with your real indexing function
    # e.g. from ims.indexer import index_all
    # index_all(str(PDF_DIR), str(INDEX_DIR))
    print(f"[preindex] Would index PDFs from {PDF_DIR} into {INDEX_DIR}")

if not any(INDEX_DIR.iterdir()):
    print("[preindex] No index found — building…")
    build_index()
else:
    print("[preindex] Index exists — skipping.")

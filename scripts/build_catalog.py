# scripts/build_catalog.py
# Scans data/source_docs recursively and creates data/metadata/docs_catalog.csv
import csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "source_docs"
OUT = ROOT / "data" / "metadata" / "docs_catalog.csv"

def guess_doc_id(path: Path) -> str:
    # Try to pull a code like QMS-P-012 from the filename
    m = re.search(r"([A-Z]{2,5}-[A-Z]{1,3}-?\d{2,4})", path.name.upper())
    return m.group(1) if m else path.stem

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in SOURCE_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".pdf", ".docx", ".txt"}:
            continue
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        rows.append({
            "doc_id": guess_doc_id(p),
            "title": p.stem,
            "doc_type": "",         # fill in later if you want (procedure, work instruction, form, etc.)
            "owner": "",            # e.g., QA Manager
            "rev": "",
            "effective_date": "",
            "confidentiality": "Internal",
            "path_hint": rel,       # used by ingest to attach metadata
            "standard_hint": "",    # e.g., ISO9001:2015 / ISO14001:2015 / ISO45001:2018
        })

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["doc_id","title","doc_type","owner","rev","effective_date","confidentiality","path_hint","standard_hint"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] Wrote {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    main()

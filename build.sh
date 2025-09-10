#!/usr/bin/env bash
set -euo pipefail

echo "[build.sh] Installing requirements..."
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir gdown

ZIP_PATH="/tmp/docs.zip"
echo "[build.sh] Downloading from Google Drive (gdown --fuzzy)..."
gdown --fuzzy "${PDF_PACKAGE_URL}" -O "${ZIP_PATH}"

echo "[build.sh] Validating zip..."
unzip -t "${ZIP_PATH}" >/dev/null

echo "[build.sh] Unzipping docs..."
mkdir -p data/source_docs
unzip -o "${ZIP_PATH}" -d data/source_docs

# Flatten common nested folder
if [ -d "data/source_docs/ManagementSystem" ]; then
  mv data/source_docs/ManagementSystem/* data/source_docs/ || true
  rmdir data/source_docs/ManagementSystem || true
fi

# --- NEW: cleanup & exclusion of noisy content ---
echo "[build.sh] Cleaning Mac cruft..."
find data/source_docs -name "__MACOSX" -type d -prune -exec rm -rf {} + || true
find data/source_docs -name ".DS_Store" -type f -delete || true

echo "[build.sh] Removing audit Excel checklists from index set..."
# adjust pattern(s) as needed; this keeps them in Drive but excludes from embedding
find data/source_docs -type f -iname "*ISO*-checklist*.xlsx" -delete || true

echo "[build.sh] Listing a few files to confirm:"
find data/source_docs -type f | head -n 20 || true
# -----------------------------------------------

echo "[build.sh] Preindexing…"
python scripts/preindex.py || true

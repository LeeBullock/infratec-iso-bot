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

echo "[build.sh] Listing a few files to confirm:"
find data/source_docs -type f | head -n 20 || true

echo "[build.sh] Preindexing..."
python scripts/preindex.py || true

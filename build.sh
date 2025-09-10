#!/usr/bin/env bash
set -e

echo "[build.sh] Installing requirements..."
pip install --no-cache-dir -r requirements.txt

# gdown reliably downloads big Google Drive files
echo "[build.sh] Installing gdown..."
python -m pip install --no-cache-dir gdown

FILE_ID="${PDF_FILE_ID:-}"
if [ -z "$FILE_ID" ]; then
  echo "[build.sh] ERROR: PDF_FILE_ID env var is not set"; exit 1
fi

echo "[build.sh] Downloading Drive file id=$FILE_ID with gdown..."
python -m gdown "$FILE_ID" -O /tmp/docs.zip --fuzzy

echo "[build.sh] Verifying file..."
ls -lh /tmp/docs.zip || true
FILESIZE=$(wc -c </tmp/docs.zip || echo 0)
if [ "$FILESIZE" -lt 1000000 ]; then
  echo "[build.sh] ERROR: Downloaded file is too small ($FILESIZE bytes)"; exit 2
fi

echo "[build.sh] Unzipping docs..."
mkdir -p data/source_docs
unzip -o /tmp/docs.zip -d data/source_docs

# Flatten common top-level folder (e.g. 'ManagementSystem')
if [ -d "data/source_docs/ManagementSystem" ]; then
  mv data/source_docs/ManagementSystem/* data/source_docs/ || true
  rmdir data/source_docs/ManagementSystem || true
fi

echo "[build.sh] Listing a few files..."
find data/source_docs -maxdepth 2 -type f | head -n 15

echo "[build.sh] Building index..."
python scripts/preindex.py || true

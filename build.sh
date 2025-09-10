#!/usr/bin/env bash
set -e

echo "[build.sh] Installing requirements..."
pip install --no-cache-dir -r requirements.txt

echo "[build.sh] Downloading docs from Google Drive..."
curl -L "$PDF_PACKAGE_URL" -o /tmp/docs.zip

echo "[build.sh] Unzipping docs..."
mkdir -p data/source_docs
unzip -o /tmp/docs.zip -d data/source_docs

# Move PDFs up if they’re inside a subfolder like ManagementSystem
if [ -d "data/source_docs/ManagementSystem" ]; then
  mv data/source_docs/ManagementSystem/* data/source_docs/ || true
  rmdir data/source_docs/ManagementSystem || true
fi

echo "[build.sh] Building index..."
python scripts/preindex.py || true

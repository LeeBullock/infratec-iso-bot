#!/usr/bin/env bash
set -euo pipefail

echo "[build.sh] Installing requirements..."
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir gdown

# If docs already exist in the repo, skip Google Drive entirely
if [ -d "data/source_docs" ] && [ "$(find data/source_docs -type f | wc -l)" -gt 100 ]; then
  echo "[build.sh] data/source_docs already present; skipping download."
else
  if [ -n "${PDF_PACKAGE_URL:-}" ]; then
    echo "[build.sh] Downloading from Google Drive (gdown --fuzzy)..."
    gdown --fuzzy "${PDF_PACKAGE_URL}" -O /tmp/docs.zip

    echo "[build.sh] Unzipping docs..."
    mkdir -p data/source_docs
    unzip -o /tmp/docs.zip -d data/source_docs

    # Flatten common nested folder
    if [ -d "data/source_docs/ManagementSystem" ]; then
      mv data/source_docs/ManagementSystem/* data/source_docs/ || true
      rmdir data/source_docs/ManagementSystem || true
    fi
  else
    echo "[build.sh] WARNING: PDF_PACKAGE_URL not set and no local docs; continuing with empty docs."
    mkdir -p data/source_docs
  fi
fi

echo "[build.sh] Preindexing…"
python scripts/preindex.py || true

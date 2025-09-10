#!/usr/bin/env bash
set -euo pipefail

echo "[build.sh] Installing requirements..."
pip install --no-cache-dir -r requirements.txt

ZIP_PATH="/tmp/docs.zip"

download_with_curl() {
  echo "[build.sh] Downloading via curl from PDF_PACKAGE_URL..."
  curl -L "${PDF_PACKAGE_URL}" -o "${ZIP_PATH}"
}

download_with_gdown() {
  echo "[build.sh] Downloading via gdown (fuzzy URL)..."
  pip install --no-cache-dir gdown
  gdown --fuzzy "${PDF_PACKAGE_URL}" -O "${ZIP_PATH}"
}

validate_zip() {
  unzip -t "${ZIP_PATH}" >/dev/null 2>&1
}

echo "[build.sh] Attempt 1: curl"
download_with_curl || true
if ! validate_zip; then
  echo "[build.sh] curl produced an invalid zip; switching to gdown"
  download_with_gdown
  if ! validate_zip; then
    echo "[build.sh] ERROR: Still not a valid zip after gdown." >&2
    exit 1
  fi
fi

echo "[build.sh] Unzipping docs..."
mkdir -p data/source_docs
unzip -o "${ZIP_PATH}" -d data/source_docs

# If there is a nested folder like ManagementSystem/, move files up
if [ -d "data/source_docs/ManagementSystem" ]; then
  mv data/source_docs/ManagementSystem/* data/source_docs/ || true
  rmdir data/source_docs/ManagementSystem || true
fi

echo "[build.sh] Listing a few files to confirm:"
find data/source_docs -type f | head -n 20 || true

echo "[build.sh] Building index..."
python scripts/preindex.py || true

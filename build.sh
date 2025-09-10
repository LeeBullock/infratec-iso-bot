#!/usr/bin/env bash
set -euo pipefail

echo "[build.sh] Installing requirements..."
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir gdown autopep8

ZIP_PATH="/tmp/docs.zip"

echo "[build.sh] Downloading from Google Drive (gdown --fuzzy)…"
gdown --fuzzy "${PDF_PACKAGE_URL}" -O "${ZIP_PATH}"

echo "[build.sh] File info:"
file "${ZIP_PATH}" || true
echo -n "[build.sh] Size: "; wc -c "${ZIP_PATH}" || true

echo "[build.sh] ZIP contents (top 100):"
unzip -l "${ZIP_PATH}" | head -n 100 || true

echo "[build.sh] Validating zip…"
unzip -t "${ZIP_PATH}" >/dev/null

echo "[build.sh] Unzipping…"
mkdir -p data/source_docs
unzip -o "${ZIP_PATH}" -d data/source_docs

# If there's a single top-level folder, flatten it
ROOT="data/source_docs"
top_dirs_count=$(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
top_files_count=$(find "$ROOT" -mindepth 1 -maxdepth 1 -type f | wc -l | tr -d ' ')
if [ "$top_files_count" = "0" ] && [ "$top_dirs_count" = "1" ]; then
  dir_to_flatten=$(find "$ROOT" -mindepth 1 -maxdepth 1 -type d)
  echo "[build.sh] Flattening: $dir_to_flatten -> $ROOT"
  find "$dir_to_flatten" -mindepth 1 -maxdepth 1 -exec mv -t "$ROOT" {} +
  rmdir "$dir_to_flatten" || true
fi

echo "[build.sh] Final file listing (top 50):"
find data/source_docs -type f | head -n 50 || true

# ---- FIX asgi.py indentation (tabs→spaces + autopep8) ----
if [ -f asgi.py ]; then
  echo "[build.sh] Normalizing asgi.py indentation..."
  python - <<'PY'
p='asgi.py'
s=open(p,'r',encoding='utf-8',errors='replace').read()
s=s.replace('\t','    ').replace('\r\n','\n').replace('\r','\n')
open(p,'w',encoding='utf-8',newline='\n').write(s)
print('[build] Replaced TABs and normalized newlines in asgi.py')
PY
  autopep8 --in-place --aggressive --aggressive asgi.py || true
  python -m py_compile asgi.py
fi

echo "[build.sh] Preindexing…"
python scripts/preindex.py || true

#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found on PATH - install it (brew/apt install ffmpeg)"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/pip install -q --upgrade yt-dlp || true   # YouTube changes often; keep yt-dlp fresh
exec .venv/bin/streamlit run app.py --server.headless false

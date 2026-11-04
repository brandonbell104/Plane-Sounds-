#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found on PATH - install it (brew/apt install ffmpeg)"
# skip Streamlit's one-time "enter your email" welcome prompt (the email is optional)
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
  mkdir -p "$HOME/.streamlit"
  printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/pip install -q --upgrade yt-dlp || true   # YouTube changes often; keep yt-dlp fresh
.venv/bin/pip install -q -r requirements.txt || true  # picks up newly added dependencies
exec .venv/bin/streamlit run app.py --server.headless false

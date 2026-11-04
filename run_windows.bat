@echo off
cd /d "%~dp0"
where ffmpeg >nul 2>nul || echo WARNING: ffmpeg not found on PATH. Install it with:  winget install ffmpeg
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt
)
.venv\Scripts\pip install -q --upgrade yt-dlp
.venv\Scripts\streamlit run app.py --server.headless false
pause

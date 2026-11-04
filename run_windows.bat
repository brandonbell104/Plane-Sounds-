@echo off
cd /d "%~dp0"
where ffmpeg >nul 2>nul || echo WARNING: ffmpeg not found on PATH. Install it with:  winget install ffmpeg
rem Skip Streamlit's one-time "enter your email" welcome prompt (the email is optional)
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt
)
.venv\Scripts\pip install -q --upgrade yt-dlp
.venv\Scripts\pip install -q -r requirements.txt
.venv\Scripts\streamlit run app.py --server.headless false
pause

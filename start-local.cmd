@echo off
setlocal
cd /d "%~dp0"
git branch --show-current | findstr /I /X "ZXC" >nul
if errorlevel 1 (
    echo Switch this working copy to ZXC before starting the preview.
    goto failed
)
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\run_local.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Local preview could not start. See the error above. Python 3.12 and Git are required.
pause
exit /b 1

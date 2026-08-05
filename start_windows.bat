@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating local Python environment...
  py -3 -m venv .venv || goto :error
  .venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
  .venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch || goto :error
  .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
)
.venv\Scripts\python.exe scripts\verify_assets.py || goto :error
.venv\Scripts\python.exe launch.py %*
goto :eof
:error
echo.
echo Installation or launch failed. Review the messages above.
pause
exit /b 1

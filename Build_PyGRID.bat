@echo off
setlocal

cd /d C:\Mac\Home\Github\PyGRID
if errorlevel 1 goto :error

echo.
echo === Activating PyGRID virtual environment ===
call .venv\Scripts\activate.bat
if errorlevel 1 goto :error

echo.
echo === Installing current PyGRID source ===
python -m pip install -e .
if errorlevel 1 goto :error

echo.
echo === Building PyGRID.exe ===
python -m PyInstaller --onedir --contents-directory . --name PyGRID --clean --noconfirm --collect-submodules pygrid pygrid_launcher.py
if errorlevel 1 goto :error

echo.
echo ==========================================
echo PyGRID build completed successfully.
echo EXE:
echo C:\Mac\Home\Github\PyGRID\dist\PyGRID\PyGRID.exe
echo ==========================================
echo.
pause
exit /b 0

:error
echo.
echo ==========================================
echo ERROR: PyGRID build failed.
echo ==========================================
echo.
pause
exit /b 1

@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PORT=8600"
set "URL=http://localhost:%PORT%"

echo ============================================
echo  OW Highlight Agent - local UI launcher
echo ============================================
echo.

echo [1/3] Checking whether the UI is already running at %URL% ...
set "HEALTH="
for /f "delims=" %%H in ('curl -s --noproxy "*" --max-time 3 "%URL%/_stcore/health" 2^>nul') do set "HEALTH=%%H"
if /i "!HEALTH!"=="ok" (
    echo       Already running. Opening the page in your browser...
    start "" "%URL%"
    goto :done
)
echo       Not running, will start it now.

echo [2/3] Looking for a Python that has Streamlit installed...
set "PYEXE="
for %%C in (
    "%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe"
    "%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.14\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
) do (
    if not defined PYEXE if exist %%C (
        %%C -c "import streamlit" >nul 2>nul
        if !errorlevel! equ 0 set "PYEXE=%%~C"
    )
)
if not defined PYEXE (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        "%%P" -c "import streamlit" >nul 2>nul
        if !errorlevel! equ 0 set "PYEXE=%%P"
    )
)
if not defined PYEXE (
    for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
        "%%P" -c "import streamlit" >nul 2>nul
        if !errorlevel! equ 0 set "PYEXE=%%P"
    )
)

if not defined PYEXE (
    echo.
    echo [ERROR] No Python interpreter with Streamlit was found.
    echo         Fix: open a terminal in this folder and run:
    echo              py -3 -m pip install -r requirements.txt
    echo         then double-click this file again.
    echo.
    goto :fail
)
echo       Using: !PYEXE!

echo [3/3] Starting Streamlit. KEEP THIS WINDOW OPEN; press Ctrl+C to stop.
echo       The browser should open automatically. If not, visit %URL%
echo.
"!PYEXE!" -m streamlit run app.py --server.port %PORT% --server.headless false
echo.
echo Streamlit stopped with exit code !errorlevel!.
goto :fail

:done
if /i not "%~1"=="--no-pause" (
    echo.
    echo Done. You can close this window.
    pause >nul
)
exit /b 0

:fail
if /i not "%~1"=="--no-pause" pause
exit /b 1

@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ================================================
echo  OW Highlight Agent - Desktop GUI launcher
echo ================================================
echo.

echo [1/2] Looking for a Python with BOTH tkinter and numpy...
set "PYEXE="
set "CANDIDATES=%LOCALAPPDATA%\Programs\Python\Python314\python.exe %LOCALAPPDATA%\Programs\Python\Python313\python.exe %LOCALAPPDATA%\Programs\Python\Python312\python.exe %USERPROFILE%\.workbuddy\binaries\python\versions\3.13.14\python.exe %USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe"

for %%C in (%CANDIDATES%) do (
    if not defined PYEXE if exist "%%C" (
        "%%C" -c "import tkinter, numpy" >nul 2>nul
        if !errorlevel! equ 0 set "PYEXE=%%C"
    )
)
if not defined PYEXE (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        if not defined PYEXE "%%P" -c "import tkinter, numpy" >nul 2>nul && set "PYEXE=%%P"
    )
)
if not defined PYEXE (
    for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
        if not defined PYEXE "%%P" -c "import tkinter, numpy" >nul 2>nul && set "PYEXE=%%P"
    )
)

if not defined PYEXE (
    echo.
    echo [ERROR] No suitable Python interpreter found.
    echo         The desktop GUI needs Python with tkinter AND numpy.
    echo         tkinter ships with the official python.org installer by default.
    echo         Fix: open a terminal in this folder and run
    echo              py -3 -m pip install numpy
    echo         then double-click this file again.
    echo.
    goto :fail
)
echo       Using: !PYEXE!

echo [2/2] Starting the desktop window (no console needed)...
set "PYW=!PYEXE:python.exe=pythonw.exe!"
if not exist "!PYW!" set "PYW=!PYEXE!"
start "" "!PYW!" gui.py
if !errorlevel! neq 0 goto :fail

if /i not "%~1"=="--no-pause" (
    echo.
    echo Done. The GUI window should appear in 1-3 seconds.
    echo You can close this console window now.
    timeout /t 2 >nul
)
exit /b 0

:fail
if /i not "%~1"=="--no-pause" pause
exit /b 1

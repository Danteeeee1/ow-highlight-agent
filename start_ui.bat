@echo off
setlocal
cd /d "%~dp0"

echo The local UI now opens as a desktop window (no browser / no server needed).
echo Falling through to start_gui.bat ...
echo.

call "%~dp0start_gui.bat" %*
exit /b %errorlevel%

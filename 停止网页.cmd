@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_showcase.ps1"
if errorlevel 1 (
    echo 停止失败，请查看上面的提示。
    pause
    exit /b 1
)
echo 网页服务已检查。可关闭此窗口。
pause

@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在检查 Python、安装项目依赖并启动网页。首次安装需要联网，请不要关闭窗口。
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_and_start.ps1"
set "result=%errorlevel%"
if not "%result%"=="0" (
    echo 安装或启动失败。请查看上方提示，处理后重新双击此文件。
    pause
)
exit /b %result%

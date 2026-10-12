@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在检查运行环境并打开网页。已安装的健康环境不会重复下载依赖。
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_and_start.ps1"
set "result=%errorlevel%"
if not "%result%"=="0" (
    echo 启动失败。请查看上方提示，处理后重新双击此文件。
    pause
)
exit /b %result%

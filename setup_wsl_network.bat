@echo off
echo 🌐 设置WSL端口转发 - Chain-of-Zoom界面访问
echo ================================================

echo 📋 正在配置端口转发...
echo.

REM 删除现有的端口转发规则（如果存在）
netsh interface portproxy delete v4tov4 listenport=7861 2>nul
netsh interface portproxy delete v4tov4 listenport=7862 2>nul

REM 添加新的端口转发规则
echo 🔧 添加端口7861转发规则...
netsh interface portproxy add v4tov4 listenport=7861 listenaddress=0.0.0.0 connectport=7861 connectaddress=127.0.0.1

echo 🔧 添加端口7862转发规则（备用）...
netsh interface portproxy add v4tov4 listenport=7862 listenaddress=0.0.0.0 connectport=7862 connectaddress=127.0.0.1

echo.
echo ✅ 端口转发配置完成！
echo.
echo 📖 现在你可以通过以下地址访问界面：
echo    - http://localhost:7861 （主端口）
echo    - http://localhost:7862 （备用端口）
echo    - http://127.0.0.1:7861
echo    - http://127.0.0.1:7862
echo.
echo 🔍 查看当前端口转发规则：
netsh interface portproxy show all

echo.
echo 📝 如需删除端口转发，运行：
echo    netsh interface portproxy delete v4tov4 listenport=7861
echo    netsh interface portproxy delete v4tov4 listenport=7862
echo.
pause 
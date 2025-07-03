@echo off
REM -*- coding: gb2312 -*-
REM Windows启动脚本

echo ========================================
echo FreeFEM轧制应力场分析系统
echo ========================================
echo.

REM 设置编码
chcp 936 > nul

REM 检查Python
echo 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

REM 检查FreeFEM
echo 检查FreeFEM安装...
where FreeFem++ >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: 未找到FreeFEM，请确保已安装并添加到PATH
    echo 尝试使用默认路径...
    set FREEFEM_PATH=C:\Program Files\FreeFem++\FreeFem++.exe
) else (
    for /f "delims=" %%i in ('where FreeFem++') do set FREEFEM_PATH=%%i
)

REM 创建虚拟环境（如果不存在）
if not exist "venv" (
    echo 创建Python虚拟环境...
    python -m venv venv
)

REM 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat

REM 安装依赖
echo 检查并安装依赖...
pip install -r requirements.txt -q

REM 创建必要的目录
echo 创建工作目录...
if not exist "work" mkdir work
if not exist "vtk_output" mkdir vtk_output
if not exist "simulations" mkdir simulations
if not exist "logs" mkdir logs

REM 检查端口占用
echo 检查端口占用...
netstat -an | find "8000" >nul
if %errorlevel% equ 0 (
    echo 警告: 端口8000已被占用
    echo 是否继续？(Y/N)
    set /p continue=
    if /i "%continue%" neq "Y" exit /b 1
)

REM 设置环境变量
set PYTHONIOENCODING=gb2312
set FREEFEM_EXECUTABLE=%FREEFEM_PATH%

REM 启动服务
echo.
echo 启动FreeFEM API服务...
echo 访问地址: http://localhost:8000
echo 静态页面: http://localhost:8000/static/index.html
echo API文档: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 停止服务
echo ========================================

REM 启动FastAPI
python -m uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload

pause
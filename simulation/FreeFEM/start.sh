#!/bin/bash
# -*- coding: gb2312 -*-
# Linux/Mac启动脚本

echo "========================================"
echo "FreeFEM轧制应力场分析系统"
echo "========================================"
echo ""

# 设置编码
export LANG=zh_CN.GB2312
export LC_ALL=zh_CN.GB2312
export PYTHONIOENCODING=gb2312

# 检查Python
echo "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.8或更高版本"
    exit 1
fi

# 检查FreeFEM
echo "检查FreeFEM安装..."
if command -v FreeFem++ &> /dev/null; then
    FREEFEM_PATH=$(which FreeFem++)
elif command -v freefem++ &> /dev/null; then
    FREEFEM_PATH=$(which freefem++)
else
    echo "警告: 未找到FreeFEM，请确保已安装并添加到PATH"
    FREEFEM_PATH="/usr/local/bin/FreeFem++"
fi

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "创建Python虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "检查并安装依赖..."
pip install -r requirements.txt -q

# 创建必要的目录
echo "创建工作目录..."
mkdir -p work vtk_output simulations logs

# 检查端口占用
echo "检查端口占用..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "警告: 端口8000已被占用"
    echo -n "是否继续？(y/n): "
    read -r continue
    if [ "$continue" != "y" ]; then
        exit 1
    fi
fi

# 设置环境变量
export FREEFEM_EXECUTABLE=$FREEFEM_PATH

# 启动服务
echo ""
echo "启动FreeFEM API服务..."
echo "访问地址: http://localhost:8000"
echo "静态页面: http://localhost:8000/static/index.html"
echo "API文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo "========================================"

# 启动FastAPI
exec uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
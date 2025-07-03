# -*- coding: gb2312 -*-
# FreeFEM + Python API Docker镜像
FROM ubuntu:20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=zh_CN.GB2312
ENV LC_ALL=zh_CN.GB2312
ENV PYTHONIOENCODING=gb2312
ENV TZ=Asia/Shanghai

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    # 基础工具
    wget \
    curl \
    git \
    build-essential \
    cmake \
    gfortran \
    # Python
    python3.8 \
    python3-pip \
    python3-dev \
    # FreeFEM依赖
    libopenblas-dev \
    liblapack-dev \
    libmumps-dev \
    libscotch-dev \
    libfftw3-dev \
    libhdf5-dev \
    libgsl-dev \
    # 图形库
    libgl1-mesa-glx \
    libglu1-mesa \
    libxrender1 \
    libxcursor1 \
    libxinerama1 \
    libxi6 \
    # 中文支持
    language-pack-zh-hans \
    fonts-wqy-microhei \
    fonts-wqy-zenhei \
    # 其他工具
    supervisor \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# 配置中文locale
RUN locale-gen zh_CN.GB2312 && \
    update-locale LANG=zh_CN.GB2312

# 安装FreeFEM
RUN wget -O /tmp/freefem.deb https://github.com/FreeFem/FreeFem-sources/releases/download/v4.11/FreeFEM_4.11_Ubuntu_amd64.deb && \
    dpkg -i /tmp/freefem.deb || apt-get install -f -y && \
    rm /tmp/freefem.deb

# 安装Python依赖
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# 安装ParaView Python（可选）
# RUN wget -O /tmp/paraview.tar.gz https://www.paraview.org/paraview-downloads/download.php?submit=Download&version=v5.10&type=binary&os=Linux&downloadFile=ParaView-5.10.1-MPI-Linux-Python3.9-x86_64.tar.gz && \
#     tar -xzf /tmp/paraview.tar.gz -C /opt/ && \
#     rm /tmp/paraview.tar.gz
# ENV PYTHONPATH=/opt/ParaView-5.10.1-MPI-Linux-Python3.9-x86_64/lib/python3.9/site-packages:$PYTHONPATH

# 复制应用代码
COPY . /app/

# 创建必要的目录
RUN mkdir -p /app/work \
    /app/vtk_output \
    /app/simulations \
    /app/logs \
    /app/static \
    /var/log/supervisor

# 设置权限
RUN chmod +x /app/scripts/*.sh

# 配置Nginx
COPY nginx.conf /etc/nginx/sites-available/default

# 配置Supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 暴露端口
EXPOSE 80 8000 9000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# 启动命令
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
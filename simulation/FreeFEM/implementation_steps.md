# FreeFEM轧辊应力场可视化系统 - 实施步骤指南

## 概述

本指南提供了从零开始部署FreeFEM + VTK + ParaviewWeb轧辊应力场可视化系统的详细步骤。

## 系统架构

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│   前端界面   │────▶│ FastAPI后端  │────▶│ FreeFEM求解器  │
│  (Vue.js)   │     │  (Python)    │     │    (C++)       │
└─────────────┘     └──────────────┘     └────────────────┘
        │                    │                      │
        │                    ▼                      ▼
        │            ┌──────────────┐      ┌──────────────┐
        └───────────▶│ ParaviewWeb  │◀─────│ VTK导出器   │
                     │   服务器     │       │  (Python)    │
                     └──────────────┘      └──────────────┘
```

## 第一步：环境准备

### 1.1 系统要求
- Windows 10/11 或 Linux (Ubuntu 20.04+)
- Python 3.8+
- Docker和Docker Compose（生产环境）
- 至少8GB内存，推荐16GB

### 1.2 安装FreeFEM
```bash
# Windows
# 下载并运行 FreeFem++-4.15-win64.exe

# Linux
sudo apt-get update
sudo apt-get install freefem++
```

### 1.3 安装Python依赖
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install fastapi uvicorn numpy scipy vtk meshio websockets redis psutil
```

## 第二步：核心模块开发

### 2.1 创建项目结构
```
rolling-simulation/
├── freefem_template_engine.py    # FreeFEM模板引擎
├── freefem_executor.py           # FreeFEM执行器
├── advanced_vtk_exporter.py      # VTK导出器
├── paraviewweb_server.py         # ParaviewWeb服务
├── main_api.py                   # FastAPI主服务
├── requirements.txt              # Python依赖
├── templates/                    # FreeFEM模板
├── static/                       # 前端文件
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── work/                         # 工作目录
├── vtk_output/                   # VTK输出
└── simulations/                  # 模拟结果
```

### 2.2 实现关键模块

1. **FreeFEM模板引擎** (freefem_template_engine.py)
   - 动态生成FreeFEM脚本
   - 参数验证和默认值管理
   - GB2312编码支持

2. **VTK导出器** (advanced_vtk_exporter.py)
   - FreeFEM网格转VTK格式
   - 场数据映射（温度、应力、位移）
   - 时间序列支持（PVD文件）

3. **ParaviewWeb服务** (paraviewweb_server.py)
   - WebSocket通信协议
   - 实时数据加载和更新
   - 相机控制和渲染设置

## 第三步：API服务搭建

### 3.1 启动FastAPI服务
```bash
# 开发模式
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn main_api:app --workers 4 --host 0.0.0.0 --port 8000
```

### 3.2 API端点测试
```bash
# 健康检查
curl http://localhost:8000/

# 创建模拟
curl -X POST http://localhost:8000/api/v1/simulation/create \
  -H "Content-Type: application/json" \
  -d '{
    "rolling_params": {
      "roll_radius": 0.5,
      "thickness_initial": 0.025,
      "thickness_final": 0.020
    }
  }'

# 查看API文档
# 浏览器访问 http://localhost:8000/docs
```

## 第四步：ParaviewWeb配置

### 4.1 安装ParaView
```bash
# 下载ParaView
wget https://www.paraview.org/download/

# 安装Python支持
pip install wslink
```

### 4.2 启动ParaviewWeb服务
```bash
# 设置环境变量
export PVW_DATA_PATH=/path/to/simulations

# 启动服务
python paraviewweb_server.py
```

## 第五步：前端界面部署

### 5.1 配置Nginx（可选）
```nginx
server {
    listen 80;
    
    location / {
        root /path/to/static;
        index index.html;
    }
    
    location /api/ {
        proxy_pass http://localhost:8000;
    }
    
    location /paraview/ {
        proxy_pass http://localhost:9000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 5.2 访问系统
```
主界面: http://localhost/
API文档: http://localhost:8000/docs
```

## 第六步：Docker容器化部署

### 6.1 构建镜像
```bash
# 构建所有镜像
docker-compose build

# 或分别构建
docker build -f Dockerfile.freefem -t rolling-freefem .
docker build -f Dockerfile.paraview -t rolling-paraview .
```

### 6.2 启动服务
```bash
# 启动所有服务
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f freefem-api
```

### 6.3 数据持久化
```yaml
# docker-compose.yml中的卷映射
volumes:
  - ./simulations:/data/simulations
  - ./vtk_output:/data/vtk_output
  - ./work:/app/work
```

## 第七步：系统测试

### 7.1 功能测试清单
- [ ] FreeFEM脚本生成和执行
- [ ] VTK文件导出（单个时间步和时间序列）
- [ ] ParaviewWeb数据加载和显示
- [ ] WebSocket实时进度推送
- [ ] 前端参数输入和结果显示

### 7.2 性能测试
```python
# 测试脚本 test_performance.py
import time
import requests
import concurrent.futures

def run_simulation(params):
    response = requests.post(
        "http://localhost:8000/api/v1/simulation/create",
        json={"rolling_params": params}
    )
    return response.json()

# 并发测试
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(10):
        params = {
            "roll_radius": 0.5 + i*0.01,
            "thickness_initial": 0.025,
            "thickness_final": 0.020
        }
        futures.append(executor.submit(run_simulation, params))
```

## 第八步：生产环境优化

### 8.1 性能优化
1. **FreeFEM优化**
   - 使用并行求解器（MUMPS）
   - 自适应网格细化
   - 优化时间步长

2. **数据传输优化**
   - 压缩VTK文件（zlib）
   - 简化网格用于Web显示
   - 使用Redis缓存结果

3. **服务器优化**
   - 使用Gunicorn代替uvicorn
   - 配置反向代理缓存
   - 启用HTTP/2

### 8.2 监控和日志
```python
# 配置日志
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
```

### 8.3 备份策略
```bash
# 自动备份脚本
#!/bin/bash
BACKUP_DIR="/backup/rolling-simulation"
DATE=$(date +%Y%m%d_%H%M%S)

# 备份模拟结果
tar -czf $BACKUP_DIR/simulations_$DATE.tar.gz /data/simulations

# 备份VTK文件
tar -czf $BACKUP_DIR/vtk_$DATE.tar.gz /data/vtk_output

# 清理旧备份（保留7天）
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

## 第九步：故障排除

### 9.1 常见问题

**问题1：FreeFEM执行超时**
```python
# 增加超时时间
exec_result = executor.execute_script(script_file, timeout=1200)  # 20分钟
```

**问题2：ParaviewWeb连接失败**
```bash
# 检查端口
netstat -an | grep 9000

# 检查进程
ps aux | grep paraview
```

**问题3：中文编码问题**
```python
# 确保使用正确的编码
with open(file, 'w', encoding='gb2312') as f:
    f.write(content)
```

### 9.2 调试技巧
1. 启用详细日志
2. 使用FreeFEM的`-v 1`选项查看详细输出
3. 检查VTK文件的完整性
4. 使用浏览器开发者工具调试WebSocket

## 第十步：扩展功能

### 10.1 添加新的物理场
1. 修改FreeFEM模板添加新的场变量
2. 更新VTK导出器处理新字段
3. 在ParaviewWeb中配置新的可视化选项

### 10.2 支持3D模拟
1. 扩展FreeFEM几何定义支持3D
2. 更新网格生成算法
3. 调整VTK导出处理3D数据

### 10.3 集成机器学习
1. 收集历史模拟数据
2. 训练预测模型
3. 提供快速预测API

## 总结

通过以上步骤，您可以成功部署一个完整的FreeFEM轧辊应力场可视化系统。该系统具有以下特点：

- **高性能计算**：使用FreeFEM核心求解器
- **实时可视化**：ParaviewWeb提供交互式3D显示  
- **易于使用**：Web界面友好，参数输入直观
- **可扩展性**：模块化设计，易于添加新功能
- **生产就绪**：Docker容器化，易于部署和维护

如需技术支持，请参考项目文档或联系开发团队。
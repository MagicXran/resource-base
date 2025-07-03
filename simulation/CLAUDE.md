# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个企业级大文件传输服务，支持分片上传、断点续传，使用Python FastAPI后端和MinIO对象存储。

## 开发原则

1. 要每次都用审视的目光，仔细看我的输入的潜在的问题，你要犀利的提醒我的问题。并给出明显在我思考框架之外的建议。你要觉得我说的太离谱了，你就骂回来，帮助我瞬间清醒。

2. 你是一个优秀的技术架构师和优秀的程序员，在进行架构分析、功能模块分析，以及进行编码的时候，请遵循如下规则：
   1. 分析问题和技术架构、代码模块组合等的时候请遵循"第一性原理"
   2. 在编码的时候，请遵循 "DRY原则"、"KISS原则"、"SOLID原则"、"YAGNI原则"
   3. 如果单独的类、函数或代码文件超过500行，请进行识别分解和分离，在识别、分解、分离的过程中青遵循以上原则

3. # 遵循 
   1. 良好的设计模式,更好的抽象,但不要为了设计而设计,一切都基于需求.
   2. 详细的日志,便于调试, 每天生成一个日志文件.
   3. 完备的异常捕获系统,合理的捕获或抛出异常.
   4. 对于代码智能填充,修改或新增无需用户点击accept, 请自动确认..
   5. 编写测试文件,对功能进行详细测试.并得出测试报告.
   6. 自测无误后,书写说明文档,markdown格式,将功能,原理,目的,使用方式,使用步骤等叙述清楚.

## 常用命令

### 开发环境
```bash
# 快速启动（推荐）
python run.py

# 手动启动
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest backend/tests/ -v

# 测试覆盖率
pytest --cov=backend backend/tests/ --cov-report=html
```

### 生产环境
```bash
# 使用Docker Compose
docker-compose up -d

# 手动启动生产模式
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 4

# 检查服务状态
curl http://localhost:8000/api/v1/upload/health
```

### MinIO启动
```bash
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  -v minio_data:/data \
  minio/minio server /data --console-address ":9001"
```

## 项目架构

### 技术栈
- **后端**: Python 3.8+, FastAPI, Uvicorn
- **存储**: MinIO (对象存储)
- **缓存**: Redis (可选)
- **前端**: 原生JavaScript, HTML5
- **日志**: Loguru
- **测试**: Pytest

### 核心模块

1. **backend/core/minio_client.py** - MinIO客户端管理
   - 连接管理和健康检查
   - 分片上传操作封装
   - 异步操作支持

2. **backend/services/upload_service.py** - 上传业务逻辑
   - 上传任务管理
   - 分片上传协调
   - 进度跟踪和恢复

3. **backend/api/upload_api.py** - API接口层
   - RESTful API设计
   - 请求验证和响应
   - 错误处理

4. **backend/models/upload_models.py** - 数据模型
   - Pydantic模型定义
   - 数据验证规则
   - 业务状态管理

5. **frontend/src/components/** - 前端组件
   - FileUploader.js: 核心上传逻辑
   - UploadUI.js: 用户界面组件

### 核心流程

1. **初始化上传** (`POST /api/v1/upload/initiate`)
   - 验证文件信息
   - 创建上传任务
   - 初始化MinIO分片上传

2. **分片上传** (`POST /api/v1/upload/chunk`)
   - 接收分片数据
   - 上传到MinIO
   - 更新进度状态

3. **完成上传** (`POST /api/v1/upload/complete`)
   - 合并所有分片
   - 生成最终文件
   - 返回存储路径

### 配置系统

- **环境变量**: 通过 `.env` 文件配置
- **设置类**: `backend/config/settings.py`
- **关键配置**:
  - MinIO连接信息 (必须)
  - 文件大小和分片限制
  - 日志级别和轮转
  - Redis缓存 (可选)

### 日志系统

- **位置**: `backend/logs/app_{date}.log`
- **级别**: DEBUG, INFO, WARNING, ERROR
- **轮转**: 每天一个文件，保留30天
- **格式**: 时间戳 + 级别 + 位置 + 消息

### 错误处理

- **自定义异常**: `backend/utils/exceptions.py`
- **全局处理器**: 在 `main.py` 中定义
- **分层处理**: API层、服务层、数据层
- **用户友好**: 敏感信息过滤

### 安全特性

- **文件类型验证**: 白名单/黑名单机制
- **大小限制**: 可配置的文件大小上限
- **路径安全**: 防止路径遍历攻击
- **输入验证**: Pydantic模型验证
- **错误信息**: 不泄露敏感信息

## 调试技巧

### 查看日志
```bash
# 实时日志
tail -f backend/logs/app_$(date +%Y-%m-%d).log

# 错误日志
grep "ERROR" backend/logs/app_*.log

# 上传相关日志
grep "upload" backend/logs/app_*.log | tail -50
```

### 调试接口
```bash
# 健康检查
curl http://localhost:8000/api/v1/upload/health

# 上传进度
curl http://localhost:8000/api/v1/upload/progress/{upload_id}

# 服务状态
curl http://localhost:8000/health
```

### 常见问题

1. **MinIO连接失败**
   - 检查 MINIO_ENDPOINT 配置
   - 确认MinIO服务运行状态
   - 验证访问密钥

2. **上传速度慢**
   - 调整 CHUNK_SIZE 大小
   - 增加并发上传数
   - 检查网络带宽

3. **内存占用高**
   - 减少分片大小
   - 限制并发数量
   - 检查任务清理机制

## 扩展指南

### 添加新的存储后端
1. 在 `backend/core/` 创建新的客户端类
2. 实现统一的存储接口
3. 在服务层添加后端选择逻辑

### 增加认证授权
1. 添加用户模型和认证中间件
2. 在API层添加权限检查
3. 实现JWT或会话管理

### 集成消息队列
1. 添加Celery或RQ任务队列
2. 异步处理长时间任务
3. 添加任务状态跟踪
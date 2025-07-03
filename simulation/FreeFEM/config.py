# -*- coding: gb2312 -*-
"""
全局配置文件
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置"""
    
    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "FreeFEM轧制应力场分析系统"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "基于FreeFEM的热轧过程有限元分析API"
    
    # 路径配置
    BASE_DIR: Path = Path(__file__).parent
    WORK_DIR: Path = BASE_DIR / "work"
    VTK_OUTPUT_DIR: Path = BASE_DIR / "vtk_output"
    SIMULATIONS_DIR: Path = BASE_DIR / "simulations"
    LOGS_DIR: Path = BASE_DIR / "logs"
    STATIC_DIR: Path = BASE_DIR / "static"
    TEMPLATES_DIR: Path = BASE_DIR / "templates"
    
    # FreeFEM配置
    FREEFEM_EXECUTABLE: str = os.environ.get("FREEFEM_EXECUTABLE", "FreeFem++")
    FREEFEM_TIMEOUT: int = 600  # 10分钟
    FREEFEM_MAX_MEMORY_MB: int = 4096  # 4GB
    
    # 任务配置
    MAX_CONCURRENT_TASKS: int = 4
    TASK_RETENTION_DAYS: int = 7
    TASK_CLEANUP_INTERVAL: int = 3600  # 1小时
    
    # Redis配置（可选）
    REDIS_URL: Optional[str] = os.environ.get("REDIS_URL", None)
    CACHE_TTL: int = 3600  # 1小时
    
    # ParaviewWeb配置
    PARAVIEW_WS_URL: str = os.environ.get("PARAVIEW_WS_URL", "ws://localhost:9000/ws")
    PARAVIEW_ENABLED: bool = os.environ.get("PARAVIEW_ENABLED", "false").lower() == "true"
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "30 days"
    
    # 安全配置
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "your-secret-key-here")
    ALLOWED_HOSTS: list = ["*"]
    CORS_ORIGINS: list = ["*"]
    
    # 文件限制
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: set = {".edp", ".json", ".dat", ".txt"}
    
    # 监控配置
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # 开发模式
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"
    RELOAD: bool = os.environ.get("RELOAD", "false").lower() == "true"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "gb2312"

# 创建配置实例
settings = Settings()

# 创建必要的目录
for dir_path in [
    settings.WORK_DIR,
    settings.VTK_OUTPUT_DIR,
    settings.SIMULATIONS_DIR,
    settings.LOGS_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 导出配置
API_HOST = settings.API_HOST
API_PORT = settings.API_PORT
WORK_DIR = settings.WORK_DIR
VTK_OUTPUT_DIR = settings.VTK_OUTPUT_DIR
SIMULATIONS_DIR = settings.SIMULATIONS_DIR
MAX_CONCURRENT_TASKS = settings.MAX_CONCURRENT_TASKS
TASK_RETENTION_DAYS = settings.TASK_RETENTION_DAYS
LOG_LEVEL = settings.LOG_LEVEL
FREEFEM_TIMEOUT = settings.FREEFEM_TIMEOUT
PARAVIEW_WS_URL = settings.PARAVIEW_WS_URL
# -*- coding: gb2312 -*-
"""
主API服务 - 集成FreeFEM、VTK和ParaviewWeb
完整的生产级FastAPI应用，包含所有错误处理和异常情况
"""

import os
import sys
import json
import uuid
import time
import asyncio
import logging
import traceback
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor
import psutil
import aiofiles
from contextlib import asynccontextmanager
import signal
import atexit

# FastAPI相关导入
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator, ValidationError
from enum import Enum
import uvicorn

# 项目模块导入
try:
    from freefem_template_engine import (
        FreeFEMTemplateEngine, RollingParameter, MaterialProperty,
        MeshParameter, SolverParameter
    )
    from freefem_executor import FreeFEMExecutor, ExecutionResult
    from advanced_vtk_exporter import AdvancedVTKExporter, VTKExportOptions
except ImportError as e:
    print(f"错误: 无法导入项目模块 - {e}")
    sys.exit(1)

# 配置文件
try:
    from config import (
        API_HOST, API_PORT, WORK_DIR, VTK_OUTPUT_DIR,
        SIMULATIONS_DIR, MAX_CONCURRENT_TASKS, TASK_RETENTION_DAYS,
        LOG_LEVEL, FREEFEM_TIMEOUT, PARAVIEW_WS_URL
    )
except ImportError:
    # 默认配置
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    WORK_DIR = Path("work")
    VTK_OUTPUT_DIR = Path("vtk_output")
    SIMULATIONS_DIR = Path("simulations")
    MAX_CONCURRENT_TASKS = 4
    TASK_RETENTION_DAYS = 7
    LOG_LEVEL = "INFO"
    FREEFEM_TIMEOUT = 600
    PARAVIEW_WS_URL = "ws://localhost:9000/ws"

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='gb2312'
)
logger = logging.getLogger(__name__)

# 创建必要的目录
for dir_path in [WORK_DIR, VTK_OUTPUT_DIR, SIMULATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================
# 数据模型
# ============================================

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SimulationRequest(BaseModel):
    """模拟请求模型"""
    rolling_params: Dict[str, float] = Field(..., description="轧制参数")
    material_props: Optional[Dict[str, float]] = Field(None, description="材料属性")
    mesh_params: Optional[Dict[str, Any]] = Field(None, description="网格参数")
    solver_params: Optional[Dict[str, Any]] = Field(None, description="求解器参数")
    export_options: Optional[Dict[str, Any]] = Field(None, description="导出选项")
    
    @validator('rolling_params')
    def validate_rolling_params(cls, v):
        try:
            params = RollingParameter(**v)
            params.validate()
            return v
        except Exception as e:
            raise ValueError(f"轧制参数验证失败: {e}")
    
    @validator('material_props')
    def validate_material_props(cls, v):
        if v is None:
            return v
        try:
            props = MaterialProperty(**v)
            props.validate()
            return v
        except Exception as e:
            raise ValueError(f"材料属性验证失败: {e}")
    
    class Config:
        schema_extra = {
            "example": {
                "rolling_params": {
                    "roll_radius": 0.5,
                    "thickness_initial": 0.025,
                    "thickness_final": 0.020,
                    "strip_width": 2.0,
                    "roll_speed": 3.8,
                    "temperature_initial": 1123,
                    "temperature_roll": 423,
                    "friction_coefficient": 0.3
                },
                "material_props": {
                    "density": 7850,
                    "youngs_modulus": 210e9,
                    "poisson_ratio": 0.3
                }
            }
        }


class SimulationTask(BaseModel):
    """模拟任务模型"""
    task_id: str
    status: TaskStatus
    progress: float = Field(0, ge=0, le=100)
    message: str = ""
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    resource_usage: Optional[Dict[str, float]] = None


class TaskManager:
    """任务管理器"""
    
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_TASKS):
        self.tasks: Dict[str, SimulationTask] = {}
        self.task_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.running_tasks = set()
        self.max_concurrent = max_concurrent
        self._shutdown = False
        
    async def add_task(self, request: SimulationRequest) -> str:
        """添加新任务"""
        task_id = str(uuid.uuid4())
        
        task = SimulationTask(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="任务已创建",
            created_at=datetime.now(),
            parameters=request.dict()
        )
        
        self.tasks[task_id] = task
        await self.task_queue.put((task_id, request))
        
        logger.info(f"任务已添加到队列: {task_id}")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[SimulationTask]:
        """获取任务信息"""
        return self.tasks.get(task_id)
    
    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        **kwargs
    ):
        """更新任务状态"""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        if status:
            task.status = status
        if progress is not None:
            task.progress = progress
        if message:
            task.message = message
        
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        # 更新时间戳
        if status == TaskStatus.RUNNING and not task.started_at:
            task.started_at = datetime.now()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.completed_at = datetime.now()
            if task.started_at:
                task.execution_time = (task.completed_at - task.started_at).total_seconds()
    
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SimulationTask]:
        """列出任务"""
        tasks = list(self.tasks.values())
        
        # 过滤状态
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        # 排序（最新的在前）
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        # 分页
        return tasks[offset:offset + limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        status_counts = {}
        for task in self.tasks.values():
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "status_counts": status_counts,
            "running_tasks": len(self.running_tasks),
            "queue_size": self.task_queue.qsize(),
            "max_concurrent": self.max_concurrent
        }
    
    async def cleanup_old_tasks(self, days: int = TASK_RETENTION_DAYS):
        """清理旧任务"""
        cutoff_date = datetime.now() - timedelta(days=days)
        removed_count = 0
        
        for task_id in list(self.tasks.keys()):
            task = self.tasks[task_id]
            if task.created_at < cutoff_date and task.status in [
                TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED
            ]:
                del self.tasks[task_id]
                removed_count += 1
                
                # 清理相关文件
                self._cleanup_task_files(task_id)
        
        if removed_count > 0:
            logger.info(f"清理了 {removed_count} 个旧任务")
    
    def _cleanup_task_files(self, task_id: str):
        """清理任务相关文件"""
        # 清理工作目录
        work_files = list(WORK_DIR.glob(f"*{task_id}*"))
        # 清理VTK输出
        vtk_files = list(VTK_OUTPUT_DIR.glob(f"*{task_id}*"))
        # 清理结果目录
        result_dir = SIMULATIONS_DIR / task_id
        
        for file_path in work_files + vtk_files:
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"清理文件失败 {file_path}: {e}")
        
        if result_dir.exists():
            try:
                shutil.rmtree(result_dir)
            except Exception as e:
                logger.error(f"清理结果目录失败 {result_dir}: {e}")
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.PENDING, TaskStatus.QUEUED]:
            self.update_task(task_id, status=TaskStatus.CANCELLED, message="任务已取消")
            return True
        elif task.status == TaskStatus.RUNNING:
            # TODO: 实现运行中任务的取消
            logger.warning(f"无法取消运行中的任务: {task_id}")
            return False
        else:
            return False
    
    def shutdown(self):
        """关闭任务管理器"""
        self._shutdown = True
        self.executor.shutdown(wait=True)
        logger.info("任务管理器已关闭")


# ============================================
# 全局对象
# ============================================

# 任务管理器
task_manager = TaskManager()

# FreeFEM组件
template_engine = FreeFEMTemplateEngine(
    template_dir="templates",
    output_dir=str(WORK_DIR)
)
freefem_executor = FreeFEMExecutor(
    working_dir=str(WORK_DIR),
    max_memory_mb=4096
)
vtk_exporter = AdvancedVTKExporter(
    output_dir=str(VTK_OUTPUT_DIR),
    options=VTKExportOptions(
        binary_format=True,
        compression=True,
        export_mesh_quality=True,
        export_field_statistics=True
    )
)

# WebSocket连接管理
websocket_connections: Dict[str, List[WebSocket]] = {}


# ============================================
# FastAPI应用
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("启动FastAPI应用...")
    
    # 检查FreeFEM安装
    if not freefem_executor.check_installation():
        logger.error("FreeFEM未正确安装！")
    
    # 启动任务处理器
    asyncio.create_task(process_task_queue())
    
    # 定期清理任务
    asyncio.create_task(periodic_cleanup())
    
    yield
    
    # 关闭时
    logger.info("关闭FastAPI应用...")
    task_manager.shutdown()
    vtk_exporter.cleanup()


app = FastAPI(
    title="轧制应力场分析系统API",
    description="FreeFEM + VTK + ParaviewWeb集成系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/vtk", StaticFiles(directory=str(VTK_OUTPUT_DIR)), name="vtk")


# ============================================
# 异常处理
# ============================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    """处理验证错误"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "body": exc.body
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """处理一般异常"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )


# ============================================
# API端点
# ============================================

@app.get("/", tags=["基础"])
async def root():
    """API根路径"""
    return {
        "name": "轧制应力场分析系统",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "api_docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "simulations": "/api/v1/simulations",
            "tasks": "/api/v1/tasks",
            "visualization": "/api/v1/visualization"
        },
        "system_time": datetime.now().isoformat()
    }


@app.get("/health", tags=["基础"])
async def health_check():
    """健康检查"""
    try:
        # 检查各组件状态
        freefem_ok = freefem_executor.check_installation()
        
        # 系统资源
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(WORK_DIR))
        
        return {
            "status": "healthy" if freefem_ok else "degraded",
            "components": {
                "freefem": "ok" if freefem_ok else "error",
                "api": "ok",
                "storage": "ok" if disk.percent < 90 else "warning"
            },
            "resources": {
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / 1024**3,
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free / 1024**3,
                "cpu_count": os.cpu_count()
            },
            "tasks": task_manager.get_statistics()
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/api/v1/simulations", tags=["模拟"])
async def create_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """创建新的模拟任务"""
    try:
        # 检查并发限制
        stats = task_manager.get_statistics()
        if stats["running_tasks"] >= task_manager.max_concurrent:
            return {
                "status": "queued",
                "message": f"已达到最大并发数 ({task_manager.max_concurrent})，任务已加入队列",
                "queue_position": stats["queue_size"] + 1
            }
        
        # 添加任务
        task_id = await task_manager.add_task(request)
        
        return {
            "task_id": task_id,
            "status": "accepted",
            "message": "模拟任务已创建",
            "created_at": datetime.now().isoformat(),
            "estimated_time": estimate_execution_time(request)
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        logger.error(f"创建模拟失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/simulations/{task_id}", tags=["模拟"])
async def get_simulation_status(task_id: str) -> Dict[str, Any]:
    """获取模拟任务状态"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    response = task.dict()
    
    # 添加额外信息
    if task.status == TaskStatus.RUNNING:
        response["elapsed_time"] = (datetime.now() - task.started_at).total_seconds()
    
    # 如果完成，添加结果链接
    if task.status == TaskStatus.COMPLETED and task.results:
        response["download_links"] = {
            "vtk_files": f"/api/v1/simulations/{task_id}/downloads/vtk",
            "results": f"/api/v1/simulations/{task_id}/results",
            "visualization": f"/api/v1/visualization/{task_id}"
        }
    
    return response


@app.get("/api/v1/simulations/{task_id}/results", tags=["模拟"])
async def get_simulation_results(task_id: str) -> Dict[str, Any]:
    """获取模拟结果"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"任务未完成，当前状态: {task.status}"
        )
    
    if not task.results:
        raise HTTPException(status_code=404, detail="结果不可用")
    
    return {
        "task_id": task_id,
        "status": "completed",
        "results": task.results,
        "execution_time": task.execution_time,
        "resource_usage": task.resource_usage,
        "visualization_url": f"{PARAVIEW_WS_URL}?simulation={task_id}"
    }


@app.delete("/api/v1/simulations/{task_id}", tags=["模拟"])
async def cancel_simulation(task_id: str) -> Dict[str, Any]:
    """取消模拟任务"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    success = task_manager.cancel_task(task_id)
    
    if success:
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "任务已取消"
        }
    else:
        return {
            "task_id": task_id,
            "status": task.status,
            "message": f"无法取消状态为 {task.status} 的任务"
        }


@app.get("/api/v1/simulations/{task_id}/downloads/{file_type}", tags=["模拟"])
async def download_results(task_id: str, file_type: str):
    """下载结果文件"""
    task = task_manager.get_task(task_id)
    if not task or task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=404, detail="结果不可用")
    
    if file_type == "vtk":
        # 创建VTK文件压缩包
        zip_file = await create_vtk_archive(task_id)
        if zip_file and zip_file.exists():
            return FileResponse(
                path=zip_file,
                media_type="application/zip",
                filename=f"{task_id}_vtk_files.zip"
            )
    elif file_type == "report":
        # 生成报告
        report_file = await generate_report(task_id)
        if report_file and report_file.exists():
            return FileResponse(
                path=report_file,
                media_type="application/pdf",
                filename=f"{task_id}_report.pdf"
            )
    
    raise HTTPException(status_code=404, detail="文件类型不支持")


@app.get("/api/v1/tasks", tags=["任务"])
async def list_tasks(
    status: Optional[TaskStatus] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """列出任务"""
    tasks = task_manager.list_tasks(status, limit, offset)
    
    return {
        "total": len(task_manager.tasks),
        "limit": limit,
        "offset": offset,
        "tasks": [task.dict() for task in tasks]
    }


@app.get("/api/v1/tasks/statistics", tags=["任务"])
async def get_task_statistics() -> Dict[str, Any]:
    """获取任务统计"""
    stats = task_manager.get_statistics()
    
    # 添加更多统计信息
    completed_tasks = [
        t for t in task_manager.tasks.values()
        if t.status == TaskStatus.COMPLETED
    ]
    
    if completed_tasks:
        exec_times = [t.execution_time for t in completed_tasks if t.execution_time]
        stats["average_execution_time"] = sum(exec_times) / len(exec_times) if exec_times else 0
        stats["total_completed"] = len(completed_tasks)
    
    return stats


@app.websocket("/ws/simulation/{task_id}")
async def simulation_progress_websocket(websocket: WebSocket, task_id: str):
    """WebSocket实时进度推送"""
    await websocket.accept()
    
    # 验证任务
    task = task_manager.get_task(task_id)
    if not task:
        await websocket.send_json({
            "type": "error",
            "message": "任务不存在"
        })
        await websocket.close()
        return
    
    # 添加到连接列表
    if task_id not in websocket_connections:
        websocket_connections[task_id] = []
    websocket_connections[task_id].append(websocket)
    
    try:
        # 发送初始状态
        await websocket.send_json({
            "type": "status",
            "data": task.dict()
        })
        
        # 保持连接，等待更新
        while True:
            # 检查任务是否完成
            task = task_manager.get_task(task_id)
            if task and task.status in [
                TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED
            ]:
                await websocket.send_json({
                    "type": "complete",
                    "data": task.dict()
                })
                break
            
            # 等待一段时间
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket断开: {task_id}")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        # 移除连接
        if task_id in websocket_connections:
            websocket_connections[task_id].remove(websocket)
            if not websocket_connections[task_id]:
                del websocket_connections[task_id]


@app.get("/api/v1/visualization/{task_id}", tags=["可视化"])
async def get_visualization_info(task_id: str) -> Dict[str, Any]:
    """获取可视化信息"""
    task = task_manager.get_task(task_id)
    if not task or task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=404, detail="可视化数据不可用")
    
    # 查找VTK文件
    vtk_files = list(VTK_OUTPUT_DIR.glob(f"{task_id}*.vt[uk]"))
    pvd_files = list(VTK_OUTPUT_DIR.glob(f"{task_id}*.pvd"))
    
    return {
        "task_id": task_id,
        "paraview_ws_url": PARAVIEW_WS_URL,
        "vtk_files": [f.name for f in vtk_files],
        "pvd_files": [f.name for f in pvd_files],
        "available_fields": task.results.get("available_fields", []) if task.results else [],
        "instructions": {
            "websocket": f"{PARAVIEW_WS_URL}?simulation={task_id}",
            "http": f"/static/visualization.html?task_id={task_id}"
        }
    }


# ============================================
# 后台任务处理
# ============================================

async def process_task_queue():
    """处理任务队列"""
    logger.info("任务队列处理器已启动")
    
    while not task_manager._shutdown:
        try:
            # 检查是否可以处理新任务
            if len(task_manager.running_tasks) >= task_manager.max_concurrent:
                await asyncio.sleep(1)
                continue
            
            # 获取任务
            try:
                task_id, request = await asyncio.wait_for(
                    task_manager.task_queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            
            # 检查任务是否已取消
            task = task_manager.get_task(task_id)
            if not task or task.status == TaskStatus.CANCELLED:
                continue
            
            # 开始处理任务
            task_manager.running_tasks.add(task_id)
            task_manager.update_task(task_id, status=TaskStatus.RUNNING)
            
            # 异步执行任务
            asyncio.create_task(execute_simulation_task(task_id, request))
            
        except Exception as e:
            logger.error(f"任务队列处理错误: {e}")
            await asyncio.sleep(1)
    
    logger.info("任务队列处理器已停止")


async def execute_simulation_task(task_id: str, request: SimulationRequest):
    """执行模拟任务"""
    try:
        logger.info(f"开始执行任务: {task_id}")
        
        # 广播进度
        await broadcast_progress(task_id, 10, "初始化模拟环境")
        
        # 1. 生成FreeFEM脚本
        script_path = await asyncio.to_thread(
            generate_freefem_script,
            task_id,
            request
        )
        await broadcast_progress(task_id, 20, "FreeFEM脚本已生成")
        
        # 2. 执行FreeFEM
        execution_result = await asyncio.to_thread(
            run_freefem_simulation,
            script_path,
            task_id
        )
        
        if not execution_result.success:
            raise Exception(f"FreeFEM执行失败: {execution_result.error_message}")
        
        await broadcast_progress(task_id, 60, "FreeFEM计算完成")
        
        # 3. 解析结果
        results = await asyncio.to_thread(
            parse_simulation_results,
            task_id
        )
        await broadcast_progress(task_id, 80, "结果解析完成")
        
        # 4. 导出VTK
        export_info = await asyncio.to_thread(
            export_vtk_files,
            results,
            task_id,
            request.export_options
        )
        await broadcast_progress(task_id, 95, "VTK文件导出完成")
        
        # 5. 完成任务
        final_results = {
            "simulation_results": results.get("final", {}),
            "mesh_info": results.get("mesh", {}),
            "field_statistics": export_info.get("statistics", {}),
            "export_files": export_info.get("files", []),
            "available_fields": list(results.get("fields", {}).keys())
        }
        
        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message="模拟完成",
            results=final_results,
            resource_usage={
                "memory_mb": execution_result.memory_usage,
                "cpu_percent": execution_result.cpu_usage
            }
        )
        
        await broadcast_progress(task_id, 100, "任务完成")
        logger.info(f"任务完成: {task_id}")
        
    except Exception as e:
        logger.error(f"任务执行失败 {task_id}: {e}")
        error_msg = str(e)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=f"模拟失败: {error_msg}",
            error=traceback.format_exc()
        )
        
        await broadcast_error(task_id, error_msg)
        
    finally:
        # 从运行列表中移除
        task_manager.running_tasks.discard(task_id)


def generate_freefem_script(task_id: str, request: SimulationRequest) -> str:
    """生成FreeFEM脚本"""
    logger.info(f"生成FreeFEM脚本: {task_id}")
    
    # 合并参数
    all_params = {
        "rolling_params": request.rolling_params,
        "material_props": request.material_props or {},
        "mesh_params": request.mesh_params or {},
        "solver_params": request.solver_params or {}
    }
    
    # 生成脚本
    script_path = template_engine.generate_script(
        all_params,
        task_id,
        f"rolling_{task_id}.edp"
    )
    
    # 验证脚本
    if not template_engine.validate_script(script_path):
        raise ValueError("生成的FreeFEM脚本验证失败")
    
    return script_path


def run_freefem_simulation(script_path: str, task_id: str) -> ExecutionResult:
    """运行FreeFEM模拟"""
    logger.info(f"执行FreeFEM模拟: {task_id}")
    
    result = freefem_executor.execute_script(
        script_path,
        timeout=FREEFEM_TIMEOUT,
        task_id=task_id,
        monitor_resources=True
    )
    
    # 更新进度（通过检查输出）
    if result.stdout:
        progress = parse_freefem_progress(result.stdout)
        if progress:
            asyncio.create_task(
                broadcast_progress(task_id, 30 + int(progress * 0.3), f"计算进度: {progress}%")
            )
    
    return result


def parse_simulation_results(task_id: str) -> Dict[str, Any]:
    """解析模拟结果"""
    logger.info(f"解析模拟结果: {task_id}")
    
    results = freefem_executor.parse_results(task_id)
    
    if not results:
        raise ValueError("无法解析FreeFEM结果")
    
    # 验证结果完整性
    required_keys = ["mesh", "fields"]
    missing_keys = [k for k in required_keys if k not in results]
    if missing_keys:
        logger.warning(f"结果缺少字段: {missing_keys}")
    
    return results


def export_vtk_files(
    results: Dict[str, Any],
    task_id: str,
    export_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """导出VTK文件"""
    logger.info(f"导出VTK文件: {task_id}")
    
    # 更新导出选项
    if export_options:
        for key, value in export_options.items():
            if hasattr(vtk_exporter.options, key):
                setattr(vtk_exporter.options, key, value)
    
    # 导出结果
    export_info = vtk_exporter.export_freefem_results(
        results,
        task_id,
        export_history=True
    )
    
    # 为ParaviewWeb导出
    if "mesh" in results and "fields" in results:
        web_files = vtk_exporter.export_for_paraviewweb(
            {
                'vertices': results['mesh'].vertices,
                'elements': results['mesh'].elements
            },
            results['fields'],
            task_id
        )
        export_info['web_files'] = web_files
    
    return export_info


def parse_freefem_progress(stdout: str) -> Optional[int]:
    """从FreeFEM输出解析进度"""
    import re
    
    # 查找进度模式
    patterns = [
        r"时间步\s*(\d+)/(\d+)",
        r"Step\s*(\d+)/(\d+)",
        r"Iteration\s*(\d+)/(\d+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, stdout)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            return int(current / total * 100)
    
    return None


async def broadcast_progress(task_id: str, progress: float, message: str):
    """广播进度更新"""
    task_manager.update_task(task_id, progress=progress, message=message)
    
    # WebSocket广播
    if task_id in websocket_connections:
        data = {
            "type": "progress",
            "data": {
                "progress": progress,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        disconnected = []
        for ws in websocket_connections[task_id]:
            try:
                await ws.send_json(data)
            except:
                disconnected.append(ws)
        
        # 清理断开的连接
        for ws in disconnected:
            websocket_connections[task_id].remove(ws)


async def broadcast_error(task_id: str, error_message: str):
    """广播错误信息"""
    if task_id in websocket_connections:
        data = {
            "type": "error",
            "data": {
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        for ws in websocket_connections[task_id]:
            try:
                await ws.send_json(data)
            except:
                pass


def estimate_execution_time(request: SimulationRequest) -> float:
    """估算执行时间（秒）"""
    # 基于参数的简单估算
    base_time = 60  # 基础时间
    
    # 根据网格密度调整
    if request.mesh_params:
        element_size = request.mesh_params.get("element_size", 0.001)
        base_time *= (0.001 / element_size)  # 网格越细，时间越长
    
    # 根据时间步数调整
    if request.solver_params:
        time_steps = request.solver_params.get("num_steps", 100)
        base_time *= (time_steps / 100)
    
    return min(base_time, 1800)  # 最多30分钟


async def create_vtk_archive(task_id: str) -> Optional[Path]:
    """创建VTK文件压缩包"""
    import zipfile
    
    vtk_files = list(VTK_OUTPUT_DIR.glob(f"{task_id}*"))
    if not vtk_files:
        return None
    
    zip_path = VTK_OUTPUT_DIR / f"{task_id}_vtk_files.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in vtk_files:
            zipf.write(file_path, file_path.name)
    
    return zip_path


async def generate_report(task_id: str) -> Optional[Path]:
    """生成报告（占位函数）"""
    # TODO: 实现报告生成
    return None


async def periodic_cleanup():
    """定期清理任务"""
    while not task_manager._shutdown:
        try:
            await asyncio.sleep(3600)  # 每小时
            await task_manager.cleanup_old_tasks()
            
            # 清理临时文件
            temp_files = list(WORK_DIR.glob("temp_*"))
            for temp_file in temp_files:
                if temp_file.stat().st_mtime < time.time() - 3600:
                    try:
                        temp_file.unlink()
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"定期清理错误: {e}")


# ============================================
# 信号处理
# ============================================

def signal_handler(signum, frame):
    """处理系统信号"""
    logger.info(f"收到信号 {signum}，准备关闭...")
    task_manager._shutdown = True


# 注册信号处理
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================
# 主程序入口
# ============================================

if __name__ == "__main__":
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("需要Python 3.8或更高版本")
        sys.exit(1)
    
    # 启动服务
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=False,
        access_log=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                }
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {
                "level": LOG_LEVEL,
                "handlers": ["default"]
            }
        }
    )
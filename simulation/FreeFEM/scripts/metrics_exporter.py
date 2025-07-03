# -*- coding: gb2312 -*-
"""
Prometheus指标导出器
"""

import os
import sys
import time
import psutil
import logging
from pathlib import Path
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='gb2312'
)
logger = logging.getLogger(__name__)

# 定义指标
# 系统指标
cpu_usage = Gauge('freefem_cpu_usage_percent', 'CPU使用率')
memory_usage = Gauge('freefem_memory_usage_bytes', '内存使用量')
memory_percent = Gauge('freefem_memory_usage_percent', '内存使用百分比')
disk_usage = Gauge('freefem_disk_usage_bytes', '磁盘使用量', ['path'])
disk_percent = Gauge('freefem_disk_usage_percent', '磁盘使用百分比', ['path'])

# 任务指标
active_tasks = Gauge('freefem_active_tasks', '活动任务数')
completed_tasks = Counter('freefem_completed_tasks_total', '完成任务总数')
failed_tasks = Counter('freefem_failed_tasks_total', '失败任务总数')
task_duration = Histogram('freefem_task_duration_seconds', '任务执行时间', 
                         buckets=(10, 30, 60, 120, 300, 600, 1800, 3600))

# 文件指标
vtk_files = Gauge('freefem_vtk_files_count', 'VTK文件数量')
simulation_files = Gauge('freefem_simulation_files_count', '模拟文件数量')
total_file_size = Gauge('freefem_total_file_size_bytes', '文件总大小')

# API指标
api_requests = Counter('freefem_api_requests_total', 'API请求总数', ['method', 'endpoint'])
api_errors = Counter('freefem_api_errors_total', 'API错误总数', ['method', 'endpoint'])
api_latency = Histogram('freefem_api_latency_seconds', 'API延迟', ['method', 'endpoint'])

# FreeFEM进程指标
freefem_processes = Gauge('freefem_processes_count', 'FreeFEM进程数')
freefem_cpu = Gauge('freefem_process_cpu_percent', 'FreeFEM进程CPU使用率')
freefem_memory = Gauge('freefem_process_memory_bytes', 'FreeFEM进程内存使用')


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.interval = 10  # 10秒更新一次
        
    def collect_system_metrics(self):
        """收集系统指标"""
        # CPU使用率
        cpu_usage.set(psutil.cpu_percent(interval=1))
        
        # 内存使用
        memory = psutil.virtual_memory()
        memory_usage.set(memory.used)
        memory_percent.set(memory.percent)
        
        # 磁盘使用
        paths = [
            (str(settings.WORK_DIR), 'work'),
            (str(settings.VTK_OUTPUT_DIR), 'vtk'),
            (str(settings.SIMULATIONS_DIR), 'simulations')
        ]
        
        for path, label in paths:
            if Path(path).exists():
                usage = psutil.disk_usage(path)
                disk_usage.labels(path=label).set(usage.used)
                disk_percent.labels(path=label).set(usage.percent)
    
    def collect_file_metrics(self):
        """收集文件指标"""
        # VTK文件
        vtk_count = 0
        vtk_size = 0
        
        if settings.VTK_OUTPUT_DIR.exists():
            for file in settings.VTK_OUTPUT_DIR.rglob('*.vtk'):
                vtk_count += 1
                vtk_size += file.stat().st_size
            
            for file in settings.VTK_OUTPUT_DIR.rglob('*.vtu'):
                vtk_count += 1
                vtk_size += file.stat().st_size
        
        vtk_files.set(vtk_count)
        
        # 模拟文件
        sim_count = 0
        sim_size = 0
        
        if settings.SIMULATIONS_DIR.exists():
            for file in settings.SIMULATIONS_DIR.rglob('*'):
                if file.is_file():
                    sim_count += 1
                    sim_size += file.stat().st_size
        
        simulation_files.set(sim_count)
        total_file_size.set(vtk_size + sim_size)
    
    def collect_process_metrics(self):
        """收集进程指标"""
        freefem_count = 0
        freefem_cpu_total = 0
        freefem_mem_total = 0
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            try:
                if 'freefem' in proc.info['name'].lower():
                    freefem_count += 1
                    freefem_cpu_total += proc.info['cpu_percent']
                    freefem_mem_total += proc.info['memory_info'].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        freefem_processes.set(freefem_count)
        freefem_cpu.set(freefem_cpu_total)
        freefem_memory.set(freefem_mem_total)
    
    def collect_task_metrics(self):
        """收集任务指标（从文件或Redis）"""
        # 这里需要根据实际的任务存储方式来实现
        # 示例：从文件系统统计
        active = 0
        completed = 0
        failed = 0
        
        if settings.WORK_DIR.exists():
            for status_file in settings.WORK_DIR.glob('*_status.json'):
                # 读取状态文件来统计任务
                pass
        
        active_tasks.set(active)
        # completed_tasks和failed_tasks是Counter，通常在任务完成时增加
    
    def run_once(self):
        """运行一次收集"""
        try:
            self.collect_system_metrics()
            self.collect_file_metrics()
            self.collect_process_metrics()
            self.collect_task_metrics()
            logger.debug("指标收集完成")
        except Exception as e:
            logger.error(f"收集指标失败: {e}")
    
    def run_forever(self):
        """持续运行收集"""
        logger.info(f"指标收集器已启动，更新间隔: {self.interval}秒")
        
        while True:
            self.run_once()
            time.sleep(self.interval)


def main():
    """主函数"""
    # 启动HTTP服务器
    port = settings.METRICS_PORT
    start_http_server(port)
    logger.info(f"Prometheus指标服务器已启动: http://0.0.0.0:{port}/metrics")
    
    # 启动收集器
    collector = MetricsCollector()
    collector.run_forever()


if __name__ == "__main__":
    main()
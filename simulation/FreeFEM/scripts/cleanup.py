# -*- coding: gb2312 -*-
"""
定期清理脚本 - 清理旧的临时文件和结果
"""

import os
import sys
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta

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


class FileCleanup:
    """文件清理器"""
    
    def __init__(self):
        self.retention_days = settings.TASK_RETENTION_DAYS
        self.cleanup_interval = settings.TASK_CLEANUP_INTERVAL
        self.directories = [
            settings.WORK_DIR,
            settings.VTK_OUTPUT_DIR,
            settings.SIMULATIONS_DIR
        ]
        
    def get_file_age_days(self, file_path: Path) -> float:
        """获取文件年龄（天）"""
        stat = file_path.stat()
        age = time.time() - stat.st_mtime
        return age / (24 * 3600)
    
    def should_delete(self, file_path: Path) -> bool:
        """判断是否应该删除文件"""
        # 跳过特殊文件
        if file_path.name in ['.gitkeep', 'README.md']:
            return False
        
        # 检查文件年龄
        age_days = self.get_file_age_days(file_path)
        return age_days > self.retention_days
    
    def cleanup_directory(self, directory: Path) -> int:
        """清理单个目录"""
        deleted_count = 0
        total_size = 0
        
        if not directory.exists():
            return deleted_count
        
        # 遍历目录
        for item in directory.rglob('*'):
            if item.is_file() and self.should_delete(item):
                try:
                    size = item.stat().st_size
                    item.unlink()
                    deleted_count += 1
                    total_size += size
                    logger.debug(f"已删除文件: {item}")
                except Exception as e:
                    logger.error(f"删除文件失败 {item}: {e}")
        
        # 清理空目录
        for item in directory.rglob('*'):
            if item.is_dir() and not any(item.iterdir()):
                try:
                    item.rmdir()
                    logger.debug(f"已删除空目录: {item}")
                except Exception as e:
                    logger.error(f"删除目录失败 {item}: {e}")
        
        if deleted_count > 0:
            logger.info(f"从 {directory} 删除了 {deleted_count} 个文件，"
                       f"释放 {self.format_size(total_size)} 空间")
        
        return deleted_count
    
    def cleanup_logs(self):
        """清理日志文件"""
        log_dir = settings.LOGS_DIR
        if not log_dir.exists():
            return
        
        deleted_count = 0
        for log_file in log_dir.glob('*.log*'):
            if self.get_file_age_days(log_file) > 30:  # 日志保留30天
                try:
                    log_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"删除日志文件失败 {log_file}: {e}")
        
        if deleted_count > 0:
            logger.info(f"清理了 {deleted_count} 个旧日志文件")
    
    def check_disk_space(self):
        """检查磁盘空间"""
        import shutil
        
        for directory in self.directories:
            if directory.exists():
                stat = shutil.disk_usage(directory)
                free_gb = stat.free / (1024**3)
                used_percent = (stat.used / stat.total) * 100
                
                if free_gb < 1:  # 小于1GB
                    logger.warning(f"{directory} 磁盘空间不足: 剩余 {free_gb:.2f} GB")
                elif used_percent > 90:
                    logger.warning(f"{directory} 磁盘使用率过高: {used_percent:.1f}%")
    
    def format_size(self, bytes_size: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.2f} TB"
    
    def run_once(self):
        """运行一次清理"""
        logger.info(f"开始清理任务 (保留期: {self.retention_days} 天)")
        
        total_deleted = 0
        
        # 清理各个目录
        for directory in self.directories:
            deleted = self.cleanup_directory(directory)
            total_deleted += deleted
        
        # 清理日志
        self.cleanup_logs()
        
        # 检查磁盘空间
        self.check_disk_space()
        
        logger.info(f"清理完成，共删除 {total_deleted} 个文件")
    
    def run_forever(self):
        """持续运行清理任务"""
        logger.info("清理服务已启动")
        
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"清理任务出错: {e}")
            
            # 等待下一次清理
            logger.info(f"等待 {self.cleanup_interval} 秒后进行下一次清理")
            time.sleep(self.cleanup_interval)


def main():
    """主函数"""
    cleanup = FileCleanup()
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        # 只运行一次
        cleanup.run_once()
    else:
        # 持续运行
        cleanup.run_forever()


if __name__ == "__main__":
    main()
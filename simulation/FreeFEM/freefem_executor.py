# -*- coding: gb2312 -*-
"""
FreeFEM执行器 - 运行FreeFEM脚本并解析结果
包含完整的错误处理、超时控制和结果解析
"""

import subprocess
import os
import sys
import json
import numpy as np
from pathlib import Path
import logging
import time
import psutil
import signal
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import re
import shutil
import tempfile
from queue import Queue, Empty
import platform

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='gb2312'
)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """执行结果数据类"""
    success: bool
    execution_time: float
    stdout: str = ""
    stderr: str = ""
    return_code: Optional[int] = None
    error_message: Optional[str] = None
    memory_usage: Optional[float] = None  # MB
    cpu_usage: Optional[float] = None  # 百分比


@dataclass
class MeshData:
    """网格数据类"""
    vertices: np.ndarray  # 节点坐标
    elements: np.ndarray  # 单元连接
    n_vertices: int
    n_elements: int
    dimension: int = 2
    element_type: str = "triangle"
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取边界框"""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)
    
    def get_center(self) -> np.ndarray:
        """获取中心点"""
        return self.vertices.mean(axis=0)


@dataclass
class FieldData:
    """场数据类"""
    name: str
    data: np.ndarray
    field_type: str = "scalar"  # scalar, vector, tensor
    location: str = "point"  # point, cell
    
    def get_range(self) -> Tuple[float, float]:
        """获取数值范围"""
        return float(self.data.min()), float(self.data.max())
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        return {
            'min': float(self.data.min()),
            'max': float(self.data.max()),
            'mean': float(self.data.mean()),
            'std': float(self.data.std()) if self.data.size > 1 else 0.0
        }


class FreeFEMExecutor:
    """FreeFEM执行器"""
    
    def __init__(
        self,
        freefem_path: Optional[str] = None,
        working_dir: Optional[str] = None,
        max_memory_mb: int = 4096,
        enable_monitoring: bool = True
    ):
        """
        初始化执行器
        
        Args:
            freefem_path: FreeFEM可执行文件路径
            working_dir: 工作目录
            max_memory_mb: 最大内存限制(MB)
            enable_monitoring: 是否启用资源监控
        """
        # 自动检测FreeFEM路径
        self.freefem_path = freefem_path or self._find_freefem_executable()
        
        # 设置工作目录
        self.working_dir = Path(working_dir) if working_dir else Path("work")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # 资源限制
        self.max_memory_mb = max_memory_mb
        self.enable_monitoring = enable_monitoring
        
        # 进程管理
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.process_lock = threading.Lock()
        
        # 结果缓存
        self.result_cache: Dict[str, Any] = {}
        
        logger.info(f"FreeFEM执行器初始化 - 路径: {self.freefem_path}, 工作目录: {self.working_dir}")
    
    def _find_freefem_executable(self) -> str:
        """自动查找FreeFEM可执行文件"""
        # 可能的FreeFEM命令名称
        possible_names = ['FreeFem++', 'freefem++', 'ff++', 'FreeFem++-mpi']
        
        # Windows特定路径
        if platform.system() == 'Windows':
            possible_paths = [
                r"C:\Program Files\FreeFem++",
                r"C:\Program Files (x86)\FreeFem++",
                r"C:\FreeFem++",
                os.path.expanduser(r"~\FreeFem++")
            ]
            
            for path in possible_paths:
                for name in possible_names:
                    exe_path = os.path.join(path, f"{name}.exe")
                    if os.path.exists(exe_path):
                        return exe_path
        
        # 尝试从PATH查找
        for name in possible_names:
            path = shutil.which(name)
            if path:
                return path
        
        # 默认值
        return "FreeFem++"
    
    def check_installation(self) -> bool:
        """检查FreeFEM安装状态"""
        try:
            # 运行版本命令
            cmd = [self.freefem_path, "-version"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logger.info(f"FreeFEM版本: {version_info}")
                
                # 检查必要的模块
                self._check_required_modules()
                
                return True
            else:
                logger.error(f"FreeFEM版本检查失败: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error(f"FreeFEM未找到: {self.freefem_path}")
            logger.info("请确保FreeFEM已安装并在系统PATH中")
            return False
        except subprocess.TimeoutExpired:
            logger.error("FreeFEM版本检查超时")
            return False
        except Exception as e:
            logger.error(f"检查FreeFEM安装时出错: {e}")
            return False
    
    def _check_required_modules(self):
        """检查必要的FreeFEM模块"""
        required_modules = ['iovtk', 'msh3']
        
        test_script = """
        try { load "iovtk" }
        catch (...) { cout << "ERROR: iovtk module not found" << endl; }
        try { load "msh3" }
        catch (...) { cout << "ERROR: msh3 module not found" << endl; }
        """
        
        # 创建临时脚本
        temp_script = self.working_dir / "test_modules.edp"
        with open(temp_script, 'w') as f:
            f.write(test_script)
        
        try:
            result = self._run_command([self.freefem_path, str(temp_script)], timeout=5)
            if "ERROR" in result.stdout:
                logger.warning(f"FreeFEM模块检查警告: {result.stdout}")
        finally:
            temp_script.unlink(missing_ok=True)
    
    def execute_script(
        self,
        script_path: str,
        timeout: int = 600,
        task_id: Optional[str] = None,
        monitor_resources: bool = True
    ) -> ExecutionResult:
        """
        执行FreeFEM脚本
        
        Args:
            script_path: 脚本路径
            timeout: 超时时间（秒）
            task_id: 任务ID（用于跟踪）
            monitor_resources: 是否监控资源使用
            
        Returns:
            ExecutionResult: 执行结果
        """
        script_path = Path(script_path)
        if not script_path.exists():
            return ExecutionResult(
                success=False,
                execution_time=0,
                error_message=f"脚本文件不存在: {script_path}"
            )
        
        # 生成任务ID
        if task_id is None:
            task_id = f"task_{int(time.time())}"
        
        logger.info(f"开始执行FreeFEM脚本: {script_path} (任务ID: {task_id})")
        
        # 创建结果目录
        result_dir = self.working_dir / "results"
        result_dir.mkdir(exist_ok=True)
        
        # 准备命令
        cmd = [
            self.freefem_path,
            str(script_path),
            "-v", "0",  # 最小输出
            "-nw"       # 无窗口模式
        ]
        
        # 开始计时
        start_time = time.time()
        
        # 资源监控
        monitor_thread = None
        resource_queue = Queue()
        
        try:
            # 创建进程
            process = self._create_process(cmd, task_id)
            
            # 启动资源监控
            if monitor_resources and self.enable_monitoring:
                monitor_thread = threading.Thread(
                    target=self._monitor_process,
                    args=(process, resource_queue, task_id)
                )
                monitor_thread.daemon = True
                monitor_thread.start()
            
            # 等待执行完成
            stdout, stderr = process.communicate(timeout=timeout)
            return_code = process.returncode
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 获取资源使用统计
            max_memory = 0
            avg_cpu = 0
            if monitor_thread:
                monitor_thread.join(timeout=1)
                
                # 收集资源数据
                resource_data = []
                while not resource_queue.empty():
                    try:
                        resource_data.append(resource_queue.get_nowait())
                    except Empty:
                        break
                
                if resource_data:
                    max_memory = max(d['memory'] for d in resource_data)
                    avg_cpu = sum(d['cpu'] for d in resource_data) / len(resource_data)
            
            # 检查执行结果
            success = return_code == 0
            error_message = None
            
            if not success:
                error_message = self._parse_error_message(stderr)
                logger.error(f"FreeFEM执行失败: {error_message}")
            else:
                logger.info(f"FreeFEM执行成功，耗时: {execution_time:.2f}秒")
            
            # 创建结果对象
            result = ExecutionResult(
                success=success,
                execution_time=execution_time,
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                error_message=error_message,
                memory_usage=max_memory,
                cpu_usage=avg_cpu
            )
            
            # 缓存结果
            self.result_cache[task_id] = result
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"FreeFEM执行超时 ({timeout}秒)")
            
            # 终止进程
            self._terminate_process(process, task_id)
            
            return ExecutionResult(
                success=False,
                execution_time=timeout,
                error_message=f"执行超时 ({timeout}秒)"
            )
            
        except Exception as e:
            logger.error(f"FreeFEM执行异常: {e}")
            
            # 清理进程
            if 'process' in locals():
                self._terminate_process(process, task_id)
            
            return ExecutionResult(
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
        
        finally:
            # 清理进程记录
            with self.process_lock:
                self.active_processes.pop(task_id, None)
    
    def _create_process(self, cmd: List[str], task_id: str) -> subprocess.Popen:
        """创建子进程"""
        # 环境变量
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(os.cpu_count() or 1)
        
        # Windows特定设置
        kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'text': True,
            'cwd': self.working_dir,
            'env': env
        }
        
        if platform.system() == 'Windows':
            # Windows下隐藏控制台窗口
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs['startupinfo'] = startupinfo
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # Unix下设置进程组
            kwargs['preexec_fn'] = os.setsid
        
        # 创建进程
        process = subprocess.Popen(cmd, **kwargs)
        
        # 记录进程
        with self.process_lock:
            self.active_processes[task_id] = process
        
        return process
    
    def _terminate_process(self, process: subprocess.Popen, task_id: str):
        """终止进程"""
        try:
            if platform.system() == 'Windows':
                # Windows下发送CTRL_BREAK_EVENT
                process.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(1)
                if process.poll() is None:
                    process.terminate()
                    time.sleep(1)
                    if process.poll() is None:
                        process.kill()
            else:
                # Unix下发送SIGTERM
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(1)
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            
            logger.warning(f"进程已终止: {task_id}")
            
        except Exception as e:
            logger.error(f"终止进程失败: {e}")
    
    def _monitor_process(self, process: subprocess.Popen, queue: Queue, task_id: str):
        """监控进程资源使用"""
        try:
            psutil_process = psutil.Process(process.pid)
            
            while process.poll() is None:
                try:
                    # 获取CPU和内存使用
                    cpu_percent = psutil_process.cpu_percent(interval=0.1)
                    memory_info = psutil_process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    
                    # 记录数据
                    queue.put({
                        'timestamp': time.time(),
                        'cpu': cpu_percent,
                        'memory': memory_mb
                    })
                    
                    # 检查内存限制
                    if memory_mb > self.max_memory_mb:
                        logger.warning(f"内存使用超限: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
                        self._terminate_process(process, task_id)
                        break
                    
                    time.sleep(1)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                    
        except Exception as e:
            logger.error(f"资源监控错误: {e}")
    
    def _parse_error_message(self, stderr: str) -> str:
        """解析错误信息"""
        if not stderr:
            return "未知错误"
        
        # 常见错误模式
        error_patterns = [
            (r"Error.*?:\s*(.+)", "错误: {}"),
            (r"Fatal error.*?:\s*(.+)", "致命错误: {}"),
            (r"Syntax error.*?:\s*(.+)", "语法错误: {}"),
            (r"Memory.*?error", "内存错误"),
            (r"Segmentation fault", "段错误"),
            (r"Matrix.*?singular", "矩阵奇异"),
            (r"Convergence.*?failed", "收敛失败")
        ]
        
        for pattern, template in error_patterns:
            match = re.search(pattern, stderr, re.IGNORECASE)
            if match:
                if '{}' in template:
                    return template.format(match.group(1))
                else:
                    return template
        
        # 返回原始错误的前200个字符
        return stderr[:200].strip()
    
    def parse_results(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        解析FreeFEM输出结果
        
        Args:
            task_id: 任务ID（如果为None，使用最新的结果）
            
        Returns:
            包含解析结果的字典
        """
        results = {}
        
        # 确定结果目录
        result_dir = self.working_dir / "results"
        if not result_dir.exists():
            logger.warning("结果目录不存在")
            return results
        
        # 构造文件名后缀
        suffix = f"_{task_id}" if task_id else ""
        
        try:
            # 1. 解析JSON结果
            json_file = result_dir / f"final_results{suffix}.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    results['final'] = json.load(f)
                logger.info(f"已加载JSON结果: {json_file}")
            
            # 2. 解析网格数据
            mesh_data = self._parse_mesh_data(result_dir, suffix)
            if mesh_data:
                results['mesh'] = mesh_data
            
            # 3. 解析场数据
            field_data = self._parse_field_data(result_dir, suffix)
            if field_data:
                results['fields'] = field_data
            
            # 4. 解析时间历史数据
            history_data = self._parse_history_data(result_dir, suffix)
            if history_data:
                results['history'] = history_data
            
            # 5. 统计VTK文件
            vtk_files = list(result_dir.glob(f"rolling{suffix}_*.vtk"))
            if vtk_files:
                results['vtk_files'] = [str(f) for f in sorted(vtk_files)]
                logger.info(f"找到 {len(vtk_files)} 个VTK文件")
            
            return results
            
        except Exception as e:
            logger.error(f"解析结果失败: {e}")
            return results
    
    def _parse_mesh_data(self, result_dir: Path, suffix: str) -> Optional[MeshData]:
        """解析网格数据"""
        try:
            # 读取坐标
            coord_file = result_dir / f"coordinates{suffix}.dat"
            if not coord_file.exists():
                return None
            
            coordinates = np.loadtxt(coord_file)
            
            # 读取单元
            elem_file = result_dir / f"elements{suffix}.dat"
            if not elem_file.exists():
                return None
            
            elements = np.loadtxt(elem_file, dtype=int)
            
            # 创建网格数据对象
            mesh_data = MeshData(
                vertices=coordinates,
                elements=elements,
                n_vertices=len(coordinates),
                n_elements=len(elements),
                dimension=2 if coordinates.shape[1] == 2 else 3
            )
            
            logger.info(f"网格数据: {mesh_data.n_vertices} 节点, {mesh_data.n_elements} 单元")
            
            return mesh_data
            
        except Exception as e:
            logger.error(f"解析网格数据失败: {e}")
            return None
    
    def _parse_field_data(self, result_dir: Path, suffix: str) -> Dict[str, FieldData]:
        """解析场数据"""
        fields = {}
        
        # 定义场文件映射
        field_files = {
            'temperature': f'temperature_field{suffix}.dat',
            'von_mises': f'stress_field{suffix}.dat',
            'displacement': f'displacement_field{suffix}.dat'
        }
        
        for field_name, filename in field_files.items():
            field_file = result_dir / filename
            if not field_file.exists():
                continue
            
            try:
                if field_name == 'displacement':
                    # 位移是矢量场
                    data = np.loadtxt(field_file)
                    if data.ndim == 1:
                        # 单个点的情况
                        data = data.reshape(1, -1)
                    
                    field = FieldData(
                        name=field_name,
                        data=data,
                        field_type="vector",
                        location="point"
                    )
                else:
                    # 标量场
                    data = np.loadtxt(field_file)
                    field = FieldData(
                        name=field_name,
                        data=data,
                        field_type="scalar",
                        location="point"
                    )
                
                fields[field_name] = field
                logger.info(f"已加载场数据: {field_name}, 形状={data.shape}")
                
            except Exception as e:
                logger.error(f"加载场数据 {field_name} 失败: {e}")
        
        return fields
    
    def _parse_history_data(self, result_dir: Path, suffix: str) -> Dict[str, Any]:
        """解析时间历史数据"""
        history = {}
        
        # 定义历史文件
        history_files = {
            'temperature': f'temperature_history{suffix}.dat',
            'stress': f'stress_history{suffix}.dat',
            'force': f'force_history{suffix}.dat',
            'energy': f'energy_history{suffix}.dat'
        }
        
        for name, filename in history_files.items():
            hist_file = result_dir / filename
            if not hist_file.exists():
                continue
            
            try:
                # 读取数据（跳过注释行）
                data = np.loadtxt(hist_file, comments='#')
                if data.size == 0:
                    continue
                
                # 确保是2D数组
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                # 解析列
                if name == 'temperature':
                    history[name] = {
                        'time': data[:, 0],
                        'max': data[:, 1],
                        'min': data[:, 2],
                        'avg': data[:, 3] if data.shape[1] > 3 else None
                    }
                elif name == 'stress':
                    history[name] = {
                        'time': data[:, 0],
                        'von_mises_max': data[:, 1],
                        'tresca_max': data[:, 2] if data.shape[1] > 2 else None,
                        'pressure_max': data[:, 3] if data.shape[1] > 3 else None
                    }
                elif name == 'force':
                    history[name] = {
                        'time': data[:, 0],
                        'rolling_force': data[:, 1],
                        'friction_force': data[:, 2] if data.shape[1] > 2 else None
                    }
                elif name == 'energy':
                    history[name] = {
                        'time': data[:, 0],
                        'elastic': data[:, 1],
                        'thermal': data[:, 2] if data.shape[1] > 2 else None,
                        'plastic': data[:, 3] if data.shape[1] > 3 else None
                    }
                
                logger.info(f"已加载历史数据: {name}, {len(data)} 个时间步")
                
            except Exception as e:
                logger.error(f"加载历史数据 {name} 失败: {e}")
        
        return history
    
    def _run_command(self, cmd: List[str], timeout: int = 30) -> ExecutionResult:
        """运行简单命令"""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir
            )
            
            return ExecutionResult(
                success=result.returncode == 0,
                execution_time=time.time() - start_time,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                execution_time=timeout,
                error_message="命令执行超时"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def cleanup_old_results(self, days: int = 7):
        """清理旧的结果文件"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        cleaned_count = 0
        
        for file_path in self.working_dir.rglob("*"):
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.error(f"删除文件失败 {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"清理了 {cleaned_count} 个旧文件")
    
    def get_active_tasks(self) -> List[str]:
        """获取活动任务列表"""
        with self.process_lock:
            return list(self.active_processes.keys())
    
    def terminate_task(self, task_id: str) -> bool:
        """终止指定任务"""
        with self.process_lock:
            process = self.active_processes.get(task_id)
            if process:
                self._terminate_process(process, task_id)
                self.active_processes.pop(task_id, None)
                return True
        return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'freefem_path': self.freefem_path,
            'working_dir': str(self.working_dir),
            'cpu_count': os.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'memory_available_gb': psutil.virtual_memory().available / 1024**3,
            'disk_usage': psutil.disk_usage(str(self.working_dir)).percent
        }


def test_executor():
    """测试执行器功能"""
    executor = FreeFEMExecutor()
    
    # 检查安装
    if not executor.check_installation():
        print("FreeFEM未正确安装")
        return
    
    # 获取系统信息
    system_info = executor.get_system_info()
    print("系统信息:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # 创建测试脚本
    test_script = executor.working_dir / "test.edp"
    with open(test_script, 'w') as f:
        f.write("""
        // 简单测试脚本
        mesh Th = square(10, 10);
        fespace Vh(Th, P1);
        Vh u, v;
        
        solve Poisson(u, v) = 
            int2d(Th)(dx(u)*dx(v) + dy(u)*dy(v))
            - int2d(Th)(1*v)
            + on(1,2,3,4, u=0);
        
        cout << "Max u = " << u[].max << endl;
        cout << "Min u = " << u[].min << endl;
        """)
    
    # 执行脚本
    print("\n执行测试脚本...")
    result = executor.execute_script(test_script, timeout=30, task_id="test_001")
    
    print(f"执行结果: {'成功' if result.success else '失败'}")
    print(f"执行时间: {result.execution_time:.2f}秒")
    if result.memory_usage:
        print(f"内存使用: {result.memory_usage:.1f}MB")
    if result.cpu_usage:
        print(f"CPU使用: {result.cpu_usage:.1f}%")
    
    if result.stdout:
        print(f"输出:\n{result.stdout}")
    if result.stderr and not result.success:
        print(f"错误:\n{result.stderr}")


if __name__ == "__main__":
    test_executor()
# -*- coding: gb2312 -*-
"""
ParaviewWeb服务器 - 轧制应力场可视化
完整的WebSocket服务实现，支持实时交互
"""

import os
import sys
import json
import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import traceback

# ParaView和wslink导入
try:
    from wslink import server
    from wslink import register as exportRPC
    from wslink.websocket import LinkProtocol
    HAS_WSLINK = True
except ImportError:
    HAS_WSLINK = False
    print("警告: wslink未安装，ParaviewWeb功能将受限")

try:
    from paraview import simple
    from paraview.web import protocols as pv_protocols
    HAS_PARAVIEW = True
except ImportError:
    HAS_PARAVIEW = False
    print("警告: ParaView Python未安装，可视化功能将受限")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='gb2312'
)
logger = logging.getLogger(__name__)


@dataclass
class ViewSettings:
    """视图设置"""
    background_color: List[float] = None
    camera_position: List[float] = None
    camera_focal_point: List[float] = None
    camera_view_up: List[float] = None
    parallel_projection: bool = False
    view_size: List[int] = None
    
    def __post_init__(self):
        if self.background_color is None:
            self.background_color = [1.0, 1.0, 1.0]
        if self.camera_position is None:
            self.camera_position = [0.5, 0.5, 2.0]
        if self.camera_focal_point is None:
            self.camera_focal_point = [0.0, 0.0, 0.0]
        if self.camera_view_up is None:
            self.camera_view_up = [0.0, 1.0, 0.0]
        if self.view_size is None:
            self.view_size = [1024, 768]


@dataclass
class ColorMapSettings:
    """颜色映射设置"""
    name: str = "Cool to Warm"
    range_mode: str = "auto"  # auto, custom
    custom_range: Optional[List[float]] = None
    show_color_bar: bool = True
    color_bar_position: List[float] = None
    color_bar_title: str = ""
    
    def __post_init__(self):
        if self.color_bar_position is None:
            self.color_bar_position = [0.85, 0.2]


class RollingVisualizationProtocol(LinkProtocol if HAS_WSLINK else object):
    """轧制模拟可视化协议"""
    
    def __init__(self):
        super().__init__()
        
        # 配置
        self.data_path = Path(os.environ.get('PVW_DATA_PATH', '/data/simulations'))
        self.temp_path = Path(os.environ.get('PVW_TEMP_PATH', '/tmp/paraview'))
        self.temp_path.mkdir(parents=True, exist_ok=True)
        
        # 状态管理
        self.active_source = None
        self.active_view = None
        self.active_display = None
        self.current_field = None
        self.animation_playing = False
        
        # 会话数据
        self.session_data = {}
        self.loaded_simulations = {}
        
        # 视图设置
        self.view_settings = ViewSettings()
        self.color_settings = {}  # field_name -> ColorMapSettings
        
        # 线程池（用于异步操作）
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"ParaviewWeb协议初始化 - 数据路径: {self.data_path}")
    
    @exportRPC("rolling.initialize")
    def initialize(self) -> Dict[str, Any]:
        """初始化可视化环境"""
        try:
            if not HAS_PARAVIEW:
                return {
                    "status": "error",
                    "message": "ParaView未安装"
                }
            
            # 清理现有数据
            simple.Delete(simple.GetSources())
            
            # 创建渲染视图
            self.active_view = simple.GetActiveViewOrCreate('RenderView')
            self._setup_view()
            
            # 获取系统信息
            system_info = self._get_system_info()
            
            return {
                "status": "success",
                "message": "可视化环境已初始化",
                "system_info": system_info
            }
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
    
    @exportRPC("rolling.load_simulation")
    def load_simulation(self, simulation_id: str, file_type: str = "auto") -> Dict[str, Any]:
        """
        加载模拟结果
        
        Args:
            simulation_id: 模拟ID
            file_type: 文件类型 (auto, pvd, vtu, vtk)
        """
        try:
            logger.info(f"加载模拟: {simulation_id}")
            
            # 查找文件
            file_path = self._find_simulation_file(simulation_id, file_type)
            if not file_path:
                return {
                    "status": "error",
                    "message": f"未找到模拟文件: {simulation_id}"
                }
            
            # 清理旧数据
            if self.active_source:
                simple.Delete(self.active_source)
            
            # 加载数据
            reader = self._load_data_file(file_path)
            if not reader:
                return {
                    "status": "error",
                    "message": f"无法加载文件: {file_path}"
                }
            
            self.active_source = reader
            self.loaded_simulations[simulation_id] = {
                'source': reader,
                'file_path': str(file_path),
                'load_time': time.time()
            }
            
            # 显示数据
            if not self.active_view:
                self.active_view = simple.GetActiveViewOrCreate('RenderView')
            
            self.active_display = simple.Show(reader, self.active_view)
            self.active_display.Representation = 'Surface'
            
            # 设置默认着色
            available_fields = self._get_available_fields()
            default_field = self._select_default_field(available_fields)
            
            if default_field:
                self.update_field(default_field['name'], default_field.get('component'))
            
            # 重置相机
            simple.ResetCamera()
            simple.Render()
            
            # 获取数据信息
            data_info = self._get_data_information()
            
            return {
                "status": "success",
                "simulation_id": simulation_id,
                "file_path": str(file_path),
                "data_info": data_info,
                "available_fields": available_fields,
                "default_field": default_field
            }
            
        except Exception as e:
            logger.error(f"加载模拟失败: {e}")
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
    
    @exportRPC("rolling.update_field")
    def update_field(
        self,
        field_name: str,
        component: Optional[str] = None,
        location: str = "POINTS"
    ) -> Dict[str, Any]:
        """更新显示字段"""
        try:
            if not self.active_source or not self.active_display:
                return {
                    "status": "error",
                    "message": "没有加载的数据"
                }
            
            # 设置着色
            if component:
                simple.ColorBy(self.active_display, (location, field_name, component))
            else:
                simple.ColorBy(self.active_display, (location, field_name))
            
            self.current_field = field_name
            
            # 获取或创建颜色设置
            if field_name not in self.color_settings:
                self.color_settings[field_name] = ColorMapSettings(
                    color_bar_title=self._get_field_title(field_name)
                )
            
            color_setting = self.color_settings[field_name]
            
            # 更新颜色范围
            if color_setting.range_mode == "auto":
                self.active_display.RescaleTransferFunctionToDataRange(True)
            elif color_setting.custom_range:
                color_func = simple.GetColorTransferFunction(field_name)
                color_func.RescaleTransferFunction(
                    color_setting.custom_range[0],
                    color_setting.custom_range[1]
                )
            
            # 更新颜色条
            if color_setting.show_color_bar:
                self._update_color_bar(field_name, color_setting)
            
            # 渲染
            simple.Render()
            
            # 获取字段范围
            field_range = self._get_field_range(field_name, location)
            
            return {
                "status": "success",
                "field": field_name,
                "component": component,
                "location": location,
                "range": field_range,
                "color_settings": {
                    "color_map": color_setting.name,
                    "show_color_bar": color_setting.show_color_bar
                }
            }
            
        except Exception as e:
            logger.error(f"更新字段失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @exportRPC("rolling.set_color_map")
    def set_color_map(
        self,
        field_name: str,
        color_map_name: str,
        custom_range: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """设置颜色映射"""
        try:
            if not self.active_display:
                return {
                    "status": "error",
                    "message": "没有活动的显示"
                }
            
            # 获取颜色传输函数
            color_func = simple.GetColorTransferFunction(field_name)
            
            # 应用预设颜色映射
            if color_map_name in self._get_available_color_maps():
                simple.ApplyPreset(color_map_name, color_func)
            else:
                return {
                    "status": "error",
                    "message": f"未知的颜色映射: {color_map_name}"
                }
            
            # 更新设置
            if field_name not in self.color_settings:
                self.color_settings[field_name] = ColorMapSettings()
            
            self.color_settings[field_name].name = color_map_name
            
            # 设置范围
            if custom_range:
                color_func.RescaleTransferFunction(custom_range[0], custom_range[1])
                self.color_settings[field_name].range_mode = "custom"
                self.color_settings[field_name].custom_range = custom_range
            
            simple.Render()
            
            return {
                "status": "success",
                "field": field_name,
                "color_map": color_map_name,
                "range": custom_range or list(color_func.RGBPoints[0:2])
            }
            
        except Exception as e:
            logger.error(f"设置颜色映射失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @exportRPC("rolling.set_time_step")
    def set_time_step(self, step: int) -> Dict[str, Any]:
        """设置时间步"""
        try:
            animation_scene = simple.GetAnimationScene()
            
            # 获取时间范围
            time_keeper = animation_scene.TimeKeeper
            time_values = time_keeper.TimestepValues
            
            if not time_values or step >= len(time_values):
                return {
                    "status": "error",
                    "message": f"无效的时间步: {step}"
                }
            
            # 设置时间
            animation_scene.AnimationTime = time_values[step]
            simple.Render()
            
            return {
                "status": "success",
                "time_step": step,
                "time_value": float(time_values[step]),
                "total_steps": len(time_values)
            }
            
        except Exception as e:
            logger.error(f"设置时间步失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @exportRPC("rolling.play_animation")
    def play_animation(
        self,
        start_step: int = 0,
        end_step: Optional[int] = None,
        frame_rate: int = 10
    ) -> Dict[str, Any]:
        """播放动画"""
        try:
            if self.animation_playing:
                return {
                    "status": "error",
                    "message": "动画正在播放"
                }
            
            animation_scene = simple.GetAnimationScene()
            animation_scene.PlayMode = 'Sequence'
            
            # 设置范围
            time_keeper = animation_scene.TimeKeeper
            time_values = time_keeper.TimestepValues
            
            if time_values:
                if end_step is None:
                    end_step = len(time_values) - 1
                
                animation_scene.StartTime = time_values[start_step]
                animation_scene.EndTime = time_values[end_step]
                animation_scene.NumberOfFrames = end_step - start_step + 1
            
            # 异步播放
            self.animation_playing = True
            self.executor.submit(self._play_animation_async, animation_scene, frame_rate)
            
            return {
                "status": "success",
                "playing": True,
                "start_step": start_step,
                "end_step": end_step,
                "frame_rate": frame_rate
            }
            
        except Exception as e:
            logger.error(f"播放动画失败: {e}")
            self.animation_playing = False
            return {
                "status": "error",
                "message": str(e)
            }
    
    @exportRPC("rolling.stop_animation")
    def stop_animation(self) -> Dict[str, Any]:
        """停止动画"""
        try:
            if not self.animation_playing:
                return {
                    "status": "success",
                    "playing": False
                }
            
            self.animation_playing = False
            animation_scene = simple.GetAnimationScene()
            animation_scene.Stop()
            
            return {
                "status": "success",
                "playing": False
            }
            
        except Exception as e:
            logger.error(f"停止动画失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @exportRPC("rolling.set_camera")
    def set_camera(
        self,
        position: Optional[List[float]] = None,
        focal_point: Optional[List[float]] = None,
        view_up: Optional[List[float]] = None,
        parallel_projection: Optional[bool] = None
    ) -> Dict[str, Any]:
        """设置相机参数"""
        try:
            if not self.active_view:
                return {
                    "status": "error",
                    "message": "没有活动视图"
                }
            
            camera = self.active_view.GetActiveCamera()
            
            if position:
                camera.SetPosition(position)
                self.view_settings.camera_position = position
            
            if focal_point:
                camera.SetFocalPoint(focal_point)
                self.view_settings.camera_focal_point = focal_point
            
            if view_up:
                camera.SetViewUp(view_up)
                self.view_settings.camera_view_up = view_up
            
            if parallel_projection is not None:
                camera.SetParallelProjection(parallel_projection)
                self.view_settings.parallel_projection = parallel_projection
            
            simple.Render()
            
            return {
                "status": "success",
                "camera": {
                    "position": list(camera.GetPosition()),
                    "focal_point": list(camera.GetFocalPoint()),
                    "view_up": list(camera.GetViewUp()),
                    "parallel_projection": bool(camera.GetParallelProjection())
                }
            }
            
        except Exception as e:
            logger.error(f"设置相机失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @exportRPC("rolling.reset_camera")
    def reset_camera(self) -> Dict[str, Any]:
        """重置相机"""
        try:
            if not self.active_view:
                return {
                    "status": "error",
                    "message": "没有活动视图"
                }
            
            simple.ResetCamera()
            simple.Render()
            
            camera = self.active_view.GetActiveCamera()
            
            return {
                "status": "success",
                "camera": {
                    "position": list(camera.GetPosition()),
                    "focal_point": list(camera.GetFocalPoint()),
                    "view_up": list(camera.GetViewUp())
                }
            }
            
        except Exception as e:
            logger.error(f"重置相机失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @exportRPC("rolling.export_screenshot")
    def export_screenshot(
        self,
        filename: str = "screenshot.png",
        width: int = 1920,
        height: int = 1080,
        transparent_background: bool = False
    ) -> Dict[str, Any]:
        """导出截图"""
        try:
            if not self.active_view:
                return {
                    "status": "error",
                    "message": "没有活动视图"
                }
            
            # 构建完整路径
            screenshot_path = self.temp_path / filename
            
            # 保存截图
            simple.SaveScreenshot(
                str(screenshot_path),
                self.active_view,
                ImageResolution=[width, height],
                TransparentBackground=int(transparent_background)
            )
            
            # 验证文件
            if not screenshot_path.exists():
                return {
                    "status": "error",
                    "message": "截图保存失败"
                }
            
            return {
                "status": "success",
                "filename": filename,
                "path": str(screenshot_path),
                "size": [width, height],
                "file_size": screenshot_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"导出截图失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @exportRPC("rolling.get_field_statistics")
    def get_field_statistics(self, field_name: str) -> Dict[str, Any]:
        """获取字段统计信息"""
        try:
            if not self.active_source:
                return {
                    "status": "error",
                    "message": "没有加载的数据"
                }
            
            # 创建描述统计过滤器
            stats_filter = simple.DescriptiveStatistics(Input=self.active_source)
            stats_filter.VariableArray = ['POINTS', field_name]
            stats_filter.UpdatePipeline()
            
            # 获取统计表
            stats_table = stats_filter.GetClientSideObject().GetOutput()
            
            # 提取统计数据
            statistics = {}
            for i in range(stats_table.GetNumberOfRows()):
                stat_name = stats_table.GetValue(i, 0).ToString()
                stat_value = stats_table.GetValue(i, 1).ToFloat()
                statistics[stat_name] = float(stat_value)
            
            # 清理
            simple.Delete(stats_filter)
            
            return {
                "status": "success",
                "field": field_name,
                "statistics": statistics
            }
            
        except Exception as e:
            logger.error(f"获取字段统计失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @exportRPC("rolling.add_filter")
    def add_filter(self, filter_type: str, **params) -> Dict[str, Any]:
        """添加过滤器"""
        try:
            if not self.active_source:
                return {
                    "status": "error",
                    "message": "没有加载的数据"
                }
            
            # 创建过滤器
            filter_obj = None
            
            if filter_type == "contour":
                filter_obj = simple.Contour(Input=self.active_source)
                filter_obj.ContourBy = params.get('field', 'VonMises_MPa')
                filter_obj.Isosurfaces = params.get('values', [100, 200, 300])
                
            elif filter_type == "slice":
                filter_obj = simple.Slice(Input=self.active_source)
                filter_obj.SliceType = 'Plane'
                filter_obj.SliceType.Origin = params.get('origin', [0, 0, 0])
                filter_obj.SliceType.Normal = params.get('normal', [0, 0, 1])
                
            elif filter_type == "clip":
                filter_obj = simple.Clip(Input=self.active_source)
                filter_obj.ClipType = 'Plane'
                filter_obj.ClipType.Origin = params.get('origin', [0, 0, 0])
                filter_obj.ClipType.Normal = params.get('normal', [0, 0, 1])
                filter_obj.Invert = params.get('invert', False)
                
            elif filter_type == "threshold":
                filter_obj = simple.Threshold(Input=self.active_source)
                filter_obj.Scalars = ['POINTS', params.get('field', 'VonMises_MPa')]
                filter_obj.ThresholdRange = params.get('range', [100, 500])
                
            elif filter_type == "warp":
                filter_obj = simple.WarpByVector(Input=self.active_source)
                filter_obj.Vectors = ['POINTS', params.get('field', 'Displacement')]
                filter_obj.ScaleFactor = params.get('scale', 1.0)
                
            else:
                return {
                    "status": "error",
                    "message": f"未知的过滤器类型: {filter_type}"
                }
            
            # 显示过滤器结果
            filter_display = simple.Show(filter_obj, self.active_view)
            simple.Hide(self.active_source, self.active_view)
            
            # 更新显示
            simple.ColorBy(filter_display, ('POINTS', self.current_field or 'VonMises_MPa'))
            simple.Render()
            
            # 保存过滤器引用
            filter_id = f"filter_{int(time.time()*1000)}"
            self.session_data[filter_id] = {
                'filter': filter_obj,
                'display': filter_display,
                'type': filter_type,
                'params': params
            }
            
            return {
                "status": "success",
                "filter_id": filter_id,
                "filter_type": filter_type,
                "params": params
            }
            
        except Exception as e:
            logger.error(f"添加过滤器失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @exportRPC("rolling.remove_filter")
    def remove_filter(self, filter_id: str) -> Dict[str, Any]:
        """移除过滤器"""
        try:
            if filter_id not in self.session_data:
                return {
                    "status": "error",
                    "message": f"未找到过滤器: {filter_id}"
                }
            
            filter_data = self.session_data[filter_id]
            
            # 删除过滤器
            simple.Delete(filter_data['filter'])
            
            # 显示原始数据
            simple.Show(self.active_source, self.active_view)
            simple.Render()
            
            # 清理引用
            del self.session_data[filter_id]
            
            return {
                "status": "success",
                "filter_id": filter_id
            }
            
        except Exception as e:
            logger.error(f"移除过滤器失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @exportRPC("rolling.get_available_fields")
    def get_available_fields(self) -> Dict[str, Any]:
        """获取可用字段列表"""
        try:
            fields = self._get_available_fields()
            return {
                "status": "success",
                "fields": fields
            }
        except Exception as e:
            logger.error(f"获取字段列表失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @exportRPC("rolling.get_session_info")
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息"""
        try:
            info = {
                "loaded_simulations": list(self.loaded_simulations.keys()),
                "active_filters": [
                    {
                        "id": fid,
                        "type": fdata['type'],
                        "params": fdata['params']
                    }
                    for fid, fdata in self.session_data.items()
                ],
                "current_field": self.current_field,
                "view_settings": {
                    "background": self.view_settings.background_color,
                    "size": self.view_settings.view_size
                }
            }
            
            # 添加数据信息
            if self.active_source:
                info["data_info"] = self._get_data_information()
            
            return {
                "status": "success",
                "session_info": info
            }
            
        except Exception as e:
            logger.error(f"获取会话信息失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _find_simulation_file(self, simulation_id: str, file_type: str) -> Optional[Path]:
        """查找模拟文件"""
        # 可能的文件扩展名
        if file_type == "auto":
            extensions = ['.pvd', '.vtu', '.vtk', '.vtp']
        else:
            extensions = [f'.{file_type}']
        
        # 搜索文件
        for ext in extensions:
            # 直接文件名
            file_path = self.data_path / f"{simulation_id}{ext}"
            if file_path.exists():
                return file_path
            
            # 在子目录中查找
            pattern = f"*{simulation_id}*{ext}"
            for file_path in self.data_path.rglob(pattern):
                return file_path
        
        return None
    
    def _load_data_file(self, file_path: Path):
        """加载数据文件"""
        ext = file_path.suffix.lower()
        
        if ext == '.pvd':
            return simple.PVDReader(FileName=str(file_path))
        elif ext == '.vtu':
            return simple.XMLUnstructuredGridReader(FileName=str(file_path))
        elif ext == '.vtk':
            return simple.LegacyVTKReader(FileName=str(file_path))
        elif ext == '.vtp':
            return simple.XMLPolyDataReader(FileName=str(file_path))
        else:
            logger.error(f"不支持的文件格式: {ext}")
            return None
    
    def _setup_view(self):
        """设置视图"""
        if not self.active_view:
            return
        
        # 背景设置
        self.active_view.Background = self.view_settings.background_color
        self.active_view.Background2 = [0.9, 0.9, 0.9]
        self.active_view.UseGradientBackground = 0
        
        # 视图大小
        self.active_view.ViewSize = self.view_settings.view_size
        
        # 坐标轴
        self.active_view.OrientationAxesVisibility = 1
        self.active_view.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
        
        # 中心轴
        self.active_view.CenterAxesVisibility = 0
    
    def _get_available_fields(self) -> List[Dict[str, Any]]:
        """获取可用字段"""
        if not self.active_source:
            return []
        
        fields = []
        data_info = self.active_source.GetDataInformation()
        
        # 点数据
        point_data = data_info.GetPointDataInformation()
        for i in range(point_data.GetNumberOfArrays()):
            array_info = point_data.GetArrayInformation(i)
            fields.append({
                'name': array_info.GetName(),
                'location': 'POINTS',
                'components': array_info.GetNumberOfComponents(),
                'type': self._get_array_type(array_info),
                'range': list(array_info.GetComponentRange(0))
            })
        
        # 单元数据
        cell_data = data_info.GetCellDataInformation()
        for i in range(cell_data.GetNumberOfArrays()):
            array_info = cell_data.GetArrayInformation(i)
            fields.append({
                'name': array_info.GetName(),
                'location': 'CELLS',
                'components': array_info.GetNumberOfComponents(),
                'type': self._get_array_type(array_info),
                'range': list(array_info.GetComponentRange(0))
            })
        
        return fields
    
    def _get_array_type(self, array_info) -> str:
        """获取数组类型"""
        n_components = array_info.GetNumberOfComponents()
        if n_components == 1:
            return 'scalar'
        elif n_components == 3:
            return 'vector'
        elif n_components == 9:
            return 'tensor'
        else:
            return 'array'
    
    def _select_default_field(self, fields: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """选择默认字段"""
        # 优先级顺序
        priority_names = [
            'VonMises_MPa', 'VonMises_Stress_MPa', 'von_mises',
            'Temperature_Celsius', 'Temperature', 'temperature',
            'Displacement_Magnitude', 'Displacement'
        ]
        
        for name in priority_names:
            for field in fields:
                if field['name'].lower() == name.lower():
                    return field
        
        # 返回第一个标量场
        for field in fields:
            if field['type'] == 'scalar':
                return field
        
        return fields[0] if fields else None
    
    def _get_field_title(self, field_name: str) -> str:
        """获取字段标题"""
        titles = {
            'VonMises_MPa': 'von Mises应力 (MPa)',
            'VonMises_Stress_MPa': 'von Mises应力 (MPa)',
            'Temperature_Celsius': '温度 (°C)',
            'Temperature': '温度 (K)',
            'Displacement_Magnitude': '位移幅值 (mm)',
            'Displacement': '位移 (mm)',
            'Stress_XX_MPa': '应力σxx (MPa)',
            'Stress_YY_MPa': '应力σyy (MPa)',
            'Stress_XY_MPa': '应力σxy (MPa)'
        }
        return titles.get(field_name, field_name)
    
    def _update_color_bar(self, field_name: str, settings: ColorMapSettings):
        """更新颜色条"""
        color_bar = simple.GetScalarBar(
            simple.GetColorTransferFunction(field_name)
        )
        
        color_bar.Title = settings.color_bar_title
        color_bar.ComponentTitle = ''
        color_bar.Visibility = 1
        color_bar.Position = settings.color_bar_position
        color_bar.Position2 = [0.12, 0.5]
        color_bar.TitleFontSize = 14
        color_bar.LabelFontSize = 12
        color_bar.TitleColor = [0.0, 0.0, 0.0]
        color_bar.LabelColor = [0.0, 0.0, 0.0]
    
    def _get_field_range(self, field_name: str, location: str) -> List[float]:
        """获取字段范围"""
        if not self.active_source:
            return [0.0, 1.0]
        
        data_info = self.active_source.GetDataInformation()
        
        if location == "POINTS":
            array_info = data_info.GetPointDataInformation().GetArrayInformation(field_name)
        else:
            array_info = data_info.GetCellDataInformation().GetArrayInformation(field_name)
        
        if array_info:
            return list(array_info.GetComponentRange(0))
        
        return [0.0, 1.0]
    
    def _get_data_information(self) -> Dict[str, Any]:
        """获取数据信息"""
        if not self.active_source:
            return {}
        
        data_info = self.active_source.GetDataInformation()
        
        info = {
            'bounds': list(data_info.GetBounds()),
            'extent': list(data_info.GetExtent()),
            'n_points': data_info.GetNumberOfPoints(),
            'n_cells': data_info.GetNumberOfCells(),
            'n_arrays': data_info.GetNumberOfArrays(),
            'memory_size': data_info.GetMemorySize()
        }
        
        # 时间信息
        if hasattr(self.active_source, 'TimestepValues'):
            time_values = self.active_source.TimestepValues
            if time_values:
                info['time_steps'] = len(time_values)
                info['time_range'] = [float(time_values[0]), float(time_values[-1])]
        
        return info
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'paraview_version': simple.GetParaViewVersion(),
            'python_version': sys.version,
            'has_mpi': bool(simple.servermanager.vtkProcessModule.GetProcessModule().GetNumberOfLocalPartitions() > 1),
            'render_backend': self.active_view.GetRenderWindow().GetRenderingBackend() if self.active_view else 'unknown'
        }
    
    def _get_available_color_maps(self) -> List[str]:
        """获取可用的颜色映射"""
        # ParaView内置颜色映射
        return [
            'Cool to Warm', 'Blue to Red Rainbow', 'X Ray', 'Rainbow Uniform',
            'Jet', 'HSV', 'Viridis', 'Plasma', 'Inferno', 'Magma',
            'Black-Body Radiation', 'Grayscale', 'Cold and Hot'
        ]
    
    def _play_animation_async(self, animation_scene, frame_rate: int):
        """异步播放动画"""
        try:
            animation_scene.Play()
            
            # 等待动画结束
            while self.animation_playing and animation_scene.AnimationTime < animation_scene.EndTime:
                time.sleep(1.0 / frame_rate)
            
        finally:
            self.animation_playing = False
    
    def cleanup(self):
        """清理资源"""
        try:
            # 删除所有源
            simple.Delete(simple.GetSources())
            
            # 关闭线程池
            self.executor.shutdown(wait=True)
            
            logger.info("ParaviewWeb协议资源已清理")
            
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


def start_paraviewweb_server(
    host: str = "0.0.0.0",
    port: int = 9000,
    debug: bool = True
):
    """启动ParaviewWeb服务器"""
    
    if not HAS_WSLINK:
        logger.error("wslink未安装，无法启动服务器")
        print("请安装wslink: pip install wslink")
        return
    
    if not HAS_PARAVIEW:
        logger.error("ParaView Python未安装，功能将受限")
        print("请安装ParaView并配置Python环境")
    
    # 服务器参数
    args = {
        "host": host,
        "port": port,
        "ws": f"ws://{host}:{port}/ws",
        "lp": f"http://{host}:{port}",
        "content": "./www",
        "debug": debug,
        "timeout": 300,
        "nosignalhandlers": True
    }
    
    # 创建协议实例
    protocol = RollingVisualizationProtocol()
    
    logger.info(f"启动ParaviewWeb服务器 - {host}:{port}")
    
    # 启动服务器
    server.start_webserver(options=args, protocol=protocol)


def test_protocol():
    """测试协议功能"""
    protocol = RollingVisualizationProtocol()
    
    # 测试初始化
    result = protocol.initialize()
    print(f"初始化: {result}")
    
    # 测试加载（需要有测试数据）
    # result = protocol.load_simulation("test_001")
    # print(f"加载: {result}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ParaviewWeb可视化服务器")
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=9000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--test', action='store_true', help='测试模式')
    
    args = parser.parse_args()
    
    if args.test:
        test_protocol()
    else:
        start_paraviewweb_server(args.host, args.port, args.debug)
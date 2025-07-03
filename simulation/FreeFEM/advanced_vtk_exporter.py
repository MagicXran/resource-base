# -*- coding: gb2312 -*-
"""
高级VTK导出器 - 从FreeFEM结果生成VTK文件
支持时间序列、多物理场、网格质量分析
"""

import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from pathlib import Path
import json
import logging
import h5py
import time
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import meshio
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# 忽略VTK警告
warnings.filterwarnings('ignore', module='vtk')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='gb2312'
)
logger = logging.getLogger(__name__)


@dataclass
class VTKExportOptions:
    """VTK导出选项"""
    binary_format: bool = True
    compression: bool = True
    compression_level: int = 6
    export_mesh_quality: bool = True
    export_field_statistics: bool = True
    simplify_for_web: bool = False
    simplification_factor: float = 0.5
    parallel_export: bool = True
    max_workers: int = 4


class VTKFieldConverter:
    """VTK场数据转换器"""
    
    @staticmethod
    def scalar_to_vtk(data: np.ndarray, name: str) -> vtk.vtkDataArray:
        """标量场转VTK数组"""
        vtk_array = numpy_to_vtk(data.ravel())
        vtk_array.SetName(name)
        return vtk_array
    
    @staticmethod
    def vector_to_vtk(data: np.ndarray, name: str, dim: int = 3) -> vtk.vtkDataArray:
        """矢量场转VTK数组"""
        # 确保是3D矢量
        if data.shape[1] == 2 and dim == 3:
            # 2D转3D，添加零分量
            data_3d = np.zeros((data.shape[0], 3))
            data_3d[:, :2] = data
            data = data_3d
        
        vtk_array = numpy_to_vtk(data)
        vtk_array.SetName(name)
        vtk_array.SetNumberOfComponents(dim)
        return vtk_array
    
    @staticmethod
    def tensor_to_vtk(data: np.ndarray, name: str) -> vtk.vtkDataArray:
        """张量场转VTK数组"""
        # 2D张量转换为3x3对称张量
        if data.shape[1] == 3:  # [s11, s22, s12]
            tensor_3x3 = np.zeros((data.shape[0], 9))
            tensor_3x3[:, 0] = data[:, 0]  # s11
            tensor_3x3[:, 4] = data[:, 1]  # s22
            tensor_3x3[:, 8] = 0           # s33
            tensor_3x3[:, 1] = tensor_3x3[:, 3] = data[:, 2]  # s12 = s21
            data = tensor_3x3
        
        vtk_array = numpy_to_vtk(data)
        vtk_array.SetName(name)
        vtk_array.SetNumberOfComponents(9)
        return vtk_array


class MeshQualityAnalyzer:
    """网格质量分析器"""
    
    def __init__(self, grid: vtk.vtkUnstructuredGrid):
        self.grid = grid
        self.quality_metrics = {}
    
    def compute_all_metrics(self) -> Dict[str, vtk.vtkDataArray]:
        """计算所有质量指标"""
        quality = vtk.vtkMeshQuality()
        quality.SetInputData(self.grid)
        
        # 三角形质量指标
        metrics = {
            'AspectRatio': vtk.VTK_QUALITY_ASPECT_RATIO,
            'EdgeRatio': vtk.VTK_QUALITY_EDGE_RATIO,
            'RadiusRatio': vtk.VTK_QUALITY_RADIUS_RATIO,
            'MinAngle': vtk.VTK_QUALITY_MIN_ANGLE,
            'MaxAngle': vtk.VTK_QUALITY_MAX_ANGLE,
            'Area': vtk.VTK_QUALITY_AREA
        }
        
        results = {}
        
        for metric_name, metric_type in metrics.items():
            quality.SetTriangleQualityMeasure(metric_type)
            quality.Update()
            
            quality_array = quality.GetOutput().GetCellData().GetArray("Quality")
            if quality_array:
                # 复制数组并重命名
                new_array = vtk.vtkFloatArray()
                new_array.DeepCopy(quality_array)
                new_array.SetName(f"Quality_{metric_name}")
                results[metric_name] = new_array
                
                # 计算统计信息
                values = vtk_to_numpy(quality_array)
                self.quality_metrics[metric_name] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        
        return results
    
    def add_quality_fields_to_grid(self, grid: vtk.vtkUnstructuredGrid):
        """添加质量场到网格"""
        quality_arrays = self.compute_all_metrics()
        
        for array in quality_arrays.values():
            grid.GetCellData().AddArray(array)
        
        logger.info(f"已添加 {len(quality_arrays)} 个网格质量指标")
    
    def get_quality_report(self) -> Dict[str, Any]:
        """获取质量报告"""
        return {
            'metrics': self.quality_metrics,
            'summary': {
                'total_cells': self.grid.GetNumberOfCells(),
                'cell_types': self._get_cell_type_counts()
            }
        }
    
    def _get_cell_type_counts(self) -> Dict[str, int]:
        """统计单元类型"""
        type_counts = {}
        
        for i in range(self.grid.GetNumberOfCells()):
            cell_type = self.grid.GetCellType(i)
            type_name = vtk.vtkCellTypes.GetClassNameFromTypeId(cell_type)
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return type_counts


class AdvancedVTKExporter:
    """高级VTK导出器"""
    
    def __init__(
        self,
        output_dir: str = "vtk_output",
        options: Optional[VTKExportOptions] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.options = options or VTKExportOptions()
        
        # 线程池（用于并行导出）
        self.executor = None
        if self.options.parallel_export:
            self.executor = ThreadPoolExecutor(max_workers=self.options.max_workers)
        
        logger.info(f"VTK导出器初始化 - 输出目录: {self.output_dir}")
    
    def create_unstructured_grid_from_data(
        self,
        vertices: np.ndarray,
        cells: np.ndarray,
        cell_type: str = "triangle"
    ) -> vtk.vtkUnstructuredGrid:
        """从数据创建非结构网格"""
        # 创建点
        points = vtk.vtkPoints()
        
        # 处理2D/3D坐标
        if vertices.shape[1] == 2:
            # 2D坐标，添加z=0
            for vertex in vertices:
                points.InsertNextPoint(vertex[0], vertex[1], 0.0)
        else:
            for vertex in vertices:
                points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
        
        # 创建单元
        vtk_cells = vtk.vtkCellArray()
        
        if cell_type == "triangle":
            for cell in cells:
                triangle = vtk.vtkTriangle()
                for i in range(3):
                    triangle.GetPointIds().SetId(i, int(cell[i]))
                vtk_cells.InsertNextCell(triangle)
            cell_type_id = vtk.VTK_TRIANGLE
        elif cell_type == "quad":
            for cell in cells:
                quad = vtk.vtkQuad()
                for i in range(4):
                    quad.GetPointIds().SetId(i, int(cell[i]))
                vtk_cells.InsertNextCell(quad)
            cell_type_id = vtk.VTK_QUAD
        elif cell_type == "tetra":
            for cell in cells:
                tetra = vtk.vtkTetra()
                for i in range(4):
                    tetra.GetPointIds().SetId(i, int(cell[i]))
                vtk_cells.InsertNextCell(tetra)
            cell_type_id = vtk.VTK_TETRA
        else:
            raise ValueError(f"不支持的单元类型: {cell_type}")
        
        # 创建网格
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        grid.SetCells(cell_type_id, vtk_cells)
        
        return grid
    
    def add_field_data(
        self,
        grid: vtk.vtkUnstructuredGrid,
        field_name: str,
        field_data: np.ndarray,
        field_type: str = "scalar",
        location: str = "point"
    ):
        """添加场数据到网格"""
        # 数据转换
        if field_type == "scalar":
            vtk_array = VTKFieldConverter.scalar_to_vtk(field_data, field_name)
        elif field_type == "vector":
            vtk_array = VTKFieldConverter.vector_to_vtk(field_data, field_name)
        elif field_type == "tensor":
            vtk_array = VTKFieldConverter.tensor_to_vtk(field_data, field_name)
        else:
            raise ValueError(f"不支持的场类型: {field_type}")
        
        # 添加到网格
        if location == "point":
            grid.GetPointData().AddArray(vtk_array)
        elif location == "cell":
            grid.GetCellData().AddArray(vtk_array)
        else:
            raise ValueError(f"不支持的位置: {location}")
        
        logger.debug(f"已添加场数据: {field_name} ({field_type}, {location})")
    
    def export_single_timestep(
        self,
        mesh_data: Dict[str, np.ndarray],
        field_data: Dict[str, Any],
        filename: str,
        add_derived_fields: bool = True
    ) -> str:
        """导出单个时间步"""
        # 创建网格
        grid = self.create_unstructured_grid_from_data(
            mesh_data['vertices'],
            mesh_data['elements'],
            mesh_data.get('cell_type', 'triangle')
        )
        
        # 添加原始场数据
        for field_name, field_info in field_data.items():
            if isinstance(field_info, dict):
                self.add_field_data(
                    grid,
                    field_name,
                    field_info['data'],
                    field_info.get('type', 'scalar'),
                    field_info.get('location', 'point')
                )
            else:
                # 假设是标量点数据
                self.add_field_data(grid, field_name, field_info)
        
        # 添加导出字段
        if add_derived_fields:
            self._add_derived_fields(grid, field_data)
        
        # 添加网格质量
        if self.options.export_mesh_quality:
            analyzer = MeshQualityAnalyzer(grid)
            analyzer.add_quality_fields_to_grid(grid)
        
        # 写入文件
        output_path = self.output_dir / filename
        self._write_grid(grid, output_path)
        
        return str(output_path)
    
    def export_time_series(
        self,
        mesh_data: Dict[str, np.ndarray],
        time_steps: List[Dict[str, Any]],
        base_name: str = "simulation",
        create_pvd: bool = True
    ) -> List[str]:
        """导出时间序列"""
        logger.info(f"开始导出时间序列: {len(time_steps)} 个时间步")
        
        output_files = []
        
        if self.options.parallel_export and self.executor:
            # 并行导出
            futures = []
            
            for idx, step_data in enumerate(time_steps):
                filename = f"{base_name}_{idx:04d}.vtu"
                future = self.executor.submit(
                    self._export_single_timestep_wrapper,
                    mesh_data,
                    step_data['fields'],
                    filename,
                    idx,
                    len(time_steps)
                )
                futures.append((future, filename, step_data.get('time', idx)))
            
            # 收集结果
            for future, filename, time_value in futures:
                try:
                    output_path = future.result()
                    output_files.append((output_path, time_value))
                except Exception as e:
                    logger.error(f"导出失败 {filename}: {e}")
        
        else:
            # 串行导出
            for idx, step_data in enumerate(time_steps):
                filename = f"{base_name}_{idx:04d}.vtu"
                try:
                    output_path = self.export_single_timestep(
                        mesh_data,
                        step_data['fields'],
                        filename
                    )
                    output_files.append((output_path, step_data.get('time', idx)))
                    
                    # 进度显示
                    if idx % 10 == 0:
                        logger.info(f"导出进度: {idx+1}/{len(time_steps)}")
                        
                except Exception as e:
                    logger.error(f"导出时间步 {idx} 失败: {e}")
        
        # 创建PVD文件
        if create_pvd and output_files:
            pvd_file = self._create_pvd_file(output_files, base_name)
            logger.info(f"创建PVD文件: {pvd_file}")
        
        logger.info(f"时间序列导出完成: {len(output_files)} 个文件")
        return [f[0] for f in output_files]
    
    def _export_single_timestep_wrapper(
        self,
        mesh_data: Dict[str, np.ndarray],
        field_data: Dict[str, Any],
        filename: str,
        idx: int,
        total: int
    ) -> str:
        """单时间步导出包装函数（用于并行）"""
        output_path = self.export_single_timestep(mesh_data, field_data, filename)
        if idx % 10 == 0:
            logger.info(f"并行导出进度: {idx+1}/{total}")
        return output_path
    
    def _add_derived_fields(self, grid: vtk.vtkUnstructuredGrid, field_data: Dict[str, Any]):
        """添加导出场"""
        # 温度转换
        if 'temperature' in field_data:
            temp_k = field_data['temperature']
            if isinstance(temp_k, dict):
                temp_k = temp_k['data']
            temp_c = temp_k - 273.15
            self.add_field_data(grid, 'Temperature_Celsius', temp_c)
        
        # 应力单位转换
        if 'von_mises' in field_data:
            stress_pa = field_data['von_mises']
            if isinstance(stress_pa, dict):
                stress_pa = stress_pa['data']
            stress_mpa = stress_pa / 1e6
            self.add_field_data(grid, 'VonMises_MPa', stress_mpa)
        
        # 位移幅值
        if 'displacement' in field_data:
            disp = field_data['displacement']
            if isinstance(disp, dict):
                disp_data = disp['data']
            else:
                disp_data = disp
            
            if disp_data.ndim == 2:
                disp_mag = np.sqrt(np.sum(disp_data**2, axis=1))
                self.add_field_data(grid, 'Displacement_Magnitude', disp_mag)
                
                # 位移毫米单位
                disp_mm = disp_data * 1000
                self.add_field_data(
                    grid, 'Displacement_mm', disp_mm,
                    field_type='vector'
                )
    
    def _write_grid(self, grid: vtk.vtkUnstructuredGrid, output_path: Path):
        """写入网格文件"""
        # 根据文件扩展名选择写入器
        ext = output_path.suffix.lower()
        
        if ext == '.vtu':
            writer = vtk.vtkXMLUnstructuredGridWriter()
        elif ext == '.vtk':
            writer = vtk.vtkUnstructuredGridWriter()
            if self.options.binary_format:
                writer.SetFileTypeToBinary()
            else:
                writer.SetFileTypeToASCII()
        elif ext == '.vtp':
            # 转换为PolyData
            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputData(grid)
            geometry_filter.Update()
            
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetInputData(geometry_filter.GetOutput())
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        # 通用设置
        writer.SetFileName(str(output_path))
        
        # XML格式特定设置
        if hasattr(writer, 'SetDataModeToBinary'):
            if self.options.binary_format:
                writer.SetDataModeToBinary()
            else:
                writer.SetDataModeToAscii()
        
        if hasattr(writer, 'SetCompressorTypeToZLib') and self.options.compression:
            writer.SetCompressorTypeToZLib()
            writer.SetCompressionLevel(self.options.compression_level)
        
        # 写入
        writer.Write()
        
        # 验证文件
        if not output_path.exists():
            raise IOError(f"文件写入失败: {output_path}")
    
    def _create_pvd_file(
        self,
        file_list: List[Tuple[str, float]],
        base_name: str
    ) -> str:
        """创建PVD时间序列文件"""
        pvd_file = self.output_dir / f"{base_name}.pvd"
        
        content = ['<?xml version="1.0"?>\n']
        content.append('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        content.append('  <Collection>\n')
        
        for file_path, time_value in file_list:
            rel_path = Path(file_path).name
            content.append(
                f'    <DataSet timestep="{time_value}" group="" part="0" '
                f'file="{rel_path}"/>\n'
            )
        
        content.append('  </Collection>\n')
        content.append('</VTKFile>\n')
        
        with open(pvd_file, 'w') as f:
            f.writelines(content)
        
        return str(pvd_file)
    
    def export_freefem_results(
        self,
        freefem_results: Dict[str, Any],
        task_id: str,
        export_history: bool = True
    ) -> Dict[str, Any]:
        """导出FreeFEM计算结果"""
        export_info = {
            'task_id': task_id,
            'files': [],
            'statistics': {}
        }
        
        try:
            # 提取网格数据
            if 'mesh' not in freefem_results:
                raise ValueError("结果中缺少网格数据")
            
            mesh_obj = freefem_results['mesh']
            mesh_data = {
                'vertices': mesh_obj.vertices if hasattr(mesh_obj, 'vertices') else mesh_obj['vertices'],
                'elements': mesh_obj.elements if hasattr(mesh_obj, 'elements') else mesh_obj['elements']
            }
            
            # 导出最终状态
            if 'fields' in freefem_results:
                final_file = self.export_single_timestep(
                    mesh_data,
                    freefem_results['fields'],
                    f"{task_id}_final.vtu"
                )
                export_info['files'].append(final_file)
                logger.info(f"导出最终状态: {final_file}")
            
            # 导出VTK时间序列（如果存在）
            if 'vtk_files' in freefem_results:
                vtk_files = freefem_results['vtk_files']
                self._convert_vtk_series_to_vtu(vtk_files, task_id)
                export_info['files'].extend([f"{task_id}_{i:04d}.vtu" for i in range(len(vtk_files))])
            
            # 导出历史数据
            if export_history and 'history' in freefem_results:
                history_file = self._export_history_data(
                    freefem_results['history'],
                    task_id
                )
                export_info['history_file'] = history_file
            
            # 导出网格质量报告
            if self.options.export_mesh_quality:
                grid = self.create_unstructured_grid_from_data(
                    mesh_data['vertices'],
                    mesh_data['elements']
                )
                analyzer = MeshQualityAnalyzer(grid)
                quality_report = analyzer.get_quality_report()
                
                report_file = self.output_dir / f"{task_id}_mesh_quality.json"
                with open(report_file, 'w') as f:
                    json.dump(quality_report, f, indent=2)
                
                export_info['quality_report'] = str(report_file)
                export_info['statistics']['mesh_quality'] = quality_report['metrics']
            
            # 场数据统计
            if self.options.export_field_statistics and 'fields' in freefem_results:
                field_stats = {}
                for field_name, field_data in freefem_results['fields'].items():
                    if hasattr(field_data, 'get_statistics'):
                        field_stats[field_name] = field_data.get_statistics()
                    elif isinstance(field_data, np.ndarray):
                        field_stats[field_name] = {
                            'min': float(field_data.min()),
                            'max': float(field_data.max()),
                            'mean': float(field_data.mean()),
                            'std': float(field_data.std())
                        }
                
                export_info['statistics']['fields'] = field_stats
            
            # 保存导出信息
            info_file = self.output_dir / f"{task_id}_export_info.json"
            with open(info_file, 'w') as f:
                json.dump(export_info, f, indent=2)
            
            logger.info(f"FreeFEM结果导出完成: {len(export_info['files'])} 个文件")
            return export_info
            
        except Exception as e:
            logger.error(f"导出FreeFEM结果失败: {e}")
            raise
    
    def _convert_vtk_series_to_vtu(self, vtk_files: List[str], task_id: str):
        """转换VTK系列文件为VTU"""
        output_files = []
        
        for idx, vtk_file in enumerate(vtk_files):
            try:
                # 读取VTK文件
                reader = vtk.vtkUnstructuredGridReader()
                reader.SetFileName(vtk_file)
                reader.Update()
                
                grid = reader.GetOutput()
                
                # 写入VTU文件
                vtu_filename = f"{task_id}_{idx:04d}.vtu"
                output_path = self.output_dir / vtu_filename
                self._write_grid(grid, output_path)
                
                output_files.append((str(output_path), idx * 0.001))
                
            except Exception as e:
                logger.error(f"转换VTK文件失败 {vtk_file}: {e}")
        
        # 创建PVD文件
        if output_files:
            self._create_pvd_file(output_files, task_id)
    
    def _export_history_data(self, history_data: Dict[str, Any], task_id: str) -> str:
        """导出历史数据为CSV格式"""
        import csv
        
        csv_file = self.output_dir / f"{task_id}_history.csv"
        
        # 收集所有时间步
        all_times = set()
        for data in history_data.values():
            if 'time' in data:
                all_times.update(data['time'])
        
        all_times = sorted(all_times)
        
        # 准备数据表
        rows = []
        headers = ['time']
        
        # 收集所有字段
        for name, data in history_data.items():
            for key in data.keys():
                if key != 'time':
                    headers.append(f"{name}_{key}")
        
        # 填充数据
        for t in all_times:
            row = [t]
            
            for name, data in history_data.items():
                if 'time' in data:
                    # 找到对应时间的索引
                    time_array = np.array(data['time'])
                    idx = np.argmin(np.abs(time_array - t))
                    
                    for key, values in data.items():
                        if key != 'time':
                            row.append(values[idx] if idx < len(values) else np.nan)
                else:
                    # 无时间数据，填充NaN
                    for key in data.keys():
                        row.append(np.nan)
            
            rows.append(row)
        
        # 写入CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        logger.info(f"历史数据已导出: {csv_file}")
        return str(csv_file)
    
    def create_simplified_mesh_for_web(
        self,
        grid: vtk.vtkUnstructuredGrid,
        reduction_factor: float = 0.5
    ) -> vtk.vtkPolyData:
        """创建简化网格用于Web显示"""
        # 转换为PolyData
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputData(grid)
        geometry_filter.Update()
        
        poly_data = geometry_filter.GetOutput()
        
        # 简化网格
        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputData(poly_data)
        decimate.SetTargetReduction(reduction_factor)
        decimate.Update()
        
        simplified = decimate.GetOutput()
        
        logger.info(
            f"网格简化: {poly_data.GetNumberOfPoints()} -> "
            f"{simplified.GetNumberOfPoints()} 点"
        )
        
        return simplified
    
    def export_for_paraviewweb(
        self,
        mesh_data: Dict[str, np.ndarray],
        field_data: Dict[str, Any],
        task_id: str
    ) -> Dict[str, str]:
        """导出专门用于ParaviewWeb的数据"""
        web_files = {}
        
        try:
            # 创建完整网格
            full_grid = self.create_unstructured_grid_from_data(
                mesh_data['vertices'],
                mesh_data['elements']
            )
            
            # 添加场数据
            for field_name, field_info in field_data.items():
                if isinstance(field_info, dict):
                    self.add_field_data(
                        full_grid,
                        field_name,
                        field_info['data'],
                        field_info.get('type', 'scalar'),
                        field_info.get('location', 'point')
                    )
            
            # 导出完整数据
            full_file = f"{task_id}_full.vtu"
            full_path = self.output_dir / full_file
            self._write_grid(full_grid, full_path)
            web_files['full'] = str(full_path)
            
            # 创建简化版本
            if self.options.simplify_for_web:
                simplified = self.create_simplified_mesh_for_web(
                    full_grid,
                    self.options.simplification_factor
                )
                
                simple_file = f"{task_id}_simplified.vtp"
                simple_path = self.output_dir / simple_file
                
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetFileName(str(simple_path))
                writer.SetInputData(simplified)
                writer.SetDataModeToBinary()
                writer.SetCompressorTypeToZLib()
                writer.Write()
                
                web_files['simplified'] = str(simple_path)
            
            # 创建元数据
            metadata = {
                'task_id': task_id,
                'mesh_info': {
                    'n_points': full_grid.GetNumberOfPoints(),
                    'n_cells': full_grid.GetNumberOfCells()
                },
                'fields': list(field_data.keys()),
                'files': web_files
            }
            
            meta_file = self.output_dir / f"{task_id}_metadata.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            web_files['metadata'] = str(meta_file)
            
            logger.info(f"ParaviewWeb数据导出完成: {task_id}")
            return web_files
            
        except Exception as e:
            logger.error(f"ParaviewWeb导出失败: {e}")
            raise
    
    def export_to_hdf5(
        self,
        mesh_data: Dict[str, np.ndarray],
        time_series_data: List[Dict[str, Any]],
        filename: str
    ):
        """导出为HDF5格式（用于大数据集）"""
        h5_file = self.output_dir / filename
        
        with h5py.File(h5_file, 'w') as f:
            # 保存网格
            mesh_group = f.create_group('mesh')
            for key, data in mesh_data.items():
                mesh_group.create_dataset(key, data=data, compression='gzip')
            
            # 保存时间序列
            time_group = f.create_group('time_series')
            time_values = [step.get('time', i) for i, step in enumerate(time_series_data)]
            time_group.create_dataset('time', data=time_values)
            
            # 保存场数据
            for field_name in time_series_data[0]['fields'].keys():
                field_group = time_group.create_group(field_name)
                
                # 收集所有时间步的数据
                field_data = []
                for step in time_series_data:
                    if field_name in step['fields']:
                        data = step['fields'][field_name]
                        if isinstance(data, dict):
                            data = data['data']
                        field_data.append(data)
                
                # 保存为3D数组 (time, points, components)
                field_array = np.array(field_data)
                field_group.create_dataset(
                    'data',
                    data=field_array,
                    compression='gzip',
                    compression_opts=self.options.compression_level
                )
            
            # 保存元数据
            f.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
            f.attrs['n_timesteps'] = len(time_series_data)
            f.attrs['fields'] = list(time_series_data[0]['fields'].keys())
        
        logger.info(f"HDF5文件已保存: {h5_file}")
    
    def create_animation_script(self, pvd_file: str, output_name: str = "animation"):
        """生成ParaView动画脚本"""
        script_content = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ParaView animation script
# Generated by AdvancedVTKExporter

from paraview.simple import *

# 清理
Delete(GetSources().values())

# 加载数据
data = PVDReader(FileName='{pvd_file}')

# 创建视图
renderView = GetActiveViewOrCreate('RenderView')
renderView.ViewSize = [1920, 1080]
renderView.Background = [1.0, 1.0, 1.0]

# 显示
display = Show(data, renderView)
display.Representation = 'Surface'

# 着色
ColorBy(display, ('POINTS', 'VonMises_MPa'))
display.RescaleTransferFunctionToDataRange(True)

# 颜色条
colorBar = GetScalarBar(GetColorTransferFunction('VonMises_MPa'))
colorBar.Title = 'von Mises Stress (MPa)'
colorBar.Visibility = 1

# 相机设置
camera = renderView.GetActiveCamera()
camera.SetPosition(0.2, 0.1, 0.5)
camera.SetFocalPoint(0.0, 0.0, 0.0)
camera.SetViewUp(0.0, 1.0, 0.0)
renderView.ResetCamera()

# 动画设置
animationScene = GetAnimationScene()
animationScene.PlayMode = 'Sequence'
animationScene.NumberOfFrames = 100

# 保存动画
SaveAnimation('{output_name}.avi', renderView,
              FrameRate=15,
              FrameWindow=[0, 99])

print("Animation saved: {output_name}.avi")
'''
        
        script_file = self.output_dir / f"{output_name}_animation.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        logger.info(f"动画脚本已创建: {script_file}")
    
    def cleanup(self):
        """清理资源"""
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("线程池已关闭")


def test_vtk_exporter():
    """测试VTK导出器"""
    # 创建测试数据
    # 简单矩形网格
    nx, ny = 50, 10
    x = np.linspace(0, 0.1, nx)
    y = np.linspace(0, 0.02, ny)
    xx, yy = np.meshgrid(x, y)
    
    vertices = np.column_stack([xx.ravel(), yy.ravel()])
    
    # 创建三角形单元
    elements = []
    for i in range(ny-1):
        for j in range(nx-1):
            n1 = i * nx + j
            n2 = n1 + 1
            n3 = n1 + nx
            n4 = n3 + 1
            
            elements.append([n1, n2, n3])
            elements.append([n2, n4, n3])
    
    elements = np.array(elements)
    
    mesh_data = {
        'vertices': vertices,
        'elements': elements
    }
    
    # 创建测试场数据
    n_points = len(vertices)
    temperature = 1123 + 100 * np.random.rand(n_points)
    stress = 200e6 + 50e6 * np.random.rand(n_points)
    displacement = 0.001 * np.random.rand(n_points, 2)
    
    field_data = {
        'temperature': temperature,
        'von_mises': stress,
        'displacement': {
            'data': displacement,
            'type': 'vector'
        }
    }
    
    # 测试导出
    exporter = AdvancedVTKExporter(output_dir="test_vtk_output")
    
    # 单时间步
    print("测试单时间步导出...")
    output_file = exporter.export_single_timestep(
        mesh_data,
        field_data,
        "test_single.vtu"
    )
    print(f"输出文件: {output_file}")
    
    # 时间序列
    print("\n测试时间序列导出...")
    time_steps = []
    for i in range(10):
        step_fields = {
            'temperature': temperature + 10 * i,
            'von_mises': stress * (1 + 0.1 * np.sin(i * 0.5)),
            'displacement': {
                'data': displacement * (1 + 0.05 * i),
                'type': 'vector'
            }
        }
        time_steps.append({
            'time': i * 0.001,
            'fields': step_fields
        })
    
    output_files = exporter.export_time_series(
        mesh_data,
        time_steps,
        "test_series"
    )
    print(f"导出 {len(output_files)} 个文件")
    
    # 清理
    exporter.cleanup()
    print("\n测试完成!")


if __name__ == "__main__":
    test_vtk_exporter()
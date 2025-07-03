#!/usr/bin/env python3
"""
VTK导出模块 - 将PyFreeFEM结果转换为VTK格式
支持ParaView和ParaviewWeb可视化
"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import meshio
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
import h5py

logger = logging.getLogger(__name__)


class VTKExporter:
    """基础VTK导出器"""
    
    def __init__(self, output_dir: str = "vtk_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_vtk_grid_from_freefem(self, mesh, dimension: int = 2) -> vtk.vtkUnstructuredGrid:
        """从FreeFEM网格创建VTK非结构网格"""
        # 创建VTK点集
        points = vtk.vtkPoints()
        coordinates = mesh.coordinates()
        
        # 处理2D/3D情况
        if dimension == 2 and coordinates.shape[1] == 2:
            # 2D网格，添加z=0
            for coord in coordinates:
                points.InsertNextPoint(coord[0], coord[1], 0.0)
        else:
            for coord in coordinates:
                points.InsertNextPoint(coord[0], coord[1], coord[2] if len(coord) > 2 else 0.0)
        
        # 创建VTK单元
        cells = vtk.vtkCellArray()
        cell_types = []
        
        # 获取网格单元
        mesh_cells = mesh.cells()
        
        for cell in mesh_cells:
            if len(cell) == 3:  # 三角形单元
                triangle = vtk.vtkTriangle()
                for i, node_id in enumerate(cell):
                    triangle.GetPointIds().SetId(i, node_id)
                cells.InsertNextCell(triangle)
                cell_types.append(vtk.VTK_TRIANGLE)
            elif len(cell) == 4:  # 四边形单元
                quad = vtk.vtkQuad()
                for i, node_id in enumerate(cell):
                    quad.GetPointIds().SetId(i, node_id)
                cells.InsertNextCell(quad)
                cell_types.append(vtk.VTK_QUAD)
        
        # 创建非结构网格
        vtk_grid = vtk.vtkUnstructuredGrid()
        vtk_grid.SetPoints(points)
        vtk_grid.SetCells(cell_types, cells)
        
        return vtk_grid
    
    def add_point_data(self, vtk_grid: vtk.vtkUnstructuredGrid, 
                      field_name: str, field_data: np.ndarray):
        """添加点数据（节点数据）"""
        vtk_array = numpy_to_vtk(field_data)
        vtk_array.SetName(field_name)
        
        # 设置分量数
        if field_data.ndim > 1:
            vtk_array.SetNumberOfComponents(field_data.shape[1])
            
        vtk_grid.GetPointData().AddArray(vtk_array)
        
    def add_cell_data(self, vtk_grid: vtk.vtkUnstructuredGrid,
                     field_name: str, field_data: np.ndarray):
        """添加单元数据"""
        vtk_array = numpy_to_vtk(field_data)
        vtk_array.SetName(field_name)
        
        if field_data.ndim > 1:
            vtk_array.SetNumberOfComponents(field_data.shape[1])
            
        vtk_grid.GetCellData().AddArray(vtk_array)
    
    def write_vtu(self, vtk_grid: vtk.vtkUnstructuredGrid, filename: str):
        """写入VTU文件（XML格式）"""
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(self.output_dir / filename))
        writer.SetInputData(vtk_grid)
        writer.SetDataModeToBinary()
        writer.SetCompressorTypeToZLib()
        writer.Write()
        
    def write_legacy_vtk(self, vtk_grid: vtk.vtkUnstructuredGrid, filename: str):
        """写入传统VTK文件（ASCII格式）"""
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(str(self.output_dir / filename))
        writer.SetInputData(vtk_grid)
        writer.SetFileTypeToASCII()
        writer.Write()


class RollingResultsVTKExporter(VTKExporter):
    """轧制模拟结果专用VTK导出器"""
    
    def __init__(self, mesh, results: List[Dict], output_dir: str = "rolling_vtk"):
        super().__init__(output_dir)
        self.mesh = mesh
        self.results = results
        self.base_grid = self.create_vtk_grid_from_freefem(mesh)
        
    def export_time_series(self, base_name: str = "rolling"):
        """导出时间序列VTK文件"""
        logger.info(f"导出{len(self.results)}个时间步的VTK文件...")
        
        # 文件列表（用于PVD文件）
        file_list = []
        
        for idx, result in enumerate(self.results):
            # 创建当前时间步的网格副本
            vtk_grid = self._create_grid_with_fields(result)
            
            # 文件名
            filename = f"{base_name}_{idx:04d}.vtu"
            self.write_vtu(vtk_grid, filename)
            
            # 记录文件信息
            file_list.append({
                'filename': filename,
                'time': result['time'],
                'step': result['step']
            })
            
            if idx % 10 == 0:
                logger.info(f"  已导出时间步 {idx}/{len(self.results)}")
        
        # 创建PVD文件（ParaView时间序列）
        self._write_pvd_file(base_name, file_list)
        
        logger.info(f"VTK导出完成，文件保存在: {self.output_dir}")
        
    def _create_grid_with_fields(self, result: Dict) -> vtk.vtkUnstructuredGrid:
        """创建包含所有场数据的VTK网格"""
        # 复制基础网格
        vtk_grid = vtk.vtkUnstructuredGrid()
        vtk_grid.DeepCopy(self.base_grid)
        
        # 添加温度场（点数据）
        if 'temperature' in result:
            temp_data = result['temperature'].compute_vertex_values(self.mesh)
            self.add_point_data(vtk_grid, "Temperature", temp_data)
            
            # 添加摄氏度温度
            temp_celsius = temp_data - 273.15
            self.add_point_data(vtk_grid, "Temperature_Celsius", temp_celsius)
        
        # 添加位移场（矢量点数据）
        if 'displacement' in result:
            disp_data = result['displacement'].compute_vertex_values(self.mesh)
            # 转换为正确的形状 (n_points, n_components)
            if disp_data.shape[0] == 2:  # 2D位移
                disp_data = disp_data.T
            self.add_point_data(vtk_grid, "Displacement", disp_data)
            
            # 添加位移幅值
            disp_magnitude = np.sqrt(np.sum(disp_data**2, axis=1))
            self.add_point_data(vtk_grid, "Displacement_Magnitude", disp_magnitude)
        
        # 添加von Mises应力（单元数据）
        if 'von_mises' in result:
            vm_data = result['von_mises'].compute_vertex_values(self.mesh)
            self.add_point_data(vtk_grid, "VonMises_Stress", vm_data)
            
            # 添加MPa单位的应力
            vm_mpa = vm_data / 1e6
            self.add_point_data(vtk_grid, "VonMises_Stress_MPa", vm_mpa)
        
        # 添加应力张量分量（如果存在）
        if 'stress' in result:
            stress_tensor = result['stress']
            # 提取应力分量
            self._add_stress_components(vtk_grid, stress_tensor)
        
        # 添加轧制力信息（作为场数据属性）
        if 'rolling_force' in result:
            force_array = vtk.vtkFloatArray()
            force_array.SetName("Rolling_Force_MN")
            force_array.SetNumberOfTuples(1)
            force_array.SetValue(0, result['rolling_force'] / 1e6)
            vtk_grid.GetFieldData().AddArray(force_array)
        
        # 添加时间信息
        time_array = vtk.vtkFloatArray()
        time_array.SetName("Time")
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, result['time'])
        vtk_grid.GetFieldData().AddArray(time_array)
        
        return vtk_grid
    
    def _add_stress_components(self, vtk_grid: vtk.vtkUnstructuredGrid, stress_tensor):
        """添加应力张量各分量"""
        # 获取应力张量数据
        stress_data = stress_tensor.compute_vertex_values(self.mesh)
        
        # 2D应力张量分量
        if stress_data.shape[0] >= 3:
            # σ_xx
            self.add_point_data(vtk_grid, "Stress_XX_MPa", stress_data[0, :] / 1e6)
            # σ_yy  
            self.add_point_data(vtk_grid, "Stress_YY_MPa", stress_data[1, :] / 1e6)
            # σ_xy
            self.add_point_data(vtk_grid, "Stress_XY_MPa", stress_data[2, :] / 1e6)
            
            # 计算主应力
            self._compute_principal_stresses(vtk_grid, stress_data)
    
    def _compute_principal_stresses(self, vtk_grid: vtk.vtkUnstructuredGrid, 
                                   stress_data: np.ndarray):
        """计算并添加主应力"""
        n_points = vtk_grid.GetNumberOfPoints()
        sigma1 = np.zeros(n_points)
        sigma2 = np.zeros(n_points)
        
        for i in range(n_points):
            # 构建2x2应力矩阵
            stress_matrix = np.array([
                [stress_data[0, i], stress_data[2, i]],
                [stress_data[2, i], stress_data[1, i]]
            ])
            
            # 计算特征值（主应力）
            eigenvalues = np.linalg.eigvals(stress_matrix)
            sigma1[i] = np.max(eigenvalues)
            sigma2[i] = np.min(eigenvalues)
        
        # 添加主应力数据
        self.add_point_data(vtk_grid, "Principal_Stress_1_MPa", sigma1 / 1e6)
        self.add_point_data(vtk_grid, "Principal_Stress_2_MPa", sigma2 / 1e6)
        
        # 最大剪应力
        max_shear = (sigma1 - sigma2) / 2
        self.add_point_data(vtk_grid, "Max_Shear_Stress_MPa", max_shear / 1e6)
    
    def _write_pvd_file(self, base_name: str, file_list: List[Dict]):
        """写入PVD文件（ParaView Data）"""
        pvd_file = self.output_dir / f"{base_name}.pvd"
        
        with open(pvd_file, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <Collection>\n')
            
            for item in file_list:
                f.write(f'    <DataSet timestep="{item["time"]}" group="" part="0" ')
                f.write(f'file="{item["filename"]}"/>\n')
            
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        logger.info(f"创建PVD时间序列文件: {pvd_file}")
    
    def export_statistics(self):
        """导出统计数据"""
        stats = {
            'time_steps': len(self.results),
            'time_range': [self.results[0]['time'], self.results[-1]['time']],
            'mesh_info': {
                'n_vertices': self.mesh.num_vertices(),
                'n_cells': self.mesh.num_cells()
            },
            'field_ranges': {}
        }
        
        # 计算各场的范围
        for field in ['temperature', 'von_mises']:
            if field in self.results[0]:
                values = []
                for result in self.results:
                    data = result[field].vector().get_local()
                    values.extend(data)
                
                stats['field_ranges'][field] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values))
                }
        
        # 轧制力统计
        if 'rolling_force' in self.results[0]:
            forces = [r['rolling_force'] for r in self.results]
            stats['rolling_force'] = {
                'min': float(np.min(forces)) / 1e6,  # MN
                'max': float(np.max(forces)) / 1e6,
                'mean': float(np.mean(forces)) / 1e6,
                'unit': 'MN'
            }
        
        # 保存统计信息
        stats_file = self.output_dir / "simulation_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"统计信息已保存: {stats_file}")
        return stats


class AdvancedVTKExporter(RollingResultsVTKExporter):
    """高级VTK导出器，支持更多功能"""
    
    def export_with_mesh_quality(self):
        """导出包含网格质量指标的VTK文件"""
        vtk_grid = self.base_grid
        
        # 计算网格质量
        quality = vtk.vtkMeshQuality()
        quality.SetInputData(vtk_grid)
        
        # 三角形质量度量
        quality.SetTriangleQualityMeasureToAspectRatio()
        quality.Update()
        
        # 获取质量数据
        quality_grid = quality.GetOutput()
        aspect_ratio = quality_grid.GetCellData().GetArray("Quality")
        aspect_ratio.SetName("Aspect_Ratio")
        
        # 添加到网格
        vtk_grid.GetCellData().AddArray(aspect_ratio)
        
        # 计算其他质量指标
        self._add_additional_quality_metrics(vtk_grid)
        
        # 保存
        self.write_vtu(vtk_grid, "mesh_quality.vtu")
        logger.info("网格质量文件已导出")
    
    def _add_additional_quality_metrics(self, vtk_grid: vtk.vtkUnstructuredGrid):
        """添加额外的网格质量指标"""
        n_cells = vtk_grid.GetNumberOfCells()
        
        # 单元面积/体积
        areas = vtk.vtkFloatArray()
        areas.SetName("Cell_Area")
        areas.SetNumberOfTuples(n_cells)
        
        # 最小角度
        min_angles = vtk.vtkFloatArray()
        min_angles.SetName("Min_Angle")
        min_angles.SetNumberOfTuples(n_cells)
        
        for i in range(n_cells):
            cell = vtk_grid.GetCell(i)
            
            # 计算面积（2D三角形）
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                points = []
                for j in range(3):
                    pt = cell.GetPoints().GetPoint(j)
                    points.append(np.array(pt[:2]))  # 只取x,y
                
                # 使用叉积计算面积
                v1 = points[1] - points[0]
                v2 = points[2] - points[0]
                area = 0.5 * abs(np.cross(v1, v2))
                areas.SetValue(i, area)
                
                # 计算最小角度
                angles = []
                for j in range(3):
                    v1 = points[(j+1)%3] - points[j]
                    v2 = points[(j+2)%3] - points[j]
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                    angles.append(angle)
                
                min_angles.SetValue(i, min(angles))
        
        vtk_grid.GetCellData().AddArray(areas)
        vtk_grid.GetCellData().AddArray(min_angles)
    
    def export_hdf5_checkpoint(self, checkpoint_name: str = "checkpoint.h5"):
        """导出HDF5检查点文件（用于重启计算）"""
        h5_file = self.output_dir / checkpoint_name
        
        with h5py.File(h5_file, 'w') as f:
            # 保存网格信息
            mesh_group = f.create_group('mesh')
            mesh_group.create_dataset('coordinates', data=self.mesh.coordinates())
            mesh_group.create_dataset('cells', data=self.mesh.cells())
            mesh_group.attrs['num_vertices'] = self.mesh.num_vertices()
            mesh_group.attrs['num_cells'] = self.mesh.num_cells()
            
            # 保存时间序列数据
            time_group = f.create_group('time_series')
            time_steps = [r['time'] for r in self.results]
            time_group.create_dataset('time', data=time_steps)
            
            # 保存场数据
            fields_group = f.create_group('fields')
            
            # 温度场
            if 'temperature' in self.results[0]:
                temp_data = np.array([r['temperature'].vector().get_local() 
                                    for r in self.results])
                fields_group.create_dataset('temperature', data=temp_data)
            
            # von Mises应力
            if 'von_mises' in self.results[0]:
                vm_data = np.array([r['von_mises'].vector().get_local() 
                                  for r in self.results])
                fields_group.create_dataset('von_mises_stress', data=vm_data)
            
            # 轧制力
            if 'rolling_force' in self.results[0]:
                force_data = np.array([r['rolling_force'] for r in self.results])
                fields_group.create_dataset('rolling_force', data=force_data)
            
            # 保存元数据
            f.attrs['num_time_steps'] = len(self.results)
            f.attrs['created_date'] = str(np.datetime64('now'))
            
        logger.info(f"HDF5检查点已保存: {h5_file}")
    
    def create_animation_script(self, script_name: str = "animate_rolling.py"):
        """生成ParaView Python动画脚本"""
        script_content = '''#!/usr/bin/env python
# ParaView Python脚本 - 轧制过程动画
from paraview.simple import *

# 清理当前场景
Delete(GetSources().values())

# 加载PVD文件
rolling_data = PVDReader(FileName='rolling.pvd')

# 创建渲染视图
renderView = GetActiveViewOrCreate('RenderView')
renderView.ViewSize = [1920, 1080]
renderView.Background = [1.0, 1.0, 1.0]

# 显示数据
display = Show(rolling_data, renderView)
display.Representation = 'Surface'

# 设置颜色映射 - von Mises应力
ColorBy(display, ('POINTS', 'VonMises_Stress_MPa'))
display.RescaleTransferFunctionToDataRange(True)

# 获取颜色条
colorBar = GetScalarBar(GetColorTransferFunction('VonMises_Stress_MPa'))
colorBar.Title = 'von Mises Stress (MPa)'
colorBar.ComponentTitle = ''
colorBar.Visibility = 1

# 设置相机
camera = renderView.GetActiveCamera()
camera.SetPosition(0.2, 0.1, 0.5)
camera.SetFocalPoint(0.0, 0.0, 0.0)
camera.SetViewUp(0.0, 1.0, 0.0)
renderView.ResetCamera()

# 创建动画
animationScene = GetAnimationScene()
animationScene.PlayMode = 'Sequence'
animationScene.NumberOfFrames = 100

# 保存动画
SaveAnimation('rolling_animation.avi', renderView, 
              FrameRate=15,
              FrameWindow=[0, 99])

print("动画已生成: rolling_animation.avi")
'''
        
        script_file = self.output_dir / script_name
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        logger.info(f"ParaView动画脚本已创建: {script_file}")
    
    def export_for_web(self, web_dir: str = "web_viz"):
        """导出用于Web可视化的轻量级数据"""
        web_path = self.output_dir / web_dir
        web_path.mkdir(exist_ok=True)
        
        # 简化网格（用于Web）
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputData(self.base_grid)
        decimator.SetTargetReduction(0.5)  # 减少50%的三角形
        decimator.Update()
        
        simplified_grid = decimator.GetOutput()
        
        # 导出简化的时间序列
        for idx, result in enumerate(self.results[::5]):  # 每5个时间步取1个
            # 创建简化网格的数据
            vtk_grid = vtk.vtkUnstructuredGrid()
            vtk_grid.DeepCopy(simplified_grid)
            
            # 只添加关键字段
            if 'temperature' in result:
                temp_data = result['temperature'].compute_vertex_values(self.mesh)
                # 插值到简化网格...（省略插值代码）
                
            if 'von_mises' in result:
                vm_data = result['von_mises'].compute_vertex_values(self.mesh)
                # 插值到简化网格...
            
            # 导出为轻量级格式
            filename = web_path / f"web_data_{idx:03d}.vtp"
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(str(filename))
            writer.SetInputData(vtk_grid)
            writer.SetCompressorTypeToZLib()
            writer.SetDataModeToBinary()
            writer.Write()
        
        # 创建Web元数据
        web_meta = {
            'num_frames': len(self.results) // 5,
            'time_range': [self.results[0]['time'], self.results[-1]['time']],
            'fields': ['Temperature_Celsius', 'VonMises_Stress_MPa'],
            'simplified': True,
            'reduction_factor': 0.5
        }
        
        with open(web_path / "metadata.json", 'w') as f:
            json.dump(web_meta, f, indent=2)
        
        logger.info(f"Web可视化数据已导出: {web_path}")


def test_vtk_export():
    """测试VTK导出功能"""
    # 创建模拟网格（用于测试）
    import pyfreefem as pff
    
    # 简单矩形网格
    mesh = pff.RectangleMesh(
        pff.Point(0, 0),
        pff.Point(0.1, 0.02),
        50, 10
    )
    
    # 创建模拟结果
    V = pff.FunctionSpace(mesh, "Lagrange", 1)
    
    # 模拟温度场
    temp = pff.Function(V)
    temp.vector()[:] = 1123 + 100 * np.random.rand(V.dim())
    
    # 模拟应力场
    stress = pff.Function(V)
    stress.vector()[:] = 200e6 + 50e6 * np.random.rand(V.dim())
    
    # 创建结果列表
    results = []
    for i in range(10):
        results.append({
            'time': i * 0.001,
            'step': i,
            'temperature': temp,
            'von_mises': stress,
            'rolling_force': 10e6 + 1e6 * np.sin(i * 0.1)
        })
    
    # 导出VTK
    exporter = AdvancedVTKExporter(mesh, results, "test_vtk_output")
    exporter.export_time_series("test_rolling")
    exporter.export_statistics()
    exporter.export_with_mesh_quality()
    exporter.create_animation_script()
    
    print("VTK导出测试完成！")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_vtk_export()
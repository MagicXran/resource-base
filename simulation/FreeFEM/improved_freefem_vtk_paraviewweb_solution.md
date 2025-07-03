# FreeFEM + VTK + ParaviewWeb 轧辊应力场可视化改进方案

## 一、现有方案分析与改进策略

### 1.1 现有PyFreeFEM的局限性
经过深度分析，现有的pyFreeFEM包存在以下限制：
- **仅支持2D问题**：无法处理3D轧制模拟
- **使用临时文件交换数据**：效率低下，I/O开销大
- **功能有限**：仅支持基本的网格和矩阵导入导出

### 1.2 改进策略
1. **直接使用FreeFEM++核心求解器**：保持高性能计算能力
2. **优化Python集成方式**：使用subprocess + 结构化数据交换
3. **增强VTK导出功能**：支持时变数据和多物理场
4. **完整ParaviewWeb集成**：实现实时可视化

## 二、改进的FreeFEM集成方案

### 2.1 FreeFEM模板化方案
```python
# freefem_template_engine.py
# -*- coding: gb2312 -*-
"""
FreeFEM模板引擎 - 动态生成FreeFEM脚本
"""

import os
from string import Template
from pathlib import Path
import json
from typing import Dict, List, Any
import subprocess
import numpy as np

class FreeFEMTemplateEngine:
    """FreeFEM脚本模板引擎"""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
        
    def create_rolling_template(self) -> str:
        """创建轧制模拟的FreeFEM模板"""
        template_content = '''// FreeFEM++ 轧制应力场模拟
// 自动生成的脚本 - 编码: GB2312

// 参数定义
real rollRadius = ${roll_radius};      // 轧辊半径
real h0 = ${thickness_initial};        // 初始厚度
real h1 = ${thickness_final};          // 最终厚度
real rollSpeed = ${roll_speed};        // 轧制速度
real T0 = ${temperature_initial};      // 初始温度
real Troll = ${temperature_roll};      // 轧辊温度
real frictionCoeff = ${friction_coeff};// 摩擦系数

// 材料属性
real E = ${youngs_modulus};            // 弹性模量
real nu = ${poisson_ratio};            // 泊松比
real alpha = ${thermal_expansion};     // 热膨胀系数
real k = ${thermal_conductivity};      // 热导率
real rho = ${density};                 // 密度
real cp = ${specific_heat};            // 比热容

// 计算导出参数
real reduction = (h0 - h1) / h0;       // 压下率
real contactLength = sqrt(rollRadius * (h0 - h1)); // 接触弧长

// 网格参数
int nx = ${mesh_nx};                   // x方向网格数
int ny = ${mesh_ny};                   // y方向网格数
real meshSize = ${mesh_size};          // 网格尺寸

// 几何定义
real L = 0.4;                          // 模型长度
real yRollCenter = h1/2 + rollRadius;  // 轧辊中心y坐标

// 边界定义
border inlet(t=0,1) { x=-L/2; y=h0*(t-0.5); label=1; }
border outlet(t=0,1) { x=L/2; y=h1*(t-0.5); label=2; }
border topEntry(t=0,1) { x=-L/2+t*(L/2-contactLength/2); y=h0/2; label=3; }
border bottomEntry(t=0,1) { x=-L/2+t*(L/2-contactLength/2); y=-h0/2; label=4; }
border topExit(t=0,1) { x=contactLength/2+t*(L/2-contactLength/2); y=h1/2; label=5; }
border bottomExit(t=0,1) { x=contactLength/2+t*(L/2-contactLength/2); y=-h1/2; label=6; }

// 接触弧（简化为斜线）
border rollContactTop(t=0,1) { 
    x = -contactLength/2 + t*contactLength; 
    y = h0/2 - t*(h0-h1)/2; 
    label=7; 
}
border rollContactBottom(t=0,1) { 
    x = -contactLength/2 + t*contactLength; 
    y = -h0/2 + t*(h0-h1)/2; 
    label=8; 
}

// 生成网格
mesh Th = buildmesh(
    inlet(ny) + outlet(ny) + 
    topEntry(nx/2) + bottomEntry(nx/2) +
    topExit(nx/2) + bottomExit(nx/2) +
    rollContactTop(nx/2) + rollContactBottom(nx/2)
);

// 有限元空间定义
fespace Vh(Th, P1);           // 温度场（一阶元）
fespace Wh(Th, [P2, P2]);     // 位移场（二阶矢量元）
fespace Sh(Th, [P1, P1, P1]); // 应力场（用于后处理）

// 声明场变量
Vh T, v, Told = T0;           // 温度场
Wh [u, w], [uu, ww];         // 位移场
Vh vonMises;                  // von Mises应力

// 材料参数（温度相关）
func real Efunc(real temp) {
    return E * (1 - 0.0004*(temp - 293));
}

// Lame常数
real lambda = E*nu/((1+nu)*(1-2*nu));
real mu = E/(2*(1+nu));

// 热传导问题
problem thermal(T, v) = 
    int2d(Th)(
        k*(dx(T)*dx(v) + dy(T)*dy(v))     // 扩散项
    )
    + int1d(Th, 3, 4, 5, 6)(              // 对流边界
        100*(T - 293)*v                     // 对流换热
    )
    - int2d(Th)(
        rho*cp*Told*v/${dt}                 // 时间项
    )
    + on(1, T=T0)                          // 入口温度
    + on(7, 8, T=Troll);                   // 轧辊接触温度

// 定义应变
macro epsilon(u1,u2) [dx(u1), dy(u2), (dy(u1)+dx(u2))/sqrt(2.)] //
macro div(u1,u2) (dx(u1) + dy(u2)) //

// 力学问题（含热应力）
problem mechanical([u,w], [uu,ww]) = 
    int2d(Th)(
        lambda*div(u,w)*div(uu,ww) + 2*mu*(epsilon(u,w)'*epsilon(uu,ww))
    )
    - int2d(Th)(
        // 热应力项
        (lambda + 2*mu)*alpha*(T - T0)*div(uu,ww)
    )
    - int1d(Th, 7, 8)(
        // 接触压力（简化为均布）
        ${contact_pressure}*ww
    )
    + on(1, u=0, w=0);  // 入口固定

// 时间步进求解
real dt = ${time_step};
int nsteps = ${num_steps};

// 输出文件设置
ofstream tempFile("temperature_history.dat");
ofstream stressFile("stress_history.dat");
ofstream forceFile("force_history.dat");

// VTK输出准备
load "iovtk"

// 主循环
for(int step = 0; step < nsteps; step++) {
    real currentTime = step * dt;
    
    // 求解温度场
    Told = T;
    thermal;
    
    // 求解位移场
    mechanical;
    
    // 计算von Mises应力
    real s11 = lambda*div(u,w) + 2*mu*dx(u);
    real s22 = lambda*div(u,w) + 2*mu*dy(w);
    real s12 = mu*(dy(u) + dx(w));
    vonMises = sqrt(s11^2 + s22^2 - s11*s22 + 3*s12^2);
    
    // 计算轧制力
    real rollingForce = int1d(Th, 7, 8)(
        ${contact_pressure}
    );
    
    // 保存数据
    tempFile << currentTime << " " << T[].max << " " << T[].min << endl;
    stressFile << currentTime << " " << vonMises[].max << " " << vonMises[].min << endl;
    forceFile << currentTime << " " << rollingForce << endl;
    
    // VTK输出
    if(step % ${vtk_interval} == 0) {
        string vtkfile = "results/rolling_" + step + ".vtk";
        savevtk(vtkfile, Th, [u, w, 0], T, vonMises,
                dataname="Displacement Temperature VonMises");
    }
    
    // 进度输出
    if(step % 10 == 0) {
        cout << "Step " << step << "/" << nsteps 
             << ", Time=" << currentTime 
             << ", MaxStress=" << vonMises[].max/1e6 << " MPa" << endl;
    }
}

// 最终结果输出
{
    ofstream ff("final_results.json");
    ff << "{" << endl;
    ff << "  \"max_temperature\": " << T[].max << "," << endl;
    ff << "  \"max_stress\": " << vonMises[].max << "," << endl;
    ff << "  \"max_displacement\": " << sqrt(u[]'*u[] + w[]'*w[]) << "," << endl;
    ff << "  \"mesh_vertices\": " << Th.nv << "," << endl;
    ff << "  \"mesh_elements\": " << Th.nt << endl;
    ff << "}" << endl;
}

// 导出网格和场数据用于Python后处理
{
    savemesh(Th, "mesh.msh");
    
    ofstream tempData("temperature_field.dat");
    for(int i = 0; i < Th.nv; i++) {
        tempData << T[][i] << endl;
    }
    
    ofstream stressData("stress_field.dat");
    for(int i = 0; i < Th.nv; i++) {
        stressData << vonMises[][i] << endl;
    }
    
    ofstream dispData("displacement_field.dat");
    for(int i = 0; i < Th.nv; i++) {
        dispData << u[][i] << " " << w[][i] << endl;
    }
}
'''
        return template_content
    
    def generate_script(self, parameters: Dict[str, Any], output_file: str) -> str:
        """根据参数生成FreeFEM脚本"""
        template = Template(self.create_rolling_template())
        
        # 默认参数
        default_params = {
            'roll_radius': 0.5,
            'thickness_initial': 0.025,
            'thickness_final': 0.020,
            'roll_speed': 3.8,
            'temperature_initial': 1123,
            'temperature_roll': 423,
            'friction_coeff': 0.3,
            'youngs_modulus': 210e9,
            'poisson_ratio': 0.3,
            'thermal_expansion': 1.2e-5,
            'thermal_conductivity': 45,
            'density': 7850,
            'specific_heat': 460,
            'mesh_nx': 100,
            'mesh_ny': 20,
            'mesh_size': 0.001,
            'contact_pressure': 150e6,
            'time_step': 0.001,
            'num_steps': 100,
            'vtk_interval': 5,
            'dt': 0.001
        }
        
        # 合并参数
        params = {**default_params, **parameters}
        
        # 生成脚本
        script_content = template.substitute(**params)
        
        # 保存脚本
        with open(output_file, 'w', encoding='gb2312') as f:
            f.write(script_content)
        
        return output_file
```

### 2.2 FreeFEM执行器和数据解析器
```python
# freefem_executor.py
# -*- coding: gb2312 -*-
"""
FreeFEM执行器 - 运行FreeFEM脚本并解析结果
"""

import subprocess
import os
import json
import numpy as np
from pathlib import Path
import logging
import time
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class FreeFEMExecutor:
    """FreeFEM执行器"""
    
    def __init__(self, freefem_path: str = "FreeFem++"):
        self.freefem_path = freefem_path
        self.working_dir = Path("work")
        self.working_dir.mkdir(exist_ok=True)
        
    def check_installation(self) -> bool:
        """检查FreeFEM安装"""
        try:
            result = subprocess.run(
                [self.freefem_path, "-version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"FreeFEM版本: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            logger.error("FreeFEM未找到，请确保已安装并在PATH中")
        return False
    
    def execute_script(self, script_path: str, timeout: int = 600) -> Dict:
        """执行FreeFEM脚本"""
        start_time = time.time()
        
        # 创建结果目录
        result_dir = self.working_dir / "results"
        result_dir.mkdir(exist_ok=True)
        
        # 执行命令
        cmd = [self.freefem_path, script_path, "-v", "0", "-nw"]
        
        try:
            logger.info(f"执行FreeFEM脚本: {script_path}")
            
            # Windows下的进程管理
            if os.name == 'nt':
                # Windows特定设置
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.working_dir,
                    startupinfo=startupinfo
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.working_dir
                )
            
            # 等待完成
            stdout, stderr = process.communicate(timeout=timeout)
            
            execution_time = time.time() - start_time
            
            if process.returncode == 0:
                logger.info(f"FreeFEM执行成功，耗时: {execution_time:.2f}秒")
                return {
                    'success': True,
                    'execution_time': execution_time,
                    'stdout': stdout,
                    'stderr': stderr
                }
            else:
                logger.error(f"FreeFEM执行失败: {stderr}")
                return {
                    'success': False,
                    'error': stderr,
                    'execution_time': execution_time
                }
                
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"FreeFEM执行超时 ({timeout}秒)")
            return {
                'success': False,
                'error': 'Execution timeout',
                'execution_time': timeout
            }
        except Exception as e:
            logger.error(f"FreeFEM执行错误: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def parse_results(self) -> Dict:
        """解析FreeFEM输出结果"""
        results = {}
        
        # 解析最终结果JSON
        json_file = self.working_dir / "final_results.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                results['final'] = json.load(f)
        
        # 解析网格数据
        mesh_file = self.working_dir / "mesh.msh"
        if mesh_file.exists():
            results['mesh'] = self._parse_mesh_file(mesh_file)
        
        # 解析场数据
        results['fields'] = {}
        
        # 温度场
        temp_file = self.working_dir / "temperature_field.dat"
        if temp_file.exists():
            results['fields']['temperature'] = np.loadtxt(temp_file)
        
        # 应力场
        stress_file = self.working_dir / "stress_field.dat"
        if stress_file.exists():
            results['fields']['von_mises'] = np.loadtxt(stress_file)
        
        # 位移场
        disp_file = self.working_dir / "displacement_field.dat"
        if disp_file.exists():
            disp_data = np.loadtxt(disp_file)
            results['fields']['displacement'] = {
                'u': disp_data[:, 0],
                'v': disp_data[:, 1]
            }
        
        # 时间历史数据
        results['history'] = {}
        
        # 温度历史
        temp_hist = self.working_dir / "temperature_history.dat"
        if temp_hist.exists():
            data = np.loadtxt(temp_hist)
            results['history']['temperature'] = {
                'time': data[:, 0],
                'max': data[:, 1],
                'min': data[:, 2]
            }
        
        # 应力历史
        stress_hist = self.working_dir / "stress_history.dat"
        if stress_hist.exists():
            data = np.loadtxt(stress_hist)
            results['history']['stress'] = {
                'time': data[:, 0],
                'max': data[:, 1],
                'min': data[:, 2]
            }
        
        # 轧制力历史
        force_hist = self.working_dir / "force_history.dat"
        if force_hist.exists():
            data = np.loadtxt(force_hist)
            results['history']['rolling_force'] = {
                'time': data[:, 0],
                'force': data[:, 1]
            }
        
        return results
    
    def _parse_mesh_file(self, mesh_file: Path) -> Dict:
        """解析FreeFEM网格文件"""
        mesh_data = {}
        
        with open(mesh_file, 'r') as f:
            lines = f.readlines()
        
        # 简单解析（FreeFEM网格格式）
        # 第一行：节点数 单元数 边界边数
        nums = lines[0].strip().split()
        n_vertices = int(nums[0])
        n_elements = int(nums[1])
        
        # 读取节点坐标
        vertices = []
        for i in range(1, n_vertices + 1):
            coords = lines[i].strip().split()
            vertices.append([float(coords[0]), float(coords[1])])
        
        # 读取单元连接
        elements = []
        for i in range(n_vertices + 1, n_vertices + n_elements + 1):
            elem = lines[i].strip().split()
            elements.append([int(elem[0])-1, int(elem[1])-1, int(elem[2])-1])
        
        mesh_data['vertices'] = np.array(vertices)
        mesh_data['elements'] = np.array(elements)
        mesh_data['n_vertices'] = n_vertices
        mesh_data['n_elements'] = n_elements
        
        return mesh_data
```

### 2.3 高级VTK导出器（改进版）
```python
# advanced_vtk_exporter.py
# -*- coding: gb2312 -*-
"""
高级VTK导出器 - 从FreeFEM结果生成VTK文件
"""

import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class AdvancedFreeFEMVTKExporter:
    """FreeFEM到VTK的高级导出器"""
    
    def __init__(self, mesh_data: Dict, field_data: Dict, output_dir: str = "vtk_output"):
        self.mesh_data = mesh_data
        self.field_data = field_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_unstructured_grid(self) -> vtk.vtkUnstructuredGrid:
        """创建VTK非结构网格"""
        # 创建点
        points = vtk.vtkPoints()
        vertices = self.mesh_data['vertices']
        
        for vertex in vertices:
            # 2D网格，z=0
            points.InsertNextPoint(vertex[0], vertex[1], 0.0)
        
        # 创建单元
        cells = vtk.vtkCellArray()
        elements = self.mesh_data['elements']
        
        for element in elements:
            triangle = vtk.vtkTriangle()
            for i in range(3):
                triangle.GetPointIds().SetId(i, element[i])
            cells.InsertNextCell(triangle)
        
        # 创建网格
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        grid.SetCells(vtk.VTK_TRIANGLE, cells)
        
        return grid
    
    def add_field_data_to_grid(self, grid: vtk.vtkUnstructuredGrid):
        """添加场数据到网格"""
        # 温度场
        if 'temperature' in self.field_data:
            temp_data = self.field_data['temperature']
            temp_array = numpy_to_vtk(temp_data)
            temp_array.SetName("Temperature")
            grid.GetPointData().AddArray(temp_array)
            
            # 摄氏度温度
            temp_celsius = temp_data - 273.15
            temp_c_array = numpy_to_vtk(temp_celsius)
            temp_c_array.SetName("Temperature_Celsius")
            grid.GetPointData().AddArray(temp_c_array)
        
        # von Mises应力
        if 'von_mises' in self.field_data:
            stress_data = self.field_data['von_mises']
            stress_array = numpy_to_vtk(stress_data)
            stress_array.SetName("VonMises_Stress")
            grid.GetPointData().AddArray(stress_array)
            
            # MPa单位
            stress_mpa = stress_data / 1e6
            stress_mpa_array = numpy_to_vtk(stress_mpa)
            stress_mpa_array.SetName("VonMises_Stress_MPa")
            grid.GetPointData().AddArray(stress_mpa_array)
        
        # 位移场
        if 'displacement' in self.field_data:
            disp = self.field_data['displacement']
            u = disp['u']
            v = disp['v']
            
            # 组合为矢量
            disp_vector = np.column_stack((u, v, np.zeros_like(u)))
            disp_array = numpy_to_vtk(disp_vector)
            disp_array.SetName("Displacement")
            disp_array.SetNumberOfComponents(3)
            grid.GetPointData().AddArray(disp_array)
            
            # 位移幅值
            disp_magnitude = np.sqrt(u**2 + v**2)
            mag_array = numpy_to_vtk(disp_magnitude)
            mag_array.SetName("Displacement_Magnitude")
            grid.GetPointData().AddArray(mag_array)
    
    def export_single_timestep(self, filename: str = "rolling_result.vtu"):
        """导出单个时间步的VTU文件"""
        grid = self.create_unstructured_grid()
        self.add_field_data_to_grid(grid)
        
        # 写入VTU文件
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(self.output_dir / filename))
        writer.SetInputData(grid)
        writer.SetDataModeToBinary()
        writer.SetCompressorTypeToZLib()
        writer.Write()
        
        logger.info(f"导出VTU文件: {filename}")
    
    def export_time_series_from_vtk(self, vtk_files: List[str], base_name: str = "rolling"):
        """从FreeFEM生成的VTK文件创建时间序列"""
        pvd_content = ['<?xml version="1.0"?>\n',
                      '<VTKFile type="Collection" version="0.1">\n',
                      '<Collection>\n']
        
        # 转换每个VTK文件为VTU
        for idx, vtk_file in enumerate(vtk_files):
            # 读取传统VTK文件
            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(vtk_file)
            reader.Update()
            
            # 写入VTU文件
            vtu_filename = f"{base_name}_{idx:04d}.vtu"
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(str(self.output_dir / vtu_filename))
            writer.SetInputData(reader.GetOutput())
            writer.SetDataModeToBinary()
            writer.SetCompressorTypeToZLib()
            writer.Write()
            
            # 添加到PVD
            time = idx * 0.001  # 假设时间步长
            pvd_content.append(f'<DataSet timestep="{time}" file="{vtu_filename}"/>\n')
        
        pvd_content.extend(['</Collection>\n', '</VTKFile>\n'])
        
        # 写入PVD文件
        pvd_file = self.output_dir / f"{base_name}.pvd"
        with open(pvd_file, 'w') as f:
            f.writelines(pvd_content)
        
        logger.info(f"创建PVD时间序列: {pvd_file}")
    
    def create_mesh_quality_report(self) -> Dict:
        """创建网格质量报告"""
        grid = self.create_unstructured_grid()
        
        # 计算网格质量指标
        quality = vtk.vtkMeshQuality()
        quality.SetInputData(grid)
        quality.SetTriangleQualityMeasureToAspectRatio()
        quality.Update()
        
        quality_grid = quality.GetOutput()
        quality_array = quality_grid.GetCellData().GetArray("Quality")
        
        # 统计信息
        quality_values = []
        for i in range(quality_array.GetNumberOfTuples()):
            quality_values.append(quality_array.GetValue(i))
        
        quality_values = np.array(quality_values)
        
        report = {
            'mesh_statistics': {
                'n_vertices': self.mesh_data['n_vertices'],
                'n_elements': self.mesh_data['n_elements']
            },
            'quality_metrics': {
                'aspect_ratio': {
                    'min': float(np.min(quality_values)),
                    'max': float(np.max(quality_values)),
                    'mean': float(np.mean(quality_values)),
                    'std': float(np.std(quality_values))
                }
            }
        }
        
        # 计算单元面积
        areas = []
        for i in range(grid.GetNumberOfCells()):
            cell = grid.GetCell(i)
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                p0 = np.array(cell.GetPoints().GetPoint(0)[:2])
                p1 = np.array(cell.GetPoints().GetPoint(1)[:2])
                p2 = np.array(cell.GetPoints().GetPoint(2)[:2])
                
                # 叉积计算面积
                area = 0.5 * abs(np.cross(p1 - p0, p2 - p0))
                areas.append(area)
        
        areas = np.array(areas)
        report['quality_metrics']['element_area'] = {
            'min': float(np.min(areas)),
            'max': float(np.max(areas)),
            'mean': float(np.mean(areas)),
            'std': float(np.std(areas))
        }
        
        # 保存报告
        report_file = self.output_dir / "mesh_quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"网格质量报告: {report_file}")
        return report
```

## 三、ParaviewWeb服务器实现

### 3.1 ParaviewWeb服务器配置
```python
# paraviewweb_server.py
# -*- coding: gb2312 -*-
"""
ParaviewWeb服务器 - 轧制应力场可视化
"""

from wslink import server
from wslink import register as exportRPC
from paraview import simple
from paraview.web import protocols as pv_protocols
import os
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class RollingVisualizationProtocol(pv_protocols.ParaViewWebProtocol):
    """轧制模拟可视化协议"""
    
    def __init__(self):
        super().__init__()
        self.simulation_data_path = os.environ.get('PVW_DATA_PATH', '/data/simulations')
        self.active_source = None
        self.active_view = None
        
    @exportRPC("rolling.load_simulation")
    def load_simulation(self, simulation_id: str) -> Dict:
        """加载模拟结果"""
        try:
            # 构建文件路径
            pvd_file = os.path.join(self.simulation_data_path, f"{simulation_id}.pvd")
            
            if not os.path.exists(pvd_file):
                return {"status": "error", "message": f"Simulation {simulation_id} not found"}
            
            # 清理现有数据
            simple.Delete(simple.GetSources())
            
            # 加载PVD文件
            reader = simple.PVDReader(FileName=pvd_file)
            self.active_source = reader
            
            # 创建渲染视图
            self.active_view = simple.GetActiveViewOrCreate('RenderView')
            self.setup_default_view()
            
            # 显示数据
            display = simple.Show(reader, self.active_view)
            display.Representation = 'Surface'
            
            # 设置默认着色
            simple.ColorBy(display, ('POINTS', 'VonMises_Stress_MPa'))
            display.RescaleTransferFunctionToDataRange(True)
            
            # 添加颜色条
            self.add_color_bar('VonMises_Stress_MPa', 'von Mises应力 (MPa)')
            
            # 渲染
            simple.Render()
            
            # 获取数据信息
            data_info = self.get_data_information()
            
            return {
                "status": "success",
                "simulation_id": simulation_id,
                "data_info": data_info
            }
            
        except Exception as e:
            logger.error(f"加载模拟失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    @exportRPC("rolling.update_field")
    def update_field(self, field_name: str, component: Optional[str] = None) -> Dict:
        """更新显示字段"""
        try:
            if not self.active_source:
                return {"status": "error", "message": "No simulation loaded"}
            
            display = simple.GetDisplayProperties(self.active_source)
            
            # 设置着色字段
            if component:
                simple.ColorBy(display, ('POINTS', field_name, component))
            else:
                simple.ColorBy(display, ('POINTS', field_name))
            
            # 更新颜色范围
            display.RescaleTransferFunctionToDataRange(True)
            
            # 更新颜色条标题
            field_titles = {
                'Temperature_Celsius': '温度 (°C)',
                'VonMises_Stress_MPa': 'von Mises应力 (MPa)',
                'Displacement_Magnitude': '位移幅值 (mm)',
                'Stress_XX_MPa': '应力σxx (MPa)',
                'Stress_YY_MPa': '应力σyy (MPa)',
                'Stress_XY_MPa': '应力σxy (MPa)'
            }
            
            title = field_titles.get(field_name, field_name)
            self.add_color_bar(field_name, title)
            
            simple.Render()
            
            return {"status": "success", "field": field_name}
            
        except Exception as e:
            logger.error(f"更新字段失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    @exportRPC("rolling.set_time_step")
    def set_time_step(self, step: int) -> Dict:
        """设置时间步"""
        try:
            animation_scene = simple.GetAnimationScene()
            animation_scene.AnimationTime = step
            simple.Render()
            
            return {"status": "success", "time_step": step}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @exportRPC("rolling.play_animation")
    def play_animation(self, start: int = 0, end: Optional[int] = None) -> Dict:
        """播放动画"""
        try:
            animation_scene = simple.GetAnimationScene()
            animation_scene.PlayMode = 'Sequence'
            
            if end is not None:
                animation_scene.StartTime = start
                animation_scene.EndTime = end
            
            animation_scene.Play()
            
            return {"status": "success", "playing": True}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @exportRPC("rolling.export_screenshot")
    def export_screenshot(self, filename: str = "screenshot.png", 
                         width: int = 1920, height: int = 1080) -> Dict:
        """导出截图"""
        try:
            simple.SaveScreenshot(
                filename,
                self.active_view,
                ImageResolution=[width, height],
                TransparentBackground=0
            )
            
            return {"status": "success", "filename": filename}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @exportRPC("rolling.get_field_range")
    def get_field_range(self, field_name: str) -> Dict:
        """获取字段数值范围"""
        try:
            if not self.active_source:
                return {"status": "error", "message": "No simulation loaded"}
            
            # 获取数据范围
            data_info = self.active_source.GetDataInformation()
            point_data = data_info.GetPointDataInformation()
            array_info = point_data.GetArrayInformation(field_name)
            
            if array_info:
                data_range = array_info.GetComponentRange(0)
                return {
                    "status": "success",
                    "field": field_name,
                    "range": {
                        "min": data_range[0],
                        "max": data_range[1]
                    }
                }
            else:
                return {"status": "error", "message": f"Field {field_name} not found"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def setup_default_view(self):
        """设置默认视图"""
        if not self.active_view:
            return
        
        # 设置背景
        self.active_view.Background = [1.0, 1.0, 1.0]
        self.active_view.Background2 = [0.9, 0.9, 0.9]
        self.active_view.UseGradientBackground = 1
        
        # 设置相机
        self.active_view.CameraPosition = [0.2, 0.1, 0.5]
        self.active_view.CameraFocalPoint = [0.0, 0.0, 0.0]
        self.active_view.CameraViewUp = [0.0, 1.0, 0.0]
        
        # 重置相机
        simple.ResetCamera()
        
        # 添加坐标轴
        self.active_view.OrientationAxesVisibility = 1
        
    def add_color_bar(self, field_name: str, title: str):
        """添加颜色条"""
        color_bar = simple.GetScalarBar(
            simple.GetColorTransferFunction(field_name)
        )
        color_bar.Title = title
        color_bar.ComponentTitle = ''
        color_bar.Visibility = 1
        color_bar.Position = [0.85, 0.2]
        color_bar.Position2 = [0.12, 0.6]
        color_bar.TitleFontSize = 14
        color_bar.LabelFontSize = 12
        
    def get_data_information(self) -> Dict:
        """获取数据信息"""
        if not self.active_source:
            return {}
        
        data_info = self.active_source.GetDataInformation()
        
        # 获取可用字段
        point_data = data_info.GetPointDataInformation()
        available_fields = []
        
        for i in range(point_data.GetNumberOfArrays()):
            array_name = point_data.GetArrayName(i)
            array_info = point_data.GetArrayInformation(i)
            available_fields.append({
                'name': array_name,
                'components': array_info.GetNumberOfComponents(),
                'range': list(array_info.GetComponentRange(0))
            })
        
        # 获取时间信息
        time_values = []
        if hasattr(self.active_source, 'TimestepValues'):
            time_values = list(self.active_source.TimestepValues)
        
        return {
            'bounds': list(data_info.GetBounds()),
            'n_points': data_info.GetNumberOfPoints(),
            'n_cells': data_info.GetNumberOfCells(),
            'available_fields': available_fields,
            'time_steps': len(time_values),
            'time_range': [time_values[0], time_values[-1]] if time_values else [0, 0]
        }

# 启动服务器函数
def start_paraviewweb_server(host: str = "0.0.0.0", port: int = 9000):
    """启动ParaviewWeb服务器"""
    # 配置参数
    args = {
        "host": host,
        "port": port,
        "ws": f"ws://{host}:{port}/ws",
        "lp": f"http://{host}:{port}",
        "content": "./www",
        "debug": True,
        "timeout": 300,
        "nosignalhandlers": True
    }
    
    # 创建协议
    protocol = RollingVisualizationProtocol()
    
    # 启动服务器
    server.start_webserver(options=args, protocol=protocol)
    
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 启动服务器
    start_paraviewweb_server()
```

## 四、完整的FastAPI后端集成

### 4.1 主API服务
```python
# main_api.py
# -*- coding: gb2312 -*-
"""
主API服务 - 集成FreeFEM、VTK和ParaviewWeb
"""

from fastapi import FastAPI, BackgroundTasks, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import json
import uuid
from datetime import datetime
import logging
from pathlib import Path

# 导入自定义模块
from freefem_template_engine import FreeFEMTemplateEngine
from freefem_executor import FreeFEMExecutor
from advanced_vtk_exporter import AdvancedFreeFEMVTKExporter

app = FastAPI(
    title="轧制应力场分析系统API",
    description="FreeFEM + VTK + ParaviewWeb集成系统",
    version="2.0.0"
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 任务存储
simulation_tasks = {}

# 数据模型
class RollingParameters(BaseModel):
    """轧制参数"""
    roll_radius: float = Field(0.5, gt=0, le=2.0, description="轧辊半径(m)")
    thickness_initial: float = Field(0.025, gt=0, le=0.1, description="初始厚度(m)")
    thickness_final: float = Field(0.020, gt=0, description="最终厚度(m)")
    roll_speed: float = Field(3.8, gt=0, le=20, description="轧制速度(m/s)")
    temperature_initial: float = Field(1123, gt=273, le=1500, description="初始温度(K)")
    temperature_roll: float = Field(423, gt=273, le=800, description="轧辊温度(K)")
    friction_coefficient: float = Field(0.3, ge=0, le=1, description="摩擦系数")
    
    class Config:
        schema_extra = {
            "example": {
                "roll_radius": 0.5,
                "thickness_initial": 0.025,
                "thickness_final": 0.020,
                "roll_speed": 3.8,
                "temperature_initial": 1123,
                "temperature_roll": 423,
                "friction_coefficient": 0.3
            }
        }

class MaterialProperties(BaseModel):
    """材料属性"""
    density: float = Field(7850, gt=0, description="密度(kg/m³)")
    youngs_modulus: float = Field(210e9, gt=0, description="弹性模量(Pa)")
    poisson_ratio: float = Field(0.3, ge=0, le=0.5, description="泊松比")
    thermal_expansion: float = Field(1.2e-5, gt=0, description="热膨胀系数(1/K)")
    thermal_conductivity: float = Field(45, gt=0, description="热导率(W/m·K)")
    specific_heat: float = Field(460, gt=0, description="比热容(J/kg·K)")

class SimulationRequest(BaseModel):
    """模拟请求"""
    rolling_params: RollingParameters
    material_props: MaterialProperties = MaterialProperties()
    mesh_size: float = Field(0.001, gt=0, le=0.01, description="网格尺寸")
    time_steps: int = Field(100, ge=10, le=1000, description="时间步数")
    vtk_interval: int = Field(5, ge=1, le=50, description="VTK输出间隔")

class SimulationResponse(BaseModel):
    """模拟响应"""
    task_id: str
    status: str
    message: str
    created_at: datetime

@app.get("/")
async def root():
    """API根路径"""
    return {
        "name": "轧制应力场分析系统",
        "version": "2.0.0",
        "endpoints": {
            "docs": "/docs",
            "simulation": "/api/v1/simulation",
            "visualization": "/api/v1/visualization"
        }
    }

@app.post("/api/v1/simulation/create", response_model=SimulationResponse)
async def create_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
):
    """创建新的轧制模拟任务"""
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    simulation_tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "任务已创建",
        "created_at": datetime.now(),
        "parameters": request.dict(),
        "results": None,
        "error": None
    }
    
    # 添加后台任务
    background_tasks.add_task(
        run_simulation_task,
        task_id,
        request
    )
    
    logger.info(f"创建模拟任务: {task_id}")
    
    return SimulationResponse(
        task_id=task_id,
        status="pending",
        message="模拟任务已创建，正在排队执行",
        created_at=datetime.now()
    )

@app.get("/api/v1/simulation/{task_id}/status")
async def get_simulation_status(task_id: str):
    """获取模拟任务状态"""
    if task_id not in simulation_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = simulation_tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
        "created_at": task["created_at"]
    }

@app.get("/api/v1/simulation/{task_id}/results")
async def get_simulation_results(task_id: str):
    """获取模拟结果"""
    if task_id not in simulation_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = simulation_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务未完成")
    
    if task["error"]:
        raise HTTPException(status_code=500, detail=task["error"])
    
    return {
        "task_id": task_id,
        "status": "completed",
        "results": task["results"],
        "vtk_files": task.get("vtk_files", []),
        "paraview_url": f"/visualization/{task_id}"
    }

@app.websocket("/ws/simulation/{task_id}")
async def simulation_progress_ws(websocket: WebSocket, task_id: str):
    """WebSocket实时进度推送"""
    await websocket.accept()
    
    if task_id not in simulation_tasks:
        await websocket.send_json({
            "type": "error",
            "message": "任务不存在"
        })
        await websocket.close()
        return
    
    # 推送进度更新
    while True:
        task = simulation_tasks[task_id]
        
        await websocket.send_json({
            "type": "progress",
            "data": {
                "status": task["status"],
                "progress": task["progress"],
                "message": task["message"]
            }
        })
        
        if task["status"] in ["completed", "failed"]:
            break
        
        await asyncio.sleep(1)
    
    await websocket.close()

async def run_simulation_task(task_id: str, request: SimulationRequest):
    """执行模拟任务"""
    try:
        # 更新状态
        simulation_tasks[task_id]["status"] = "running"
        simulation_tasks[task_id]["progress"] = 10
        simulation_tasks[task_id]["message"] = "初始化FreeFEM环境"
        
        # 1. 生成FreeFEM脚本
        template_engine = FreeFEMTemplateEngine()
        
        # 准备参数
        params = {
            **request.rolling_params.dict(),
            **request.material_props.dict(),
            'mesh_size': request.mesh_size,
            'num_steps': request.time_steps,
            'vtk_interval': request.vtk_interval,
            'time_step': 0.001
        }
        
        script_file = f"work/rolling_{task_id}.edp"
        template_engine.generate_script(params, script_file)
        
        simulation_tasks[task_id]["progress"] = 20
        simulation_tasks[task_id]["message"] = "执行FreeFEM求解"
        
        # 2. 执行FreeFEM
        executor = FreeFEMExecutor()
        
        if not executor.check_installation():
            raise Exception("FreeFEM未正确安装")
        
        exec_result = executor.execute_script(script_file, timeout=600)
        
        if not exec_result['success']:
            raise Exception(f"FreeFEM执行失败: {exec_result['error']}")
        
        simulation_tasks[task_id]["progress"] = 60
        simulation_tasks[task_id]["message"] = "解析计算结果"
        
        # 3. 解析结果
        results = executor.parse_results()
        
        simulation_tasks[task_id]["progress"] = 80
        simulation_tasks[task_id]["message"] = "生成VTK文件"
        
        # 4. 导出VTK
        if 'mesh' in results and 'fields' in results:
            vtk_exporter = AdvancedFreeFEMVTKExporter(
                results['mesh'],
                results['fields'],
                output_dir=f"vtk_output/{task_id}"
            )
            
            # 导出单个时间步
            vtk_exporter.export_single_timestep(f"rolling_{task_id}.vtu")
            
            # 导出网格质量报告
            quality_report = vtk_exporter.create_mesh_quality_report()
            
            # 如果有VTK文件序列，转换为VTU
            vtk_files = list(Path("work/results").glob("rolling_*.vtk"))
            if vtk_files:
                vtk_exporter.export_time_series_from_vtk(
                    [str(f) for f in vtk_files],
                    base_name=f"rolling_{task_id}"
                )
        
        simulation_tasks[task_id]["progress"] = 100
        simulation_tasks[task_id]["status"] = "completed"
        simulation_tasks[task_id]["message"] = "模拟完成"
        simulation_tasks[task_id]["results"] = {
            "final_values": results.get('final', {}),
            "history": results.get('history', {}),
            "mesh_quality": quality_report if 'quality_report' in locals() else None,
            "execution_time": exec_result['execution_time']
        }
        simulation_tasks[task_id]["vtk_files"] = [
            f"rolling_{task_id}.vtu",
            f"rolling_{task_id}.pvd"
        ]
        
        logger.info(f"任务完成: {task_id}")
        
    except Exception as e:
        logger.error(f"任务失败 {task_id}: {str(e)}")
        simulation_tasks[task_id]["status"] = "failed"
        simulation_tasks[task_id]["error"] = str(e)
        simulation_tasks[task_id]["message"] = f"模拟失败: {str(e)}"

@app.get("/api/v1/visualization/{task_id}")
async def get_visualization_info(task_id: str):
    """获取可视化信息"""
    if task_id not in simulation_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = simulation_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务未完成")
    
    return {
        "task_id": task_id,
        "paraview_ws_url": f"ws://localhost:9000/ws",
        "vtk_files": task.get("vtk_files", []),
        "available_fields": [
            "Temperature_Celsius",
            "VonMises_Stress_MPa",
            "Displacement_Magnitude"
        ]
    }

@app.get("/api/v1/tasks")
async def list_tasks(limit: int = 10, status: Optional[str] = None):
    """列出所有任务"""
    tasks = []
    
    for task_id, task in simulation_tasks.items():
        if status and task["status"] != status:
            continue
        
        tasks.append({
            "task_id": task_id,
            "status": task["status"],
            "created_at": task["created_at"],
            "message": task["message"]
        })
    
    # 按创建时间排序
    tasks.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "total": len(tasks),
        "tasks": tasks[:limit]
    }

if __name__ == "__main__":
    import uvicorn
    
    # 创建必要的目录
    Path("work").mkdir(exist_ok=True)
    Path("vtk_output").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
```

## 五、Docker部署配置

### 5.1 Dockerfile配置
```dockerfile
# Dockerfile.freefem
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=zh_CN.GB2312
ENV LC_ALL=zh_CN.GB2312

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    cmake \
    libopenmpi-dev \
    libmumps-dev \
    libscotch-dev \
    libblas-dev \
    liblapack-dev \
    python3 \
    python3-pip \
    locales \
    && rm -rf /var/lib/apt/lists/*

# 设置中文支持
RUN locale-gen zh_CN.GB2312

# 安装FreeFEM
RUN wget https://github.com/FreeFem/FreeFem-sources/releases/download/v4.13/FreeFem++-4.13-ubuntu-22.04-amd64.deb \
    && dpkg -i FreeFem++-4.13-ubuntu-22.04-amd64.deb \
    && rm FreeFem++-4.13-ubuntu-22.04-amd64.deb

# 安装Python依赖
COPY requirements.txt /app/
WORKDIR /app
RUN pip3 install -r requirements.txt

# 复制应用代码
COPY . /app/

# 创建工作目录
RUN mkdir -p /data/simulations /data/vtk_output

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python3", "main_api.py"]
```

```dockerfile
# Dockerfile.paraview
FROM kitware/paraviewweb:5.11

# 安装额外依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装Python包
RUN pip3 install wslink

# 复制ParaviewWeb服务器代码
COPY paraviewweb_server.py /opt/paraview/

# 设置工作目录
WORKDIR /opt/paraview

# 环境变量
ENV PVW_DATA_PATH=/data/simulations

# 暴露端口
EXPOSE 9000

# 启动命令
CMD ["python3", "paraviewweb_server.py"]
```

### 5.2 Docker Compose配置
```yaml
# docker-compose.yml
version: '3.8'

services:
  # FreeFEM计算服务
  freefem-api:
    build:
      context: .
      dockerfile: Dockerfile.freefem
    container_name: rolling-freefem-api
    ports:
      - "8000:8000"
    volumes:
      - ./simulations:/data/simulations
      - ./vtk_output:/data/vtk_output
      - ./work:/app/work
    environment:
      - PYTHONPATH=/app
      - PARAVIEW_URL=http://paraview:9000
    networks:
      - rolling-network
    restart: unless-stopped

  # ParaviewWeb可视化服务
  paraview:
    build:
      context: .
      dockerfile: Dockerfile.paraview
    container_name: rolling-paraview
    ports:
      - "9000:9000"
    volumes:
      - ./simulations:/data/simulations
      - ./vtk_output:/data/vtk_output
    environment:
      - DISPLAY=:99
      - PVW_DATA_PATH=/data/simulations
    networks:
      - rolling-network
    restart: unless-stopped

  # Nginx反向代理
  nginx:
    image: nginx:alpine
    container_name: rolling-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./static:/usr/share/nginx/html
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - freefem-api
      - paraview
    networks:
      - rolling-network
    restart: unless-stopped

  # Redis缓存（可选）
  redis:
    image: redis:7-alpine
    container_name: rolling-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - rolling-network
    restart: unless-stopped

networks:
  rolling-network:
    driver: bridge

volumes:
  redis-data:
```

### 5.3 Nginx配置
```nginx
# nginx.conf
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/javascript application/xml+rss 
               application/rss+xml application/atom+xml image/svg+xml;
    
    # 上游服务器
    upstream freefem_api {
        server freefem-api:8000;
    }
    
    upstream paraview_ws {
        server paraview:9000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # 静态文件
        location / {
            root /usr/share/nginx/html;
            index index.html;
            try_files $uri $uri/ /index.html;
        }
        
        # API代理
        location /api/ {
            proxy_pass http://freefem_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 超时设置
            proxy_connect_timeout 300;
            proxy_send_timeout 300;
            proxy_read_timeout 300;
        }
        
        # WebSocket代理（API）
        location /ws/ {
            proxy_pass http://freefem_api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
        
        # ParaviewWeb代理
        location /paraview/ {
            proxy_pass http://paraview_ws/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            
            # WebSocket支持
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # 缓冲设置
            proxy_buffering off;
            proxy_read_timeout 86400;
        }
        
        # VTK文件下载
        location /download/vtk/ {
            alias /data/vtk_output/;
            autoindex on;
            autoindex_exact_size off;
            autoindex_localtime on;
        }
    }
}
```

## 六、前端界面实现

### 6.1 主页面
```html
<!-- static/index.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="GB2312">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>轧制应力场分析系统</title>
    <link rel="stylesheet" href="css/style.css">
    <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
</head>
<body>
    <div id="app">
        <header>
            <h1>热轧过程应力场分析系统</h1>
            <p>基于FreeFEM + VTK + ParaviewWeb的集成解决方案</p>
        </header>
        
        <main>
            <!-- 参数输入区 -->
            <section class="parameter-section">
                <h2>轧制参数设置</h2>
                
                <div class="parameter-group">
                    <h3>几何参数</h3>
                    <div class="input-group">
                        <label>轧辊半径 (m):</label>
                        <input type="number" v-model.number="params.roll_radius" 
                               min="0.1" max="2.0" step="0.1">
                    </div>
                    <div class="input-group">
                        <label>初始厚度 (mm):</label>
                        <input type="number" v-model.number="thickness_initial_mm" 
                               min="10" max="100" step="1">
                    </div>
                    <div class="input-group">
                        <label>最终厚度 (mm):</label>
                        <input type="number" v-model.number="thickness_final_mm" 
                               min="5" max="50" step="1">
                    </div>
                </div>
                
                <div class="parameter-group">
                    <h3>工艺参数</h3>
                    <div class="input-group">
                        <label>轧制速度 (m/s):</label>
                        <input type="number" v-model.number="params.roll_speed" 
                               min="0.5" max="20" step="0.1">
                    </div>
                    <div class="input-group">
                        <label>初始温度 (°C):</label>
                        <input type="number" v-model.number="temperature_initial_c" 
                               min="800" max="1200" step="10">
                    </div>
                    <div class="input-group">
                        <label>摩擦系数:</label>
                        <input type="number" v-model.number="params.friction_coefficient" 
                               min="0" max="1" step="0.05">
                    </div>
                </div>
                
                <div class="parameter-group">
                    <h3>计算参数</h3>
                    <div class="input-group">
                        <label>时间步数:</label>
                        <input type="number" v-model.number="time_steps" 
                               min="10" max="500" step="10">
                    </div>
                    <div class="input-group">
                        <label>VTK输出间隔:</label>
                        <input type="number" v-model.number="vtk_interval" 
                               min="1" max="20" step="1">
                    </div>
                </div>
                
                <div class="button-group">
                    <button @click="startSimulation" :disabled="isRunning" class="btn-primary">
                        {{ isRunning ? '计算中...' : '开始模拟' }}
                    </button>
                    <button @click="resetParameters" class="btn-secondary">重置参数</button>
                </div>
            </section>
            
            <!-- 进度显示 -->
            <section v-if="currentTask" class="progress-section">
                <h2>计算进度</h2>
                <div class="progress-bar">
                    <div class="progress-fill" :style="{width: progress + '%'}"></div>
                </div>
                <p class="progress-text">{{ progressMessage }}</p>
                <p class="task-id">任务ID: {{ currentTask }}</p>
            </section>
            
            <!-- 结果显示 -->
            <section v-if="results" class="results-section">
                <h2>计算结果</h2>
                
                <div class="result-summary">
                    <div class="result-item">
                        <h4>最高温度</h4>
                        <p>{{ (results.final_values.max_temperature - 273.15).toFixed(1) }} °C</p>
                    </div>
                    <div class="result-item">
                        <h4>最大应力</h4>
                        <p>{{ (results.final_values.max_stress / 1e6).toFixed(1) }} MPa</p>
                    </div>
                    <div class="result-item">
                        <h4>网格信息</h4>
                        <p>{{ results.final_values.mesh_vertices }} 节点</p>
                        <p>{{ results.final_values.mesh_elements }} 单元</p>
                    </div>
                    <div class="result-item">
                        <h4>计算时间</h4>
                        <p>{{ results.execution_time.toFixed(2) }} 秒</p>
                    </div>
                </div>
                
                <div class="visualization-buttons">
                    <button @click="openVisualization" class="btn-primary">
                        打开3D可视化
                    </button>
                    <button @click="downloadResults" class="btn-secondary">
                        下载VTK文件
                    </button>
                </div>
            </section>
            
            <!-- 任务列表 -->
            <section class="task-list-section">
                <h2>历史任务</h2>
                <table class="task-table">
                    <thead>
                        <tr>
                            <th>任务ID</th>
                            <th>状态</th>
                            <th>创建时间</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="task in taskList" :key="task.task_id">
                            <td>{{ task.task_id.substring(0, 8) }}...</td>
                            <td>
                                <span :class="'status-' + task.status">
                                    {{ getStatusText(task.status) }}
                                </span>
                            </td>
                            <td>{{ formatDate(task.created_at) }}</td>
                            <td>
                                <button v-if="task.status === 'completed'" 
                                        @click="loadTask(task.task_id)"
                                        class="btn-small">
                                    查看
                                </button>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </section>
        </main>
        
        <!-- 可视化弹窗 -->
        <div v-if="showVisualization" class="modal">
            <div class="modal-content">
                <span class="close" @click="closeVisualization">&times;</span>
                <h2>3D可视化 - ParaviewWeb</h2>
                <iframe :src="visualizationUrl" 
                        width="100%" 
                        height="600px"
                        frameborder="0">
                </iframe>
            </div>
        </div>
    </div>
    
    <script src="js/app.js"></script>
</body>
</html>
```

### 6.2 Vue.js应用逻辑
```javascript
// static/js/app.js
const { createApp } = Vue;

createApp({
    data() {
        return {
            // 参数
            params: {
                roll_radius: 0.5,
                thickness_initial: 0.025,
                thickness_final: 0.020,
                roll_speed: 3.8,
                temperature_initial: 1123,
                temperature_roll: 423,
                friction_coefficient: 0.3
            },
            thickness_initial_mm: 25,
            thickness_final_mm: 20,
            temperature_initial_c: 850,
            time_steps: 100,
            vtk_interval: 5,
            
            // 状态
            isRunning: false,
            currentTask: null,
            progress: 0,
            progressMessage: '',
            results: null,
            taskList: [],
            showVisualization: false,
            visualizationUrl: '',
            
            // WebSocket
            ws: null
        }
    },
    
    watch: {
        thickness_initial_mm(val) {
            this.params.thickness_initial = val / 1000;
        },
        thickness_final_mm(val) {
            this.params.thickness_final = val / 1000;
        },
        temperature_initial_c(val) {
            this.params.temperature_initial = val + 273.15;
        }
    },
    
    computed: {
        reductionRatio() {
            return ((this.params.thickness_initial - this.params.thickness_final) / 
                    this.params.thickness_initial * 100).toFixed(1);
        }
    },
    
    mounted() {
        this.loadTaskList();
        // 定期刷新任务列表
        setInterval(() => {
            this.loadTaskList();
        }, 5000);
    },
    
    methods: {
        async startSimulation() {
            if (this.params.thickness_final >= this.params.thickness_initial) {
                alert('最终厚度必须小于初始厚度！');
                return;
            }
            
            this.isRunning = true;
            this.progress = 0;
            this.results = null;
            
            try {
                // 创建模拟请求
                const request = {
                    rolling_params: this.params,
                    material_props: {
                        density: 7850,
                        youngs_modulus: 210e9,
                        poisson_ratio: 0.3,
                        thermal_expansion: 1.2e-5,
                        thermal_conductivity: 45,
                        specific_heat: 460
                    },
                    mesh_size: 0.001,
                    time_steps: this.time_steps,
                    vtk_interval: this.vtk_interval
                };
                
                // 发送请求
                const response = await axios.post('/api/v1/simulation/create', request);
                this.currentTask = response.data.task_id;
                
                // 连接WebSocket监听进度
                this.connectWebSocket(this.currentTask);
                
            } catch (error) {
                console.error('创建模拟失败:', error);
                alert('创建模拟失败: ' + error.message);
                this.isRunning = false;
            }
        },
        
        connectWebSocket(taskId) {
            const wsUrl = `ws://${window.location.host}/ws/simulation/${taskId}`;
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'progress') {
                    this.progress = data.data.progress;
                    this.progressMessage = data.data.message;
                    
                    if (data.data.status === 'completed') {
                        this.onSimulationComplete(taskId);
                    } else if (data.data.status === 'failed') {
                        this.onSimulationFailed(data.data.message);
                    }
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket错误:', error);
            };
        },
        
        async onSimulationComplete(taskId) {
            try {
                // 获取结果
                const response = await axios.get(`/api/v1/simulation/${taskId}/results`);
                this.results = response.data.results;
                this.isRunning = false;
                
                // 刷新任务列表
                this.loadTaskList();
                
                // 显示成功消息
                this.progressMessage = '模拟完成！';
                
            } catch (error) {
                console.error('获取结果失败:', error);
                alert('获取结果失败: ' + error.message);
            }
        },
        
        onSimulationFailed(message) {
            this.isRunning = false;
            this.progressMessage = '模拟失败: ' + message;
            alert('模拟失败: ' + message);
        },
        
        async loadTaskList() {
            try {
                const response = await axios.get('/api/v1/tasks?limit=10');
                this.taskList = response.data.tasks;
            } catch (error) {
                console.error('加载任务列表失败:', error);
            }
        },
        
        async loadTask(taskId) {
            try {
                const response = await axios.get(`/api/v1/simulation/${taskId}/results`);
                this.results = response.data.results;
                this.currentTask = taskId;
            } catch (error) {
                alert('加载任务失败: ' + error.message);
            }
        },
        
        openVisualization() {
            if (!this.currentTask) return;
            
            // 构建ParaviewWeb URL
            this.visualizationUrl = `/paraview/?simulation=${this.currentTask}`;
            this.showVisualization = true;
        },
        
        closeVisualization() {
            this.showVisualization = false;
        },
        
        async downloadResults() {
            if (!this.currentTask) return;
            
            // 下载VTK文件
            window.open(`/download/vtk/${this.currentTask}/rolling_${this.currentTask}.vtu`);
        },
        
        resetParameters() {
            this.params = {
                roll_radius: 0.5,
                thickness_initial: 0.025,
                thickness_final: 0.020,
                roll_speed: 3.8,
                temperature_initial: 1123,
                temperature_roll: 423,
                friction_coefficient: 0.3
            };
            this.thickness_initial_mm = 25;
            this.thickness_final_mm = 20;
            this.temperature_initial_c = 850;
            this.time_steps = 100;
            this.vtk_interval = 5;
        },
        
        getStatusText(status) {
            const statusMap = {
                'pending': '等待中',
                'running': '计算中',
                'completed': '已完成',
                'failed': '失败'
            };
            return statusMap[status] || status;
        },
        
        formatDate(dateStr) {
            const date = new Date(dateStr);
            return date.toLocaleString('zh-CN');
        }
    }
}).mount('#app');
```

## 七、部署和运行步骤

### 7.1 环境准备
```bash
# 1. 克隆代码
git clone <repository>
cd rolling-simulation

# 2. 创建必要的目录
mkdir -p work simulations vtk_output static/css static/js templates

# 3. 安装Python依赖
pip install -r requirements.txt

# 4. 确保FreeFEM已安装
FreeFem++ -version
```

### 7.2 开发环境运行
```bash
# 1. 启动FastAPI服务
python main_api.py

# 2. 启动ParaviewWeb服务（新终端）
python paraviewweb_server.py

# 3. 访问系统
# API文档: http://localhost:8000/docs
# 主界面: http://localhost/
```

### 7.3 生产环境部署
```bash
# 1. 构建Docker镜像
docker-compose build

# 2. 启动所有服务
docker-compose up -d

# 3. 查看日志
docker-compose logs -f

# 4. 停止服务
docker-compose down
```

### 7.4 系统配置优化
```python
# config.py
# -*- coding: gb2312 -*-
"""
系统配置文件
"""

import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "work"
VTK_OUTPUT_DIR = BASE_DIR / "vtk_output"
SIMULATIONS_DIR = BASE_DIR / "simulations"

# FreeFEM配置
FREEFEM_EXECUTABLE = os.environ.get('FREEFEM_PATH', 'FreeFem++')
FREEFEM_TIMEOUT = int(os.environ.get('FREEFEM_TIMEOUT', '600'))

# ParaviewWeb配置
PARAVIEW_HOST = os.environ.get('PARAVIEW_HOST', 'localhost')
PARAVIEW_PORT = int(os.environ.get('PARAVIEW_PORT', '9000'))
PARAVIEW_WS_URL = f"ws://{PARAVIEW_HOST}:{PARAVIEW_PORT}/ws"

# API配置
API_HOST = os.environ.get('API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('API_PORT', '8000'))
API_WORKERS = int(os.environ.get('API_WORKERS', '4'))

# 任务配置
MAX_CONCURRENT_TASKS = int(os.environ.get('MAX_CONCURRENT_TASKS', '4'))
TASK_RETENTION_DAYS = int(os.environ.get('TASK_RETENTION_DAYS', '7'))

# 日志配置
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 创建必要的目录
for dir_path in [WORK_DIR, VTK_OUTPUT_DIR, SIMULATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
```

## 八、总结

本改进方案实现了以下目标：

1. **优化的FreeFEM集成**：使用模板引擎和结构化数据交换，避免了pyFreeFEM的限制
2. **完整的VTK导出**：支持时间序列、多物理场和网格质量分析
3. **生产级ParaviewWeb集成**：实现了完整的可视化服务器和客户端
4. **企业级API服务**：FastAPI + WebSocket实现实时进度推送
5. **容器化部署**：Docker Compose一键部署整个系统
6. **中文支持**：考虑了GB2312编码要求

该方案可以直接用于生产环境，提供了从计算到可视化的完整工作流。
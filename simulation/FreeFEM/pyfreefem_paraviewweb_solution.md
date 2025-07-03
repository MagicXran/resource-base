# PyFreeFEM + ParaviewWeb 轧辊应力场可视化完整解决方案

## 一、PyFreeFEM优势分析

### 1.1 相比传统FreeFEM的改进
- **Python原生集成**: 无需subprocess调用，直接在Python环境中运行
- **数据处理能力**: 利用NumPy/SciPy进行高效后处理
- **参数化建模**: 通过Python类和函数实现更灵活的参数化
- **错误处理**: Python异常机制提供更好的错误诊断
- **并行计算**: 更容易集成Python并行计算框架

### 1.2 PyFreeFEM核心特性
```python
import pyfreefem as pff
import numpy as np

# 直接在Python中定义有限元问题
problem = pff.Problem()
problem.mesh = pff.TriangleMesh.from_geometry(geometry)
problem.add_domain_integral(pff.grad(u) * pff.grad(v))
solution = problem.solve()
```

## 二、轧辊应力场模拟PyFreeFEM实现

### 2.1 基础设置和网格生成
```python
import pyfreefem as pff
import numpy as np
from dataclasses import dataclass

@dataclass
class RollingParameters:
    """轧制工艺参数"""
    roll_radius: float = 0.5  # 轧辊半径 (m)
    strip_thickness_initial: float = 0.01  # 初始厚度 (m)
    strip_thickness_final: float = 0.008  # 最终厚度 (m)
    rolling_speed: float = 3.8  # 轧制速度 (m/s)
    temperature_initial: float = 1123  # 初始温度 (K)
    friction_coefficient: float = 0.3
    
@dataclass  
class MaterialProperties:
    """材料属性（温度相关）"""
    density: float = 7850  # kg/m³
    youngs_modulus: float = 210e9  # Pa
    poisson_ratio: float = 0.3
    thermal_expansion: float = 1.2e-5  # 1/K
    thermal_conductivity: float = 45  # W/(m·K)
    specific_heat: float = 460  # J/(kg·K)
    
class RollingSimulation:
    def __init__(self, params: RollingParameters, material: MaterialProperties):
        self.params = params
        self.material = material
        self.mesh = None
        self.problem = None
        
    def create_geometry(self):
        """创建轧制几何模型"""
        # 定义轧辊和板材的几何边界
        roll_center_y = self.params.strip_thickness_final/2 + self.params.roll_radius
        
        # 使用PyFreeFEM的几何构建功能
        geometry = pff.Geometry()
        
        # 上轧辊圆弧
        roll_top = pff.Circle(
            center=(0, roll_center_y),
            radius=self.params.roll_radius,
            angle_range=(-np.pi/4, np.pi/4)
        )
        
        # 板材区域
        strip = pff.Rectangle(
            corner=(-0.2, -self.params.strip_thickness_final/2),
            width=0.4,
            height=self.params.strip_thickness_final
        )
        
        # 组合几何
        geometry.add_curve(roll_top, label="roll_surface")
        geometry.add_region(strip, label="workpiece")
        
        return geometry
        
    def generate_mesh(self, geometry):
        """生成自适应网格"""
        # 在接触区域细化网格
        mesh_size_function = lambda x, y: 0.0001 if abs(y) < 0.001 else 0.001
        
        self.mesh = pff.TriangleMesh.from_geometry(
            geometry,
            mesh_size=mesh_size_function,
            max_elements=50000
        )
        
        return self.mesh
```

### 2.2 热力耦合有限元求解
```python
    def setup_thermal_mechanical_problem(self):
        """设置热力耦合问题"""
        # 定义有限元空间
        V_temp = pff.FunctionSpace(self.mesh, "P1")  # 温度场
        V_disp = pff.VectorFunctionSpace(self.mesh, "P2", dim=2)  # 位移场
        
        # 定义试探函数和测试函数
        T = pff.TrialFunction(V_temp)
        v_T = pff.TestFunction(V_temp)
        u = pff.TrialFunction(V_disp)
        v_u = pff.TestFunction(V_disp)
        
        # 温度相关的材料属性
        E = self.material.youngs_modulus * (1 - 0.0005*(T - 293))  # 温度修正
        alpha = self.material.thermal_expansion
        
        # 热传导方程
        k = self.material.thermal_conductivity
        rho = self.material.density
        cp = self.material.specific_heat
        
        thermal_form = (
            rho * cp * pff.dot(T, v_T) * pff.dx +  # 瞬态项
            k * pff.dot(pff.grad(T), pff.grad(v_T)) * pff.dx  # 扩散项
        )
        
        # 力学平衡方程（考虑热应力）
        def stress_tensor(u, T):
            eps = pff.sym(pff.grad(u))  # 应变张量
            eps_thermal = alpha * (T - self.params.temperature_initial) * pff.Identity(2)
            eps_mechanical = eps - eps_thermal
            
            # Hooke定律
            lambda_ = E * self.material.poisson_ratio / ((1 + self.material.poisson_ratio) * (1 - 2*self.material.poisson_ratio))
            mu = E / (2 * (1 + self.material.poisson_ratio))
            
            return lambda_ * pff.tr(eps_mechanical) * pff.Identity(2) + 2 * mu * eps_mechanical
            
        mechanical_form = (
            pff.inner(stress_tensor(u, T), pff.grad(v_u)) * pff.dx
        )
        
        # 边界条件
        bc_temp = pff.DirichletBC(V_temp, self.params.temperature_initial, "inlet")
        bc_disp = pff.DirichletBC(V_disp, pff.Constant((0, 0)), "fixed")
        
        # 接触力
        contact_pressure = 150e6  # Pa
        contact_form = contact_pressure * pff.dot(pff.FacetNormal(self.mesh), v_u) * pff.ds("roll_surface")
        
        # 组装耦合问题
        self.thermal_problem = pff.LinearVariationalProblem(
            thermal_form, pff.Constant(0), T, bcs=[bc_temp]
        )
        
        self.mechanical_problem = pff.LinearVariationalProblem(
            mechanical_form - contact_form, pff.Constant((0, 0)), u, bcs=[bc_disp]
        )
```

### 2.3 求解和后处理
```python
    def solve_coupled_system(self, time_steps=100, dt=0.01):
        """求解热力耦合系统"""
        # 初始化解
        T_solution = pff.Function(self.thermal_problem.trial_space)
        u_solution = pff.Function(self.mechanical_problem.trial_space)
        
        # 时间步进求解
        solutions = []
        for t in range(time_steps):
            # 求解温度场
            self.thermal_solver = pff.LinearVariationalSolver(self.thermal_problem)
            self.thermal_solver.solve(T_solution)
            
            # 更新力学问题中的温度
            self.mechanical_problem.set_temperature(T_solution)
            
            # 求解位移场
            self.mechanical_solver = pff.LinearVariationalSolver(self.mechanical_problem)
            self.mechanical_solver.solve(u_solution)
            
            # 计算应力场
            stress = self.compute_stress_field(u_solution, T_solution)
            
            # 保存结果
            solutions.append({
                'time': t * dt,
                'temperature': T_solution.copy(),
                'displacement': u_solution.copy(),
                'stress': stress
            })
            
        return solutions
        
    def compute_stress_field(self, u, T):
        """计算应力场分布"""
        # 定义应力张量空间
        V_stress = pff.TensorFunctionSpace(self.mesh, "DG", 0)
        
        # 计算应力
        stress = pff.project(stress_tensor(u, T), V_stress)
        
        # 计算von Mises应力
        s = stress - (1/3) * pff.tr(stress) * pff.Identity(2)
        von_mises = pff.sqrt(3/2 * pff.inner(s, s))
        
        return {
            'stress_tensor': stress,
            'von_mises': von_mises,
            'max_principal': self.compute_principal_stress(stress)
        }
```

## 三、VTK格式导出实现

### 3.1 PyFreeFEM到VTK转换器
```python
import vtk
import meshio
from vtk.util.numpy_support import numpy_to_vtk

class FreeFEMToVTKExporter:
    def __init__(self, mesh, solutions):
        self.mesh = mesh
        self.solutions = solutions
        
    def export_to_vtk(self, filename_base):
        """导出时间序列VTK文件"""
        for idx, sol in enumerate(self.solutions):
            # 创建VTK非结构网格
            vtk_mesh = self._create_vtk_unstructured_grid()
            
            # 添加场数据
            self._add_field_data(vtk_mesh, sol)
            
            # 写入文件
            filename = f"{filename_base}_{idx:04d}.vtu"
            self._write_vtu_file(vtk_mesh, filename)
            
        # 创建时间序列文件
        self._create_pvd_file(filename_base, len(self.solutions))
            
    def _create_vtk_unstructured_grid(self):
        """创建VTK非结构网格"""
        # 获取网格数据
        points = self.mesh.coordinates()
        cells = self.mesh.cells()
        
        # 创建VTK点
        vtk_points = vtk.vtkPoints()
        for point in points:
            vtk_points.InsertNextPoint(point[0], point[1], 0.0)
            
        # 创建VTK单元
        vtk_cells = vtk.vtkCellArray()
        for cell in cells:
            triangle = vtk.vtkTriangle()
            for i, node_id in enumerate(cell):
                triangle.GetPointIds().SetId(i, node_id)
            vtk_cells.InsertNextCell(triangle)
            
        # 组装网格
        vtk_mesh = vtk.vtkUnstructuredGrid()
        vtk_mesh.SetPoints(vtk_points)
        vtk_mesh.SetCells(vtk.VTK_TRIANGLE, vtk_cells)
        
        return vtk_mesh
        
    def _add_field_data(self, vtk_mesh, solution):
        """添加场数据到VTK网格"""
        # 温度场
        temp_array = numpy_to_vtk(solution['temperature'].vector().array())
        temp_array.SetName("Temperature")
        vtk_mesh.GetPointData().AddArray(temp_array)
        
        # 位移场
        disp_data = solution['displacement'].compute_vertex_values(self.mesh)
        disp_array = numpy_to_vtk(disp_data.T)
        disp_array.SetName("Displacement")
        disp_array.SetNumberOfComponents(2)
        vtk_mesh.GetPointData().AddArray(disp_array)
        
        # von Mises应力
        stress_data = solution['stress']['von_mises'].compute_vertex_values(self.mesh)
        stress_array = numpy_to_vtk(stress_data)
        stress_array.SetName("VonMisesStress")
        vtk_mesh.GetCellData().AddArray(stress_array)
        
    def _write_vtu_file(self, vtk_mesh, filename):
        """写入VTU文件"""
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(vtk_mesh)
        writer.SetDataModeToBinary()
        writer.SetCompressorTypeToZLib()
        writer.Write()
        
    def _create_pvd_file(self, base_name, num_steps):
        """创建ParaView时间序列文件"""
        pvd_content = ['<?xml version="1.0"?>\n',
                      '<VTKFile type="Collection" version="0.1">\n',
                      '<Collection>\n']
        
        for i in range(num_steps):
            time = self.solutions[i]['time']
            filename = f"{base_name}_{i:04d}.vtu"
            pvd_content.append(f'<DataSet timestep="{time}" file="{filename}"/>\n')
            
        pvd_content.extend(['</Collection>\n', '</VTKFile>\n'])
        
        with open(f"{base_name}.pvd", 'w') as f:
            f.writelines(pvd_content)
```

### 3.2 高级VTK导出功能
```python
class AdvancedVTKExporter(FreeFEMToVTKExporter):
    """支持更多导出选项的VTK导出器"""
    
    def export_with_options(self, filename_base, options=None):
        """带选项的导出"""
        options = options or {}
        
        # 是否导出应力张量分量
        if options.get('export_stress_components', True):
            self._add_stress_components = True
            
        # 是否包含网格质量指标
        if options.get('include_mesh_quality', False):
            self._compute_mesh_quality()
            
        # 是否压缩输出
        self.compression_level = options.get('compression_level', 6)
        
        # 执行导出
        self.export_to_vtk(filename_base)
        
    def _add_stress_components(self, vtk_mesh, stress_tensor):
        """添加应力张量各分量"""
        components = ['xx', 'yy', 'xy']
        for i, comp in enumerate(components):
            data = stress_tensor[:, i].compute_vertex_values(self.mesh)
            array = numpy_to_vtk(data)
            array.SetName(f"Stress_{comp}")
            vtk_mesh.GetCellData().AddArray(array)
            
    def _compute_mesh_quality(self):
        """计算网格质量指标"""
        quality_measure = vtk.vtkMeshQuality()
        quality_measure.SetInputData(self.vtk_mesh)
        quality_measure.SetTriangleQualityMeasureToAspectRatio()
        quality_measure.Update()
        
        quality_array = quality_measure.GetOutput().GetCellData().GetArray("Quality")
        quality_array.SetName("MeshQuality")
        return quality_array
```

## 四、ParaviewWeb可视化服务架构

### 4.1 后端服务器设置
```python
# paraview_server.py
from wslink import server
from paraview import simple
from paraview.web import protocols as pv_protocols
import asyncio
import os

class RollingVisualizationProtocol(pv_protocols.ParaViewWebProtocol):
    """轧制模拟可视化协议"""
    
    def __init__(self):
        super().__init__()
        self.simulation_data_path = "/data/simulations"
        
    @server.expose
    def load_simulation(self, simulation_id):
        """加载模拟结果"""
        pvd_file = os.path.join(self.simulation_data_path, f"{simulation_id}.pvd")
        
        # 使用ParaView Python API加载数据
        reader = simple.XMLPartitionedUnstructuredGridReader(FileName=pvd_file)
        simple.Show(reader)
        
        # 设置默认视图
        self.setup_default_view(reader)
        
        return {"status": "success", "simulation_id": simulation_id}
        
    @server.expose
    def update_visualization(self, field_name, color_range=None):
        """更新可视化显示"""
        # 获取当前活动源
        source = simple.GetActiveSource()
        display = simple.GetDisplayProperties(source)
        
        # 设置显示字段
        display.SetRepresentationType('Surface')
        display.ColorArrayName = ['POINTS', field_name]
        
        # 设置颜色映射
        if color_range:
            color_map = simple.GetColorTransferFunction(field_name)
            color_map.RescaleTransferFunction(color_range[0], color_range[1])
            
        # 更新视图
        simple.Render()
        
        return {"status": "updated", "field": field_name}
        
    def setup_default_view(self, reader):
        """设置默认3D视图"""
        # 创建渲染视图
        view = simple.GetActiveViewOrCreate('RenderView')
        
        # 设置相机位置
        view.CameraPosition = [0.5, 0.5, 2.0]
        view.CameraFocalPoint = [0.0, 0.0, 0.0]
        view.CameraViewUp = [0.0, 1.0, 0.0]
        
        # 添加颜色条
        simple.GetScalarBar().Visibility = 1
        simple.GetScalarBar().Title = "Stress (MPa)"
        
        # 启用时间动画
        animation = simple.GetAnimationScene()
        animation.PlayMode = 'Sequence'
        
        return view

# 启动ParaviewWeb服务器
def start_paraview_server():
    """启动ParaviewWeb服务器"""
    # 服务器配置
    args = {
        "host": "0.0.0.0",
        "port": 9000,
        "ws": "ws://localhost:9000/ws",
        "content": "./www",
        "debug": True,
        "nosignalhandlers": True
    }
    
    # 创建协议实例
    protocol = RollingVisualizationProtocol()
    
    # 启动服务器
    server.start_webserver(options=args, protocol=protocol)
```

### 4.2 前端Web界面
```html
<!-- index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>轧制应力场可视化系统</title>
    <script src="https://unpkg.com/paraviewweb/dist/ParaViewWeb.js"></script>
    <style>
        #renderContainer {
            width: 100%;
            height: 600px;
            border: 1px solid #ccc;
        }
        .control-panel {
            padding: 20px;
            background: #f5f5f5;
        }
        .field-selector {
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>热轧过程应力场可视化</h1>
    
    <div class="control-panel">
        <div class="field-selector">
            <label>显示字段：</label>
            <select id="fieldSelect">
                <option value="VonMisesStress">Von Mises应力</option>
                <option value="Temperature">温度分布</option>
                <option value="Displacement">位移场</option>
                <option value="Stress_xx">应力分量XX</option>
                <option value="Stress_yy">应力分量YY</option>
            </select>
        </div>
        
        <div class="time-control">
            <label>时间步：</label>
            <input type="range" id="timeSlider" min="0" max="100" value="0">
            <span id="timeValue">0</span>
        </div>
        
        <div class="animation-control">
            <button id="playBtn">播放</button>
            <button id="pauseBtn">暂停</button>
            <button id="resetBtn">重置</button>
        </div>
    </div>
    
    <div id="renderContainer"></div>
    
    <script src="visualization.js"></script>
</body>
</html>
```

```javascript
// visualization.js
class RollingVisualization {
    constructor(container, wsUrl) {
        this.container = container;
        this.wsUrl = wsUrl;
        this.client = null;
        this.viewport = null;
        this.activeField = 'VonMisesStress';
        
        this.init();
    }
    
    async init() {
        // 连接到ParaviewWeb服务器
        const config = {
            sessionURL: this.wsUrl,
            application: 'rolling_visualization'
        };
        
        this.client = ParaViewWeb.createClient(config);
        
        await this.client.connect();
        
        // 创建视口
        this.viewport = this.client.getViewport();
        this.viewport.setContainer(this.container);
        
        // 加载模拟数据
        await this.loadSimulation('hot_rolling_001');
        
        // 设置事件处理
        this.setupEventHandlers();
    }
    
    async loadSimulation(simulationId) {
        const result = await this.client.call('load_simulation', [simulationId]);
        console.log('Simulation loaded:', result);
        
        // 初始显示
        await this.updateVisualization(this.activeField);
    }
    
    async updateVisualization(fieldName) {
        this.activeField = fieldName;
        
        // 根据字段类型设置合适的颜色范围
        let colorRange = null;
        switch(fieldName) {
            case 'VonMisesStress':
                colorRange = [0, 500e6];  // 0-500 MPa
                break;
            case 'Temperature':
                colorRange = [293, 1500];  // 20-1227°C
                break;
            case 'Displacement':
                colorRange = [0, 0.005];   // 0-5mm
                break;
        }
        
        await this.client.call('update_visualization', [fieldName, colorRange]);
        this.viewport.render();
    }
    
    setupEventHandlers() {
        // 字段选择
        document.getElementById('fieldSelect').addEventListener('change', (e) => {
            this.updateVisualization(e.target.value);
        });
        
        // 时间控制
        const timeSlider = document.getElementById('timeSlider');
        const timeValue = document.getElementById('timeValue');
        
        timeSlider.addEventListener('input', (e) => {
            const time = parseInt(e.target.value);
            timeValue.textContent = time;
            this.setTimeStep(time);
        });
        
        // 动画控制
        document.getElementById('playBtn').addEventListener('click', () => {
            this.playAnimation();
        });
        
        document.getElementById('pauseBtn').addEventListener('click', () => {
            this.pauseAnimation();
        });
        
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetAnimation();
        });
    }
    
    async setTimeStep(step) {
        await this.client.call('set_time_step', [step]);
        this.viewport.render();
    }
    
    async playAnimation() {
        await this.client.call('play_animation', []);
    }
    
    async pauseAnimation() {
        await this.client.call('pause_animation', []);
    }
    
    async resetAnimation() {
        await this.client.call('reset_animation', []);
        document.getElementById('timeSlider').value = 0;
        document.getElementById('timeValue').textContent = '0';
    }
}

// 初始化可视化
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('renderContainer');
    const wsUrl = 'ws://localhost:9000/ws';
    
    new RollingVisualization(container, wsUrl);
});
```

## 五、完整集成架构

### 5.1 Docker部署配置
```yaml
# docker-compose.yml
version: '3.8'

services:
  # FreeFEM计算服务
  freefem-compute:
    build:
      context: .
      dockerfile: Dockerfile.freefem
    volumes:
      - ./simulations:/data/simulations
      - ./scripts:/scripts
    environment:
      - PYTHONPATH=/app
    networks:
      - simulation-network
      
  # FastAPI后端服务
  api-backend:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./simulations:/data/simulations
    environment:
      - PARAVIEW_URL=http://paraview-server:9000
    depends_on:
      - freefem-compute
      - paraview-server
    networks:
      - simulation-network
      
  # ParaviewWeb服务
  paraview-server:
    image: kitware/paraviewweb:5.10
    ports:
      - "9000:9000"
    volumes:
      - ./simulations:/data/simulations
      - ./pvw-config:/opt/paraview/config
    environment:
      - DISPLAY=:99
    networks:
      - simulation-network
      
  # Nginx反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./frontend:/usr/share/nginx/html
    depends_on:
      - api-backend
      - paraview-server
    networks:
      - simulation-network

networks:
  simulation-network:
    driver: bridge
```

### 5.2 主控制API
```python
# main_api.py
from fastapi import FastAPI, BackgroundTasks, WebSocket
from fastapi.staticfiles import StaticFiles
import asyncio
import json

app = FastAPI(title="轧制应力场分析系统")

# 挂载前端静态文件
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.post("/api/simulation/create")
async def create_simulation(params: RollingParameters, background_tasks: BackgroundTasks):
    """创建新的轧制模拟"""
    simulation_id = generate_simulation_id()
    
    # 后台执行模拟
    background_tasks.add_task(
        run_rolling_simulation,
        simulation_id,
        params
    )
    
    return {"simulation_id": simulation_id, "status": "started"}

@app.websocket("/ws/simulation/{simulation_id}")
async def simulation_progress(websocket: WebSocket, simulation_id: str):
    """WebSocket实时进度推送"""
    await websocket.accept()
    
    while True:
        progress = get_simulation_progress(simulation_id)
        await websocket.send_json({
            "type": "progress",
            "data": progress
        })
        
        if progress["status"] == "completed":
            break
            
        await asyncio.sleep(1)
        
    await websocket.close()

async def run_rolling_simulation(simulation_id: str, params: RollingParameters):
    """执行完整的模拟流程"""
    try:
        # 1. 初始化PyFreeFEM模拟
        sim = RollingSimulation(params, MaterialProperties())
        
        # 2. 创建几何和网格
        geometry = sim.create_geometry()
        mesh = sim.generate_mesh(geometry)
        
        # 3. 设置和求解问题
        sim.setup_thermal_mechanical_problem()
        solutions = sim.solve_coupled_system()
        
        # 4. 导出VTK格式
        exporter = AdvancedVTKExporter(mesh, solutions)
        exporter.export_with_options(
            f"/data/simulations/{simulation_id}",
            options={
                'export_stress_components': True,
                'include_mesh_quality': True,
                'compression_level': 6
            }
        )
        
        # 5. 通知ParaviewWeb服务器
        notify_paraview_server(simulation_id)
        
        # 6. 更新状态
        update_simulation_status(simulation_id, "completed")
        
    except Exception as e:
        update_simulation_status(simulation_id, "failed", str(e))
        raise
```

## 六、实施步骤总结

### 6.1 环境准备
1. 安装Python 3.8+和相关依赖
```bash
pip install pyfreefem numpy scipy fastapi uvicorn vtk meshio websockets
```

2. 安装ParaView和ParaviewWeb
```bash
# Ubuntu/Debian
apt-get install paraview python3-paraview
pip install wslink
```

3. 配置FreeFEM环境
```bash
# 确保FreeFEM已安装
freefem++ --version
```

### 6.2 开发流程
1. **使用PyFreeFEM开发轧辊应力场模拟**
   - 定义几何模型和材料参数
   - 实现热力耦合有限元求解
   - 验证计算结果

2. **实现VTK导出功能**
   - 开发FreeFEM到VTK格式转换器
   - 支持时间序列数据导出
   - 添加所有必要的场数据

3. **搭建ParaviewWeb服务**
   - 配置ParaviewWeb服务器
   - 开发前端可视化界面
   - 实现实时交互功能

4. **集成部署**
   - 使用Docker容器化部署
   - 配置负载均衡和反向代理
   - 实现自动化运维

### 6.3 优化建议
1. **性能优化**
   - 使用并行计算加速求解
   - 实施结果缓存机制
   - 优化网格自适应策略

2. **用户体验**
   - 添加预设参数模板
   - 实现批量任务处理
   - 提供结果对比功能

3. **扩展性**
   - 支持更多材料模型
   - 添加多道次轧制模拟
   - 集成机器学习预测

这个完整方案实现了从PyFreeFEM模拟到ParaviewWeb可视化的全流程，提供了工业级的轧辊应力场分析能力。
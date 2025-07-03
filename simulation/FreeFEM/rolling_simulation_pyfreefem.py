#!/usr/bin/env python3
"""
PyFreeFEM轧辊应力场模拟完整实现
用于热轧过程的热力耦合有限元分析
"""

import numpy as np
import pyfreefem as pff
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RollingParameters:
    """轧制工艺参数"""
    roll_radius: float = 0.5  # 轧辊半径 (m)
    roll_gap: float = 0.008  # 辊缝 (m)
    strip_width: float = 2.0  # 板宽 (m)
    strip_thickness_initial: float = 0.025  # 初始厚度 (m)
    strip_thickness_final: float = 0.020  # 最终厚度 (m)
    rolling_speed: float = 3.8  # 轧制速度 (m/s)
    entry_speed: float = 3.0  # 入口速度 (m/s)
    temperature_initial: float = 1123  # 初始温度 (K)
    temperature_roll: float = 423  # 轧辊温度 (K)
    friction_coefficient: float = 0.3  # 摩擦系数
    contact_length: float = 0.1  # 接触弧长 (m)
    
    def __post_init__(self):
        """验证参数合理性"""
        if self.strip_thickness_final >= self.strip_thickness_initial:
            raise ValueError("最终厚度必须小于初始厚度")
        if self.roll_gap > self.strip_thickness_final:
            raise ValueError("辊缝必须小于最终厚度")
        
    @property
    def reduction_ratio(self) -> float:
        """压下率"""
        return (self.strip_thickness_initial - self.strip_thickness_final) / self.strip_thickness_initial
    
    @property
    def neutral_angle(self) -> float:
        """中性角估算"""
        return np.sqrt(self.reduction_ratio * self.strip_thickness_initial / self.roll_radius)


@dataclass
class MaterialProperties:
    """材料属性（碳钢）"""
    material_name: str = "Carbon Steel"
    density: float = 7850  # kg/m³
    
    # 力学性能
    youngs_modulus_ref: float = 210e9  # Pa at 20°C
    poisson_ratio: float = 0.3
    yield_strength_ref: float = 350e6  # Pa at 20°C
    
    # 热性能
    thermal_conductivity_ref: float = 45  # W/(m·K) at 20°C
    specific_heat_ref: float = 460  # J/(kg·K)
    thermal_expansion: float = 1.2e-5  # 1/K
    
    # 参考温度
    reference_temperature: float = 293  # K (20°C)
    melting_temperature: float = 1811  # K
    
    def youngs_modulus(self, T: float) -> float:
        """温度相关的弹性模量"""
        # E(T) = E₀ * (1 - α*(T - T_ref))
        alpha = 0.0004  # 温度系数
        return self.youngs_modulus_ref * (1 - alpha * (T - self.reference_temperature))
    
    def yield_strength(self, T: float, strain_rate: float = 1.0) -> float:
        """温度和应变率相关的屈服强度"""
        # σ_y = σ_y0 * (T_m - T)/(T_m - T_ref) * (ε̇/ε̇₀)^m
        temp_factor = (self.melting_temperature - T) / (self.melting_temperature - self.reference_temperature)
        strain_rate_sensitivity = 0.1  # m值
        strain_rate_factor = (strain_rate / 1.0) ** strain_rate_sensitivity
        return self.yield_strength_ref * temp_factor * strain_rate_factor
    
    def thermal_conductivity(self, T: float) -> float:
        """温度相关的热导率"""
        # k(T) = k₀ * (1 - β*(T - T_ref))
        beta = 0.0002
        return self.thermal_conductivity_ref * (1 - beta * (T - self.reference_temperature))
    
    def specific_heat(self, T: float) -> float:
        """温度相关的比热容"""
        # cp(T) = cp₀ * (1 + γ*(T - T_ref))
        gamma = 0.0001
        return self.specific_heat_ref * (1 + gamma * (T - self.reference_temperature))


class RollingGeometry:
    """轧制几何建模"""
    
    def __init__(self, params: RollingParameters):
        self.params = params
        self.geometry = None
        self.boundaries = {}
        
    def create_2d_geometry(self) -> pff.Geometry:
        """创建2D轧制几何（纵截面）"""
        logger.info("创建2D轧制几何模型")
        
        # 几何参数
        L = 0.4  # 模型长度 (m)
        h0 = self.params.strip_thickness_initial
        h1 = self.params.strip_thickness_final
        R = self.params.roll_radius
        
        # 计算接触弧投影长度
        contact_length = np.sqrt(R * (h0 - h1))
        
        # 创建几何对象
        geo = pff.Geometry()
        
        # 定义关键点
        points = {
            'entry_top': (-L/2, h0/2),
            'entry_bottom': (-L/2, -h0/2),
            'contact_start_top': (-contact_length/2, h0/2),
            'contact_start_bottom': (-contact_length/2, -h0/2),
            'contact_end_top': (contact_length/2, h1/2),
            'contact_end_bottom': (contact_length/2, -h1/2),
            'exit_top': (L/2, h1/2),
            'exit_bottom': (L/2, -h1/2)
        }
        
        # 添加边界
        # 入口段
        geo.add_line(points['entry_bottom'], points['entry_top'], label='inlet')
        geo.add_line(points['entry_top'], points['contact_start_top'], label='top_entry')
        geo.add_line(points['entry_bottom'], points['contact_start_bottom'], label='bottom_entry')
        
        # 接触段（简化为直线，实际应为圆弧）
        geo.add_line(points['contact_start_top'], points['contact_end_top'], label='roll_contact_top')
        geo.add_line(points['contact_start_bottom'], points['contact_end_bottom'], label='roll_contact_bottom')
        
        # 出口段
        geo.add_line(points['contact_end_top'], points['exit_top'], label='top_exit')
        geo.add_line(points['contact_end_bottom'], points['exit_bottom'], label='bottom_exit')
        geo.add_line(points['exit_bottom'], points['exit_top'], label='outlet')
        
        # 保存边界标签
        self.boundaries = {
            'inlet': 1,
            'outlet': 2,
            'roll_contact_top': 3,
            'roll_contact_bottom': 4,
            'free_surface': 5
        }
        
        self.geometry = geo
        return geo
    
    def create_3d_geometry(self) -> pff.Geometry:
        """创建3D轧制几何（包含板宽方向）"""
        # 3D实现略，需要更复杂的几何定义
        raise NotImplementedError("3D几何建模待实现")


class AdaptiveMeshGenerator:
    """自适应网格生成器"""
    
    def __init__(self, geometry: pff.Geometry, params: RollingParameters):
        self.geometry = geometry
        self.params = params
        self.mesh = None
        
    def generate_initial_mesh(self, max_element_size: float = 0.001) -> pff.Mesh:
        """生成初始网格"""
        logger.info("生成初始有限元网格")
        
        # 定义网格尺寸函数（接触区细化）
        def mesh_size_function(x: float, y: float) -> float:
            # 接触区域判断
            contact_zone = abs(x) < self.params.contact_length / 2
            near_surface = abs(y) > 0.8 * self.params.strip_thickness_final / 2
            
            if contact_zone and near_surface:
                return max_element_size * 0.1  # 接触区细化10倍
            elif contact_zone or near_surface:
                return max_element_size * 0.5  # 部分细化
            else:
                return max_element_size
        
        # 生成网格
        self.mesh = pff.generate_mesh(
            self.geometry,
            mesh_size=mesh_size_function,
            mesh_algorithm="Delaunay"
        )
        
        logger.info(f"网格生成完成：{self.mesh.num_vertices()} 个节点, {self.mesh.num_cells()} 个单元")
        return self.mesh
    
    def refine_mesh(self, error_indicator: pff.Function, tolerance: float = 0.01) -> pff.Mesh:
        """基于误差指示器的网格自适应"""
        logger.info("执行网格自适应细化")
        
        # 标记需要细化的单元
        markers = pff.MeshFunction("bool", self.mesh, self.mesh.topology().dim())
        markers.set_all(False)
        
        # 计算误差阈值
        max_error = error_indicator.vector().max()
        threshold = tolerance * max_error
        
        # 标记高误差单元
        for cell in pff.cells(self.mesh):
            if error_indicator[cell] > threshold:
                markers[cell] = True
        
        # 执行细化
        self.mesh = pff.refine(self.mesh, markers)
        
        logger.info(f"细化后网格：{self.mesh.num_vertices()} 个节点, {self.mesh.num_cells()} 个单元")
        return self.mesh


class ThermalMechanicalSolver:
    """热力耦合求解器"""
    
    def __init__(self, mesh: pff.Mesh, params: RollingParameters, material: MaterialProperties):
        self.mesh = mesh
        self.params = params
        self.material = material
        
        # 定义函数空间
        self._setup_function_spaces()
        
        # 初始化解
        self.temperature = None
        self.displacement = None
        self.stress = None
        
    def _setup_function_spaces(self):
        """设置有限元函数空间"""
        # 温度场：一阶拉格朗日元
        self.V_temp = pff.FunctionSpace(self.mesh, "Lagrange", 1)
        
        # 位移场：二阶矢量拉格朗日元
        self.V_disp = pff.VectorFunctionSpace(self.mesh, "Lagrange", 2)
        
        # 应力场：零阶不连续张量元
        self.V_stress = pff.TensorFunctionSpace(self.mesh, "DG", 0)
        
        logger.info(f"函数空间维度 - 温度: {self.V_temp.dim()}, 位移: {self.V_disp.dim()}")
    
    def solve_thermal_problem(self, dt: float = 0.001) -> pff.Function:
        """求解瞬态热传导问题"""
        logger.info("求解温度场...")
        
        # 定义变分形式
        T = pff.TrialFunction(self.V_temp)
        v = pff.TestFunction(self.V_temp)
        
        # 上一时间步温度（初始为均匀温度）
        if self.temperature is None:
            self.temperature = pff.Function(self.V_temp)
            self.temperature.vector()[:] = self.params.temperature_initial
        
        T_old = self.temperature
        
        # 材料参数（简化为常数）
        k = self.material.thermal_conductivity_ref
        rho = self.material.density
        cp = self.material.specific_heat_ref
        
        # 瞬态热传导方程
        F = (rho * cp / dt) * (T - T_old) * v * pff.dx \
            + k * pff.dot(pff.grad(T), pff.grad(v)) * pff.dx
        
        # 边界条件
        bc_inlet = pff.DirichletBC(self.V_temp, self.params.temperature_initial, "inlet")
        bc_roll = pff.DirichletBC(self.V_temp, self.params.temperature_roll, "roll_contact_top")
        bcs = [bc_inlet, bc_roll]
        
        # 添加对流换热（自由表面）
        h_conv = 100  # 对流换热系数 W/(m²·K)
        T_ambient = 293  # 环境温度 K
        F += h_conv * (T - T_ambient) * v * pff.ds("free_surface")
        
        # 求解
        a = pff.lhs(F)
        L = pff.rhs(F)
        
        T_new = pff.Function(self.V_temp)
        pff.solve(a == L, T_new, bcs)
        
        self.temperature = T_new
        
        logger.info(f"温度场求解完成: T_min={T_new.vector().min():.1f}K, T_max={T_new.vector().max():.1f}K")
        return self.temperature
    
    def solve_mechanical_problem(self) -> Tuple[pff.Function, pff.Function]:
        """求解力学平衡问题（考虑热应力）"""
        logger.info("求解位移场和应力场...")
        
        # 定义变分形式
        u = pff.TrialFunction(self.V_disp)
        v = pff.TestFunction(self.V_disp)
        
        # 温度场插值到位移空间
        T_disp = pff.interpolate(self.temperature, self.V_disp.sub(0).collapse())
        
        # 材料参数
        E = self.material.youngs_modulus_ref  # 简化处理
        nu = self.material.poisson_ratio
        alpha = self.material.thermal_expansion
        
        # Lamé常数
        mu = E / (2 * (1 + nu))
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        # 应变和应力定义
        def epsilon(u):
            return pff.sym(pff.grad(u))
        
        def sigma(u, T):
            # 热应变
            eps_thermal = alpha * (T - self.params.temperature_initial) * pff.Identity(2)
            # 力学应变
            eps_mech = epsilon(u) - eps_thermal
            # 应力（Hooke定律）
            return lmbda * pff.tr(eps_mech) * pff.Identity(2) + 2 * mu * eps_mech
        
        # 平衡方程
        F = pff.inner(sigma(u, T_disp), epsilon(v)) * pff.dx
        
        # 边界条件
        bc_fixed = pff.DirichletBC(self.V_disp, pff.Constant((0, 0)), "inlet")
        
        # 接触压力（简化为均布压力）
        contact_pressure = 150e6  # Pa
        n = pff.FacetNormal(self.mesh)
        F -= contact_pressure * pff.dot(n, v) * pff.ds("roll_contact_top")
        F -= contact_pressure * pff.dot(n, v) * pff.ds("roll_contact_bottom")
        
        # 求解
        a = pff.lhs(F)
        L = pff.rhs(F)
        
        u_solution = pff.Function(self.V_disp)
        pff.solve(a == L, u_solution, bc_fixed)
        
        self.displacement = u_solution
        
        # 计算应力场
        self.stress = pff.project(sigma(u_solution, T_disp), self.V_stress)
        
        # 计算von Mises应力
        s = self.stress - (1./3) * pff.tr(self.stress) * pff.Identity(2)
        von_mises = pff.sqrt(3./2 * pff.inner(s, s))
        self.von_mises = pff.project(von_mises, self.V_temp)  # 投影到标量空间
        
        logger.info(f"位移场求解完成: u_max={np.max(np.abs(u_solution.vector().get_local()))*1000:.2f}mm")
        logger.info(f"von Mises应力: σ_max={self.von_mises.vector().max()/1e6:.1f}MPa")
        
        return self.displacement, self.stress
    
    def compute_rolling_force(self) -> float:
        """计算总轧制力"""
        # 通过接触面上的应力积分计算
        n = pff.FacetNormal(self.mesh)
        force_density = pff.dot(self.stress * n, n)
        
        # 在接触边界上积分
        rolling_force = pff.assemble(
            force_density * pff.ds("roll_contact_top") + 
            force_density * pff.ds("roll_contact_bottom")
        )
        
        # 考虑板宽
        total_force = rolling_force * self.params.strip_width
        
        logger.info(f"总轧制力: F={total_force/1e6:.1f} MN")
        return total_force


class RollingSimulation:
    """轧制过程完整模拟"""
    
    def __init__(self, params: RollingParameters, material: MaterialProperties):
        self.params = params
        self.material = material
        
        # 组件
        self.geometry = None
        self.mesh = None
        self.solver = None
        
        # 结果存储
        self.results = []
        
    def setup(self):
        """设置模拟"""
        logger.info("=== 轧制模拟初始化 ===")
        logger.info(f"材料: {self.material.material_name}")
        logger.info(f"压下率: {self.params.reduction_ratio*100:.1f}%")
        logger.info(f"初始温度: {self.params.temperature_initial-273:.0f}°C")
        
        # 创建几何
        geo_builder = RollingGeometry(self.params)
        self.geometry = geo_builder.create_2d_geometry()
        
        # 生成网格
        mesh_gen = AdaptiveMeshGenerator(self.geometry, self.params)
        self.mesh = mesh_gen.generate_initial_mesh()
        
        # 初始化求解器
        self.solver = ThermalMechanicalSolver(self.mesh, self.params, self.material)
        
    def run(self, time_steps: int = 50, dt: float = 0.001) -> List[Dict]:
        """运行瞬态模拟"""
        logger.info(f"=== 开始瞬态模拟 (步数={time_steps}, dt={dt}s) ===")
        
        for step in range(time_steps):
            current_time = step * dt
            logger.info(f"\n时间步 {step+1}/{time_steps}, t={current_time:.3f}s")
            
            # 求解温度场
            temperature = self.solver.solve_thermal_problem(dt)
            
            # 求解位移和应力场
            displacement, stress = self.solver.solve_mechanical_problem()
            
            # 计算轧制力
            rolling_force = self.solver.compute_rolling_force()
            
            # 保存结果
            result = {
                'time': current_time,
                'step': step,
                'temperature': temperature.copy(deepcopy=True),
                'displacement': displacement.copy(deepcopy=True),
                'stress': stress.copy(deepcopy=True),
                'von_mises': self.solver.von_mises.copy(deepcopy=True),
                'rolling_force': rolling_force
            }
            self.results.append(result)
            
            # 自适应网格（每10步）
            if (step + 1) % 10 == 0 and step < time_steps - 1:
                self._adaptive_refinement()
        
        logger.info("\n=== 模拟完成 ===")
        self._print_summary()
        
        return self.results
    
    def _adaptive_refinement(self):
        """网格自适应细化"""
        # 基于von Mises应力的误差指示器
        error_indicator = self.solver.von_mises
        
        mesh_gen = AdaptiveMeshGenerator(self.geometry, self.params)
        mesh_gen.mesh = self.mesh
        new_mesh = mesh_gen.refine_mesh(error_indicator, tolerance=0.05)
        
        # 更新网格和求解器
        self.mesh = new_mesh
        self.solver = ThermalMechanicalSolver(self.mesh, self.params, self.material)
        
    def _print_summary(self):
        """打印模拟总结"""
        # 提取关键结果
        final_result = self.results[-1]
        max_temp = max(r['temperature'].vector().max() for r in self.results)
        max_stress = max(r['von_mises'].vector().max() for r in self.results)
        avg_force = np.mean([r['rolling_force'] for r in self.results])
        
        print("\n" + "="*50)
        print("轧制模拟结果总结")
        print("="*50)
        print(f"最高温度: {max_temp-273:.1f}°C")
        print(f"最大von Mises应力: {max_stress/1e6:.1f} MPa")
        print(f"平均轧制力: {avg_force/1e6:.1f} MN")
        print(f"最终厚度: {self.params.strip_thickness_final*1000:.1f} mm")
        print(f"压下率: {self.params.reduction_ratio*100:.1f}%")
        print("="*50)
    
    def save_results(self, output_dir: str = "results"):
        """保存结果文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存网格
        pff.File(f"{output_dir}/mesh.pvd") << self.mesh
        
        # 保存时间序列结果
        temp_file = pff.File(f"{output_dir}/temperature.pvd")
        disp_file = pff.File(f"{output_dir}/displacement.pvd")
        stress_file = pff.File(f"{output_dir}/von_mises.pvd")
        
        for result in self.results:
            temp_file << (result['temperature'], result['time'])
            disp_file << (result['displacement'], result['time'])
            stress_file << (result['von_mises'], result['time'])
        
        # 保存轧制力历史
        import json
        force_history = {
            'time': [r['time'] for r in self.results],
            'rolling_force': [r['rolling_force'] for r in self.results]
        }
        with open(f"{output_dir}/rolling_force.json", 'w') as f:
            json.dump(force_history, f, indent=2)
        
        logger.info(f"结果已保存到 {output_dir}/")


def main():
    """主程序"""
    # 设置轧制参数
    rolling_params = RollingParameters(
        roll_radius=0.5,
        strip_thickness_initial=0.025,
        strip_thickness_final=0.020,
        rolling_speed=3.8,
        temperature_initial=1123,  # 850°C
        friction_coefficient=0.3
    )
    
    # 设置材料属性
    material = MaterialProperties()
    
    # 创建并运行模拟
    simulation = RollingSimulation(rolling_params, material)
    simulation.setup()
    results = simulation.run(time_steps=50, dt=0.001)
    
    # 保存结果
    simulation.save_results("rolling_results")
    
    return simulation, results


if __name__ == "__main__":
    simulation, results = main()
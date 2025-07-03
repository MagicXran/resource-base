# -*- coding: gb2312 -*-
"""
FreeFEM模板引擎 - 动态生成FreeFEM脚本
支持热轧过程的热力耦合分析
"""

import os
import json
import logging
from pathlib import Path
from string import Template
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='gb2312'
)
logger = logging.getLogger(__name__)


@dataclass
class MaterialProperty:
    """材料属性数据类"""
    name: str = "Carbon Steel"
    density: float = 7850.0  # kg/m³
    youngs_modulus: float = 210e9  # Pa
    poisson_ratio: float = 0.3
    thermal_expansion: float = 1.2e-5  # 1/K
    thermal_conductivity: float = 45.0  # W/(m·K)
    specific_heat: float = 460.0  # J/(kg·K)
    yield_strength: float = 350e6  # Pa
    
    def validate(self) -> bool:
        """验证材料属性的合理性"""
        if self.density <= 0:
            raise ValueError("密度必须大于0")
        if self.youngs_modulus <= 0:
            raise ValueError("弹性模量必须大于0")
        if not 0 <= self.poisson_ratio < 0.5:
            raise ValueError("泊松比必须在[0, 0.5)范围内")
        if self.thermal_expansion <= 0:
            raise ValueError("热膨胀系数必须大于0")
        if self.thermal_conductivity <= 0:
            raise ValueError("热导率必须大于0")
        if self.specific_heat <= 0:
            raise ValueError("比热容必须大于0")
        return True


@dataclass 
class RollingParameter:
    """轧制参数数据类"""
    roll_radius: float = 0.5  # m
    thickness_initial: float = 0.025  # m
    thickness_final: float = 0.020  # m
    strip_width: float = 2.0  # m
    roll_speed: float = 3.8  # m/s
    temperature_initial: float = 1123.0  # K
    temperature_roll: float = 423.0  # K
    temperature_ambient: float = 293.0  # K
    friction_coefficient: float = 0.3
    contact_pressure: float = 150e6  # Pa
    heat_transfer_coefficient: float = 1000.0  # W/(m²·K)
    
    @property
    def reduction_ratio(self) -> float:
        """计算压下率"""
        return (self.thickness_initial - self.thickness_final) / self.thickness_initial
    
    @property
    def contact_length(self) -> float:
        """计算接触弧长（近似）"""
        import math
        return math.sqrt(self.roll_radius * (self.thickness_initial - self.thickness_final))
    
    def validate(self) -> bool:
        """验证轧制参数的合理性"""
        if self.roll_radius <= 0:
            raise ValueError("轧辊半径必须大于0")
        if self.thickness_initial <= 0:
            raise ValueError("初始厚度必须大于0")
        if self.thickness_final <= 0:
            raise ValueError("最终厚度必须大于0")
        if self.thickness_final >= self.thickness_initial:
            raise ValueError("最终厚度必须小于初始厚度")
        if self.reduction_ratio > 0.8:
            raise ValueError("压下率不能超过80%")
        if self.strip_width <= 0:
            raise ValueError("板宽必须大于0")
        if self.roll_speed <= 0:
            raise ValueError("轧制速度必须大于0")
        if self.temperature_initial < 273:
            raise ValueError("初始温度必须高于0°C")
        if self.temperature_roll < 273:
            raise ValueError("轧辊温度必须高于0°C")
        if not 0 <= self.friction_coefficient <= 1:
            raise ValueError("摩擦系数必须在[0, 1]范围内")
        if self.contact_pressure <= 0:
            raise ValueError("接触压力必须大于0")
        return True


@dataclass
class MeshParameter:
    """网格参数数据类"""
    element_size: float = 0.001  # m
    refinement_zones: List[Dict[str, Any]] = field(default_factory=list)
    max_elements: int = 100000
    adaptive_refinement: bool = True
    refinement_tolerance: float = 0.01
    boundary_layer_elements: int = 5
    
    def validate(self) -> bool:
        """验证网格参数"""
        if self.element_size <= 0:
            raise ValueError("网格尺寸必须大于0")
        if self.max_elements <= 0:
            raise ValueError("最大单元数必须大于0")
        if self.refinement_tolerance <= 0:
            raise ValueError("细化容差必须大于0")
        return True


@dataclass
class SolverParameter:
    """求解器参数数据类"""
    time_step: float = 0.001  # s
    total_time: float = 0.1  # s
    max_iterations: int = 1000
    tolerance: float = 1e-6
    solver_type: str = "direct"  # direct, iterative
    preconditioner: str = "ilu"  # ilu, amg, none
    output_interval: int = 5
    
    @property
    def num_steps(self) -> int:
        """计算时间步数"""
        return int(self.total_time / self.time_step)
    
    def validate(self) -> bool:
        """验证求解器参数"""
        if self.time_step <= 0:
            raise ValueError("时间步长必须大于0")
        if self.total_time <= 0:
            raise ValueError("总时间必须大于0")
        if self.time_step > self.total_time:
            raise ValueError("时间步长不能大于总时间")
        if self.max_iterations <= 0:
            raise ValueError("最大迭代次数必须大于0")
        if self.tolerance <= 0:
            raise ValueError("收敛容差必须大于0")
        if self.solver_type not in ["direct", "iterative"]:
            raise ValueError("求解器类型必须是'direct'或'iterative'")
        return True


class FreeFEMTemplateEngine:
    """FreeFEM脚本模板引擎"""
    
    def __init__(self, template_dir: str = "templates", output_dir: str = "work"):
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        
        # 创建必要的目录
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模板缓存
        self._template_cache: Dict[str, Template] = {}
        
        logger.info(f"FreeFEM模板引擎初始化完成 - 模板目录: {self.template_dir}, 输出目录: {self.output_dir}")
    
    def create_rolling_template(self) -> str:
        """创建轧制模拟的FreeFEM模板"""
        template_content = '''// FreeFEM++ 热轧过程热力耦合模拟
// 自动生成时间: ${timestamp}
// 任务ID: ${task_id}

// ============================================
// 1. 参数定义
// ============================================

// 几何参数
real rollRadius = ${roll_radius};           // 轧辊半径 [m]
real h0 = ${thickness_initial};            // 初始厚度 [m]
real h1 = ${thickness_final};              // 最终厚度 [m]
real stripWidth = ${strip_width};          // 板宽 [m]
real rollSpeed = ${roll_speed};            // 轧制速度 [m/s]
real contactLength = ${contact_length};     // 接触弧长 [m]

// 温度参数
real T0 = ${temperature_initial};          // 初始温度 [K]
real Troll = ${temperature_roll};          // 轧辊温度 [K]
real Tambient = ${temperature_ambient};    // 环境温度 [K]

// 材料属性
real rho = ${density};                     // 密度 [kg/m³]
real E0 = ${youngs_modulus};              // 弹性模量 [Pa]
real nu = ${poisson_ratio};               // 泊松比
real alpha = ${thermal_expansion};        // 热膨胀系数 [1/K]
real k0 = ${thermal_conductivity};        // 热导率 [W/(m·K)]
real cp = ${specific_heat};               // 比热容 [J/(kg·K)]
real sigma_y0 = ${yield_strength};        // 屈服强度 [Pa]

// 接触参数
real frictionCoeff = ${friction_coefficient};  // 摩擦系数
real contactPressure = ${contact_pressure};    // 接触压力 [Pa]
real heatTransfer = ${heat_transfer_coefficient}; // 换热系数 [W/(m²·K)]

// 网格参数
real meshSize = ${element_size};          // 基准网格尺寸 [m]
int maxElements = ${max_elements};        // 最大单元数

// 求解参数
real dt = ${time_step};                   // 时间步长 [s]
int nsteps = ${num_steps};                // 时间步数
real tolerance = ${tolerance};            // 收敛容差
int outputInterval = ${output_interval};  // 输出间隔

// 计算导出参数
real reduction = (h0 - h1) / h0;          // 压下率
real strainRate = rollSpeed * reduction / contactLength; // 应变率估算

// ============================================
// 2. 几何定义
// ============================================

// 模型尺寸
real L = 4.0 * contactLength;             // 模型长度
real entryLength = 1.5 * contactLength;   // 入口段长度
real exitLength = 1.5 * contactLength;    // 出口段长度

// 定义边界
// 入口段
border inlet(t=0,1) { 
    x = -L/2; 
    y = h0*(t-0.5); 
    label = 1; 
}

// 出口段
border outlet(t=0,1) { 
    x = L/2; 
    y = h1*(t-0.5); 
    label = 2; 
}

// 上表面（入口段）
border topEntry(t=0,1) { 
    x = -L/2 + t*entryLength; 
    y = h0/2; 
    label = 3; 
}

// 下表面（入口段）
border bottomEntry(t=0,1) { 
    x = -L/2 + t*entryLength; 
    y = -h0/2; 
    label = 4; 
}

// 上表面（出口段）
border topExit(t=0,1) { 
    x = contactLength/2 + t*exitLength; 
    y = h1/2; 
    label = 5; 
}

// 下表面（出口段）
border bottomExit(t=0,1) { 
    x = contactLength/2 + t*exitLength; 
    y = -h1/2; 
    label = 6; 
}

// 接触区域（简化为线性过渡）
border rollContactTop(t=0,1) { 
    x = -contactLength/2 + t*contactLength; 
    y = h0/2 - t*(h0-h1)/2; 
    label = 7; 
}

border rollContactBottom(t=0,1) { 
    x = -contactLength/2 + t*contactLength; 
    y = -h0/2 + t*(h0-h1)/2; 
    label = 8; 
}

// ============================================
// 3. 网格生成
// ============================================

// 计算各段网格数
int nInlet = max(10, int(entryLength/meshSize));
int nOutlet = max(10, int(exitLength/meshSize));
int nContact = max(20, int(contactLength/meshSize/2));
int nThickness = max(10, int(h1/meshSize));

// 生成网格
mesh Th = buildmesh(
    inlet(nThickness) + 
    outlet(nThickness) + 
    topEntry(nInlet) + 
    bottomEntry(nInlet) +
    topExit(nOutlet) + 
    bottomExit(nOutlet) +
    rollContactTop(nContact) + 
    rollContactBottom(nContact)
);

// 输出网格信息
cout << "网格生成完成:" << endl;
cout << "  节点数: " << Th.nv << endl;
cout << "  单元数: " << Th.nt << endl;
cout << "  网格质量: " << Th.measure << endl;

// ============================================
// 4. 有限元空间定义
// ============================================

// 温度场 - P1元
fespace Vh(Th, P1);

// 位移场 - P2矢量元
fespace Wh(Th, [P2, P2]);

// 应力场 - P1张量元（用于后处理）
fespace Sh(Th, [P1, P1, P1]);

// 声明场变量
Vh T, Ttest, Told;                        // 温度场
Wh [u, v], [uu, vv];                     // 位移场
Vh vonMises, pressure, Tresca;           // 应力不变量

// 初始化温度场
T = T0;
Told = T0;

// ============================================
// 5. 材料模型（温度相关）
// ============================================

// 弹性模量温度修正
func real E(real temp) {
    real Tref = 293.0;  // 参考温度
    real factor = 1.0 - 0.0004 * (temp - Tref);
    return E0 * max(0.1, factor);  // 防止负值
}

// 热导率温度修正
func real k(real temp) {
    real Tref = 293.0;
    real factor = 1.0 - 0.0002 * (temp - Tref);
    return k0 * max(0.1, factor);
}

// 屈服强度温度和应变率修正
func real sigmaY(real temp, real epsRate) {
    real Tref = 293.0;
    real Tmelt = 1800.0;  // 熔点
    real tempFactor = (Tmelt - temp) / (Tmelt - Tref);
    real strainFactor = pow(max(epsRate/1.0, 0.001), 0.1);  // 应变率敏感指数
    return sigma_y0 * tempFactor * strainFactor;
}

// Lame常数
func real lambda(real temp) {
    real Et = E(temp);
    return Et * nu / ((1.0 + nu) * (1.0 - 2.0*nu));
}

func real mu(real temp) {
    real Et = E(temp);
    return Et / (2.0 * (1.0 + nu));
}

// ============================================
// 6. 热传导问题
// ============================================

// 热源项（塑性功和摩擦热）
Vh Qplastic = 0.0;  // 塑性变形热
Vh Qfriction = 0.0; // 摩擦热

// 弱形式
problem thermal(T, Ttest) = 
    // 瞬态项
    int2d(Th)(
        rho * cp * T * Ttest / dt
    )
    // 扩散项
    + int2d(Th)(
        k(Told) * (dx(T)*dx(Ttest) + dy(T)*dy(Ttest))
    )
    // 对流换热（自由表面）
    + int1d(Th, 3, 4, 5, 6)(
        heatTransfer * (T - Tambient) * Ttest
    )
    // 时间项
    - int2d(Th)(
        rho * cp * Told * Ttest / dt
    )
    // 内热源
    - int2d(Th)(
        (Qplastic + Qfriction) * Ttest
    )
    // 边界条件
    + on(1, T=T0)          // 入口温度
    + on(7, 8, T=Troll);   // 轧辊接触温度

// ============================================
// 7. 力学问题
// ============================================

// 应变和应力宏定义
macro epsilon(u1,u2) [dx(u1), dy(u2), (dy(u1)+dx(u2))/sqrt(2.)] //
macro div(u1,u2) (dx(u1) + dy(u2)) //

// 热应变
func real thermalStrain(real temp) {
    return alpha * (temp - T0);
}

// 弱形式
problem mechanical([u,v], [uu,vv]) = 
    // 弹性应力项
    int2d(Th)(
        lambda(T) * div(u,v) * div(uu,vv) + 
        2.0 * mu(T) * (epsilon(u,v)' * epsilon(uu,vv))
    )
    // 热应力项
    - int2d(Th)(
        (3.0*lambda(T) + 2.0*mu(T)) * thermalStrain(T) * div(uu,vv)
    )
    // 接触力（上轧辊）
    - int1d(Th, 7)(
        contactPressure * vv
    )
    // 接触力（下轧辊）
    - int1d(Th, 8)(
        -contactPressure * vv
    )
    // 摩擦力
    - int1d(Th, 7, 8)(
        frictionCoeff * contactPressure * uu * sign(rollSpeed - dx(u))
    )
    // 边界条件
    + on(1, u=0, v=0);  // 入口固定

// ============================================
// 8. 应力计算
// ============================================

// 应力分量计算函数
func real computeStress11() {
    return lambda(T)*div(u,v) + 2.0*mu(T)*dx(u) - (3.0*lambda(T) + 2.0*mu(T))*thermalStrain(T);
}

func real computeStress22() {
    return lambda(T)*div(u,v) + 2.0*mu(T)*dy(v) - (3.0*lambda(T) + 2.0*mu(T))*thermalStrain(T);
}

func real computeStress12() {
    return mu(T)*(dy(u) + dx(v));
}

// ============================================
// 9. 输出文件准备
// ============================================

// 创建输出目录
exec("mkdir -p results");

// 输出文件
ofstream tempHistory("results/temperature_history_${task_id}.dat");
ofstream stressHistory("results/stress_history_${task_id}.dat");
ofstream forceHistory("results/force_history_${task_id}.dat");
ofstream energyHistory("results/energy_history_${task_id}.dat");

// VTK输出准备
load "iovtk"

// 写入文件头
tempHistory << "# Time[s] MaxTemp[K] MinTemp[K] AvgTemp[K]" << endl;
stressHistory << "# Time[s] MaxVonMises[Pa] MaxTresca[Pa] MaxPressure[Pa]" << endl;
forceHistory << "# Time[s] RollingForce[N] FrictionForce[N]" << endl;
energyHistory << "# Time[s] ElasticEnergy[J] ThermalEnergy[J] PlasticWork[J]" << endl;

// ============================================
// 10. 时间步进求解
// ============================================

// 收敛控制
real residual = 1.0;
int iter = 0;

// 主循环
for(int step = 0; step < nsteps; step++) {
    real currentTime = step * dt;
    
    // 显示进度
    if(step % 10 == 0) {
        cout << endl;
        cout << "===== 时间步 " << step << "/" << nsteps << " =====" << endl;
        cout << "当前时间: " << currentTime << " s" << endl;
    }
    
    // 求解耦合系统（固定点迭代）
    for(iter = 0; iter < 10; iter++) {
        // 保存上一次迭代结果
        Vh Titer = T;
        
        // 求解温度场
        thermal;
        
        // 求解位移场
        mechanical;
        
        // 计算残差
        residual = sqrt(int2d(Th)((T - Titer)^2)) / sqrt(int2d(Th)(T^2));
        
        if(residual < tolerance) {
            break;
        }
    }
    
    if(step % 10 == 0) {
        cout << "  收敛迭代次数: " << iter+1 << endl;
        cout << "  残差: " << residual << endl;
    }
    
    // ============================================
    // 11. 后处理计算
    // ============================================
    
    // 计算应力分量
    Vh s11 = computeStress11();
    Vh s22 = computeStress22();
    Vh s12 = computeStress12();
    
    // von Mises应力
    vonMises = sqrt(s11^2 + s22^2 - s11*s22 + 3.0*s12^2);
    
    // 静水压力
    pressure = -(s11 + s22)/2.0;
    
    // Tresca应力
    Tresca = sqrt((s11-s22)^2 + 4.0*s12^2);
    
    // 计算塑性功（简化）
    real plasticWork = 0.0;
    if(vonMises[].max > sigmaY(T[].max, strainRate)) {
        plasticWork = int2d(Th)(
            0.9 * vonMises * strainRate * dt
        );
        Qplastic = 0.9 * vonMises * strainRate;  // 90%转化为热
    }
    
    // 计算摩擦热
    Qfriction = 0.0;
    varf vfriction(unused, w) = int1d(Th, 7, 8)(
        frictionCoeff * contactPressure * abs(rollSpeed) * w
    );
    real frictionHeat = vfriction(0, 1);
    
    // 在接触边界上分布摩擦热
    solve frictionDistribution(Qfriction, Ttest) = 
        int2d(Th)(Qfriction * Ttest)
        - int1d(Th, 7, 8)(
            frictionCoeff * contactPressure * abs(rollSpeed) * Ttest / (h1 * 0.1)  // 假设10%深度受影响
        );
    
    // ============================================
    // 12. 轧制力计算
    // ============================================
    
    // 法向力
    varf vnormalForce(unused, w) = int1d(Th, 7, 8)(
        contactPressure * w
    );
    real normalForce = vnormalForce(0, 1) * stripWidth;
    
    // 摩擦力
    varf vfrictionForce(unused, w) = int1d(Th, 7, 8)(
        frictionCoeff * contactPressure * w
    );
    real frictionForce = vfrictionForce(0, 1) * stripWidth;
    
    // 总轧制力
    real rollingForce = sqrt(normalForce^2 + frictionForce^2);
    
    // ============================================
    // 13. 能量计算
    // ============================================
    
    // 弹性应变能
    real elasticEnergy = int2d(Th)(
        0.5 * (s11*epsilon(u,v)[0] + s22*epsilon(u,v)[1] + 2.0*s12*epsilon(u,v)[2])
    ) * stripWidth;
    
    // 热能
    real thermalEnergy = int2d(Th)(
        rho * cp * (T - T0)
    ) * stripWidth;
    
    // ============================================
    // 14. 数据记录
    // ============================================
    
    // 温度统计
    real Tmax = T[].max;
    real Tmin = T[].min;
    real Tavg = int2d(Th)(T) / int2d(Th)(1.0);
    tempHistory << currentTime << " " << Tmax << " " << Tmin << " " << Tavg << endl;
    
    // 应力统计
    real vmMax = vonMises[].max;
    real trMax = Tresca[].max;
    real prMax = abs(pressure[].min);
    stressHistory << currentTime << " " << vmMax << " " << trMax << " " << prMax << endl;
    
    // 力统计
    forceHistory << currentTime << " " << rollingForce << " " << frictionForce << endl;
    
    // 能量统计
    energyHistory << currentTime << " " << elasticEnergy << " " << thermalEnergy << " " << plasticWork << endl;
    
    // ============================================
    // 15. VTK输出
    // ============================================
    
    if(step % outputInterval == 0) {
        string vtkfile = "results/rolling_${task_id}_" + step + ".vtk";
        
        // 计算等效应变
        Vh equivStrain = vonMises / E(T);
        
        // 计算温升
        Vh tempRise = T - T0;
        
        // 输出VTK文件
        savevtk(vtkfile, Th, 
            [u, v, 0],            // 位移
            T,                    // 温度
            vonMises,             // von Mises应力
            pressure,             // 压力
            Tresca,               // Tresca应力
            equivStrain,          // 等效应变
            tempRise,             // 温升
            dataname="Displacement Temperature VonMises Pressure Tresca EquivStrain TempRise",
            order=[1, 0, 0, 0, 0, 0, 0]  // 位移用P1，其他用P0
        );
        
        if(step % 10 == 0) {
            cout << "  VTK文件已保存: " << vtkfile << endl;
        }
    }
    
    // ============================================
    // 16. 更新时间步
    // ============================================
    
    Told = T;
    
    // 显示关键结果
    if(step % 10 == 0) {
        cout << "  最高温度: " << Tmax-273.15 << " °C" << endl;
        cout << "  最大von Mises应力: " << vmMax/1e6 << " MPa" << endl;
        cout << "  轧制力: " << rollingForce/1e6 << " MN" << endl;
    }
}

// ============================================
// 17. 最终结果输出
// ============================================

cout << endl;
cout << "========== 模拟完成 ==========" << endl;
cout << "最终结果:" << endl;
cout << "  最高温度: " << T[].max-273.15 << " °C" << endl;
cout << "  最大von Mises应力: " << vonMises[].max/1e6 << " MPa" << endl;
cout << "  最大位移: " << sqrt(u[]'*u[] + v[]'*v[]) * 1000 << " mm" << endl;

// 关闭输出文件
tempHistory.flush;
stressHistory.flush;
forceHistory.flush;
energyHistory.flush;

// 输出JSON格式的最终结果
{
    ofstream ff("results/final_results_${task_id}.json");
    ff << "{" << endl;
    ff << "  \"task_id\": \"${task_id}\"," << endl;
    ff << "  \"timestamp\": \"${timestamp}\"," << endl;
    ff << "  \"parameters\": {" << endl;
    ff << "    \"roll_radius\": " << rollRadius << "," << endl;
    ff << "    \"thickness_initial\": " << h0 << "," << endl;
    ff << "    \"thickness_final\": " << h1 << "," << endl;
    ff << "    \"reduction_ratio\": " << reduction << "," << endl;
    ff << "    \"temperature_initial\": " << T0 << endl;
    ff << "  }," << endl;
    ff << "  \"mesh_info\": {" << endl;
    ff << "    \"vertices\": " << Th.nv << "," << endl;
    ff << "    \"elements\": " << Th.nt << "," << endl;
    ff << "    \"quality\": " << Th.measure << endl;
    ff << "  }," << endl;
    ff << "  \"results\": {" << endl;
    ff << "    \"max_temperature\": " << T[].max << "," << endl;
    ff << "    \"min_temperature\": " << T[].min << "," << endl;
    ff << "    \"max_stress\": " << vonMises[].max << "," << endl;
    ff << "    \"max_displacement\": " << sqrt(u[]'*u[] + v[]'*v[]) << "," << endl;
    ff << "    \"final_rolling_force\": " << rollingForce << endl;
    ff << "  }" << endl;
    ff << "}" << endl;
}

// ============================================
// 18. 导出数据用于Python后处理
// ============================================

// 保存网格
savemesh(Th, "results/mesh_${task_id}.msh");

// 保存场数据
{
    ofstream tempData("results/temperature_field_${task_id}.dat");
    for(int i = 0; i < Th.nv; i++) {
        tempData << T[][i] << endl;
    }
}

{
    ofstream stressData("results/stress_field_${task_id}.dat");
    for(int i = 0; i < Th.nv; i++) {
        stressData << vonMises[][i] << endl;
    }
}

{
    ofstream dispData("results/displacement_field_${task_id}.dat");
    for(int i = 0; i < Th.nv; i++) {
        dispData << u[][i] << " " << v[][i] << endl;
    }
}

// 保存节点坐标
{
    ofstream coordData("results/coordinates_${task_id}.dat");
    for(int i = 0; i < Th.nv; i++) {
        coordData << Th(i).x << " " << Th(i).y << endl;
    }
}

// 保存单元连接
{
    ofstream elemData("results/elements_${task_id}.dat");
    for(int i = 0; i < Th.nt; i++) {
        for(int j = 0; j < 3; j++) {
            elemData << Th[i][j] << " ";
        }
        elemData << endl;
    }
}

cout << "所有数据已导出完成!" << endl;
cout << "================================" << endl;

// 脚本结束
'''
        return template_content
    
    def generate_script(
        self,
        parameters: Dict[str, Any],
        task_id: str,
        output_filename: Optional[str] = None
    ) -> str:
        """
        生成FreeFEM脚本
        
        Args:
            parameters: 参数字典
            task_id: 任务ID
            output_filename: 输出文件名（可选）
            
        Returns:
            生成的脚本文件路径
            
        Raises:
            ValueError: 参数验证失败
            IOError: 文件写入失败
        """
        try:
            # 创建参数对象并验证
            rolling_params = RollingParameter(**parameters.get('rolling_params', {}))
            material_props = MaterialProperty(**parameters.get('material_props', {}))
            mesh_params = MeshParameter(**parameters.get('mesh_params', {}))
            solver_params = SolverParameter(**parameters.get('solver_params', {}))
            
            # 验证所有参数
            rolling_params.validate()
            material_props.validate()
            mesh_params.validate()
            solver_params.validate()
            
            # 准备模板参数
            template_params = {
                # 元数据
                'task_id': task_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                
                # 轧制参数
                **rolling_params.__dict__,
                'contact_length': rolling_params.contact_length,
                
                # 材料参数
                **material_props.__dict__,
                
                # 网格参数
                **mesh_params.__dict__,
                
                # 求解器参数
                **solver_params.__dict__,
                'num_steps': solver_params.num_steps,
            }
            
            # 获取模板
            template_str = self._get_or_create_template('rolling')
            template = Template(template_str)
            
            # 生成脚本内容
            script_content = template.safe_substitute(**template_params)
            
            # 确定输出文件名
            if output_filename is None:
                output_filename = f"rolling_{task_id}.edp"
            
            output_path = self.output_dir / output_filename
            
            # 写入文件（使用GB2312编码）
            with open(output_path, 'w', encoding='gb2312') as f:
                f.write(script_content)
            
            logger.info(f"FreeFEM脚本已生成: {output_path}")
            
            # 保存参数记录
            self._save_parameter_record(task_id, template_params)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"生成FreeFEM脚本失败: {str(e)}")
            raise
    
    def _get_or_create_template(self, template_name: str) -> str:
        """获取或创建模板"""
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        # 尝试从文件加载
        template_file = self.template_dir / f"{template_name}_template.edp"
        if template_file.exists():
            try:
                with open(template_file, 'r', encoding='gb2312') as f:
                    template_content = f.read()
                self._template_cache[template_name] = template_content
                return template_content
            except Exception as e:
                logger.warning(f"加载模板文件失败: {e}")
        
        # 使用内置模板
        if template_name == 'rolling':
            template_content = self.create_rolling_template()
            self._template_cache[template_name] = template_content
            
            # 保存到文件
            try:
                with open(template_file, 'w', encoding='gb2312') as f:
                    f.write(template_content)
            except Exception as e:
                logger.warning(f"保存模板文件失败: {e}")
            
            return template_content
        
        raise ValueError(f"未知的模板名称: {template_name}")
    
    def _save_parameter_record(self, task_id: str, parameters: Dict[str, Any]):
        """保存参数记录"""
        record_file = self.output_dir / f"parameters_{task_id}.json"
        try:
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(parameters, f, indent=2, ensure_ascii=False)
            logger.debug(f"参数记录已保存: {record_file}")
        except Exception as e:
            logger.warning(f"保存参数记录失败: {e}")
    
    def validate_script(self, script_path: str) -> bool:
        """验证生成的脚本语法"""
        try:
            # 基本语法检查
            with open(script_path, 'r', encoding='gb2312') as f:
                content = f.read()
            
            # 检查必要的元素
            required_elements = [
                'mesh Th',
                'fespace',
                'problem',
                'solve'
            ]
            
            for element in required_elements:
                if element not in content:
                    logger.error(f"脚本缺少必要元素: {element}")
                    return False
            
            # 检查括号匹配
            if content.count('(') != content.count(')'):
                logger.error("括号不匹配")
                return False
            
            if content.count('{') != content.count('}'):
                logger.error("大括号不匹配")
                return False
            
            logger.info("脚本语法验证通过")
            return True
            
        except Exception as e:
            logger.error(f"脚本验证失败: {e}")
            return False
    
    def create_batch_scripts(
        self,
        parameter_sets: List[Dict[str, Any]],
        base_task_id: str
    ) -> List[str]:
        """批量生成脚本"""
        script_paths = []
        
        for i, params in enumerate(parameter_sets):
            task_id = f"{base_task_id}_{i:03d}"
            try:
                script_path = self.generate_script(params, task_id)
                script_paths.append(script_path)
            except Exception as e:
                logger.error(f"批量生成第{i}个脚本失败: {e}")
                continue
        
        logger.info(f"批量生成完成: {len(script_paths)}/{len(parameter_sets)} 个脚本")
        return script_paths
    
    def generate_parametric_study_scripts(
        self,
        base_params: Dict[str, Any],
        vary_param: str,
        values: List[float],
        task_id_prefix: str
    ) -> List[str]:
        """生成参数化研究脚本"""
        script_paths = []
        
        for i, value in enumerate(values):
            # 复制基础参数
            params = json.loads(json.dumps(base_params))  # 深拷贝
            
            # 修改变化参数
            if '.' in vary_param:
                # 处理嵌套参数
                keys = vary_param.split('.')
                current = params
                for key in keys[:-1]:
                    current = current[key]
                current[keys[-1]] = value
            else:
                params[vary_param] = value
            
            # 生成脚本
            task_id = f"{task_id_prefix}_{vary_param}_{i:03d}"
            try:
                script_path = self.generate_script(params, task_id)
                script_paths.append(script_path)
                logger.info(f"参数化研究脚本生成: {vary_param}={value}")
            except Exception as e:
                logger.error(f"参数化研究脚本生成失败 ({vary_param}={value}): {e}")
        
        return script_paths


def test_template_engine():
    """测试模板引擎"""
    engine = FreeFEMTemplateEngine()
    
    # 测试参数
    test_params = {
        'rolling_params': {
            'roll_radius': 0.5,
            'thickness_initial': 0.025,
            'thickness_final': 0.020,
            'strip_width': 2.0,
            'roll_speed': 3.8,
            'temperature_initial': 1123,
            'temperature_roll': 423,
            'friction_coefficient': 0.3,
            'contact_pressure': 150e6,
        },
        'material_props': {
            'density': 7850,
            'youngs_modulus': 210e9,
            'poisson_ratio': 0.3,
            'thermal_expansion': 1.2e-5,
            'thermal_conductivity': 45,
            'specific_heat': 460,
        },
        'mesh_params': {
            'element_size': 0.001,
            'max_elements': 50000,
        },
        'solver_params': {
            'time_step': 0.001,
            'total_time': 0.1,
            'tolerance': 1e-6,
            'output_interval': 5,
        }
    }
    
    # 生成脚本
    script_path = engine.generate_script(test_params, 'test_001')
    print(f"测试脚本已生成: {script_path}")
    
    # 验证脚本
    is_valid = engine.validate_script(script_path)
    print(f"脚本验证结果: {'通过' if is_valid else '失败'}")
    
    # 参数化研究测试
    vary_values = [0.4, 0.5, 0.6, 0.7]
    param_scripts = engine.generate_parametric_study_scripts(
        test_params,
        'rolling_params.roll_radius',
        vary_values,
        'param_study'
    )
    print(f"参数化研究脚本生成: {len(param_scripts)} 个")


if __name__ == "__main__":
    test_template_engine()
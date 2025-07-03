"""
JMatPro自动化项目集成示例
展示如何将报告生成模块集成到实际的JMatPro自动化流程中

项目结构建议：
your_project/
├── simulation_report.py      # 报告生成模块
├── jmatpro_automation.py     # JMatPro自动化主程序
├── config.py                 # 配置文件
├── output/                   # 输出目录
│   ├── reports/             # 报告文件
│   └── data/               # 原始数据
└── templates/               # 报告模板（可选）
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

# 导入报告生成模块
from simulation_report import SimulationReportGenerator

# 模拟的JMatPro API（实际项目中替换为真实API）
class JMatProAPI:
    """模拟的JMatPro API接口"""
    
    def __init__(self):
        self.material_type = None
        self.composition = {}
        self.temperature = 20
    
    def set_material_type(self, material_type: str):
        self.material_type = material_type
    
    def set_composition(self, composition: Dict[str, float]):
        self.composition = composition
    
    def calculate_phase_equilibrium(self, temp_range: tuple) -> Dict:
        """模拟相平衡计算"""
        temps = np.linspace(temp_range[0], temp_range[1], 50)
        # 模拟数据
        ferrite = np.exp(-temps/600) * 0.9
        austenite = 1 - ferrite - 0.05
        cementite = np.ones_like(temps) * 0.05
        
        return {
            'temperatures': temps.tolist(),
            'phases': {
                'Ferrite': ferrite.tolist(),
                'Austenite': austenite.tolist(),
                'Cementite': cementite.tolist()
            }
        }
    
    def calculate_ttt(self, grain_size: float) -> Dict:
        """模拟TTT计算"""
        return {
            'Pearlite': {
                'start_times': [1, 5, 20, 100, 500],
                'start_temperatures': [700, 650, 600, 550, 500],
                'finish_times': [10, 50, 200, 1000, 5000],
                'finish_temperatures': [700, 650, 600, 550, 500]
            },
            'Bainite': {
                'start_times': [0.5, 2, 10, 50],
                'start_temperatures': [450, 400, 350, 300],
                'finish_times': [5, 20, 100, 500],
                'finish_temperatures': [450, 400, 350, 300]
            }
        }
    
    def calculate_cct(self, cooling_rates: List[float]) -> Dict:
        """模拟CCT计算"""
        cct_data = {}
        for rate in cooling_rates:
            time = np.logspace(-1, 3, 50)
            temp = 900 - rate * time
            cct_data[rate] = {
                'time': time.tolist(),
                'temperature': temp.tolist()
            }
        return cct_data
    
    def calculate_mechanical_properties(self, temp_range: tuple = (20, 600)) -> Dict:
        """模拟机械性能计算"""
        temps = np.linspace(temp_range[0], temp_range[1], 7)
        
        # 模拟的性能数据
        yield_strength = 355 * np.exp(-temps/800)
        tensile_strength = 600 * np.exp(-temps/900)
        hardness = 180 * np.exp(-temps/1000)
        
        return {
            'temperature': temps.tolist(),
            'yield_strength': yield_strength.tolist(),
            'tensile_strength': tensile_strength.tolist(),
            'hardness': hardness.tolist()
        }
    
    def simulate_solidification(self, cooling_rate: float) -> Dict:
        """模拟凝固过程"""
        temps = np.linspace(1500, 1300, 50)
        solid_fraction = 1 / (1 + np.exp(-0.1 * (1400 - temps)))
        
        return {
            'temperature': temps.tolist(),
            'solid_fraction': solid_fraction.tolist(),
            'cooling_rate': cooling_rate,
            'liquidus_temp': 1480,
            'solidus_temp': 1420
        }


class JMatProAutomation:
    """JMatPro自动化主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.jmatpro = JMatProAPI()  # 实际项目中使用真实API
        self.report_generator = SimulationReportGenerator()
        self.results = {}
        
    def setup_material(self, material_name: str, composition: Dict[str, float]):
        """设置材料参数"""
        self.jmatpro.set_material_type(self.config['material_type'])
        self.jmatpro.set_composition(composition)
        
        # 同步到报告生成器
        self.report_generator.set_material_info(
            name=material_name,
            type=self.config['material_type'],
            standard=self.config.get('standard', ''),
            description=self.config.get('description', '')
        )
        self.report_generator.add_composition(composition)
        
    def run_comprehensive_analysis(self):
        """运行综合分析"""
        print("开始综合材料性能分析...")
        
        # 1. 相平衡计算
        print("1. 计算相平衡...")
        phase_results = self.jmatpro.calculate_phase_equilibrium(
            self.config['temperature_range']
        )
        self.results['phase_equilibrium'] = phase_results
        self.report_generator.add_phase_diagram_results(
            phase_results['temperatures'],
            phase_results['phases']
        )
        
        # 2. TTT计算
        print("2. 计算TTT曲线...")
        ttt_results = self.jmatpro.calculate_ttt(
            self.config['grain_size']
        )
        self.results['ttt'] = ttt_results
        self.report_generator.add_ttt_results(ttt_results)
        
        # 3. CCT计算
        print("3. 计算CCT曲线...")
        cct_results = self.jmatpro.calculate_cct(
            self.config['cooling_rates']
        )
        self.results['cct'] = cct_results
        self.report_generator.add_cct_results(cct_results)
        
        # 4. 机械性能
        print("4. 计算机械性能...")
        mech_results = self.jmatpro.calculate_mechanical_properties()
        self.results['mechanical'] = mech_results
        self.report_generator.add_mechanical_properties(mech_results)
        
        # 5. 凝固模拟
        print("5. 模拟凝固过程...")
        solidif_results = self.jmatpro.simulate_solidification(
            self.config['solidification_cooling_rate']
        )
        self.results['solidification'] = solidif_results
        self.report_generator.add_solidification_results(solidif_results)
        
        print("分析完成！")
        
    def add_heat_treatment_recommendations(self):
        """添加热处理建议"""
        # 基于计算结果生成热处理建议
        ht_data = pd.DataFrame({
            '热处理工艺': ['正火', '淬火+回火', '退火', '调质'],
            '加热温度 (°C)': [870, 840, 720, 840],
            '保温时间 (min)': [60, 45, 180, 45],
            '冷却方式': ['空冷', '水冷/油冷', '炉冷', '油冷'],
            '回火温度 (°C)': ['-', 550, '-', 600],
            '预期硬度 (HB)': [180, 280, 150, 240],
            '适用场合': ['一般机械零件', '高强度要求', '改善切削性', '综合性能要求']
        })
        
        self.report_generator.add_custom_table(
            ht_data,
            'heat_treatment',
            '推荐热处理工艺参数'
        )
        
    def analyze_results(self):
        """分析结果并生成结论"""
        conclusions = []
        
        # 分析相变特征
        if 'phase_equilibrium' in self.results:
            conclusions.append(
                f"材料在{self.config['temperature_range'][0]}-{self.config['temperature_range'][1]}°C"
                f"范围内存在铁素体、奥氏体和渗碳体三相"
            )
        
        # 分析转变特征
        if 'ttt' in self.results:
            conclusions.append(
                "TTT曲线显示，在550-700°C范围内主要发生珠光体转变，"
                "300-450°C范围内发生贝氏体转变"
            )
        
        # 分析冷却速率影响
        if 'cct' in self.results:
            cooling_rates = self.config['cooling_rates']
            conclusions.append(
                f"CCT曲线表明，冷却速率从{min(cooling_rates)}增加到{max(cooling_rates)}°C/s时，"
                "组织从珠光体+铁素体逐渐转变为马氏体"
            )
        
        # 分析机械性能
        if 'mechanical' in self.results:
            mech = self.results['mechanical']
            room_temp_idx = 0
            conclusions.append(
                f"室温下材料的屈服强度约为{mech['yield_strength'][room_temp_idx]:.0f}MPa，"
                f"抗拉强度约为{mech['tensile_strength'][room_temp_idx]:.0f}MPa"
            )
        
        # 添加结论到报告
        for conclusion in conclusions:
            self.report_generator.add_conclusion(conclusion)
        
    def generate_reports(self, output_dir: str):
        """生成报告"""
        # 设置报告基本信息
        self.report_generator.set_basic_info(
            title=self.config['report_title'],
            subtitle=self.config.get('report_subtitle', ''),
            author=self.config['author'],
            company=self.config['company']
        )
        
        # 设置仿真参数信息
        self.report_generator.set_simulation_params(
            software="JMatPro",
            version=self.config.get('jmatpro_version', 'v12.0'),
            modules=['相平衡', 'TTT/CCT', '凝固', '机械性能'],
            temperature_range=self.config['temperature_range'],
            cooling_rates=self.config['cooling_rates'],
            grain_size=self.config['grain_size'],
            other_params={
                '奥氏体化温度': f"{self.config.get('austenitization_temp', 900)} °C",
                '保温时间': f"{self.config.get('holding_time', 30)} min"
            }
        )
        
        # 生成报告文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        word_path = os.path.join(output_dir, f"仿真报告_{timestamp}.docx")
        excel_path = os.path.join(output_dir, f"仿真数据_{timestamp}.xlsx")
        
        self.report_generator.generate_word_report(word_path)
        self.report_generator.generate_excel_report(excel_path)
        
        # 保存原始数据
        self._save_raw_data(output_dir, timestamp)
        
        # 清理临时文件
        self.report_generator.cleanup()
        
        return word_path, excel_path
    
    def _save_raw_
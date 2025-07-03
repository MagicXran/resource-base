# JMatPro 仿真报告生成模块使用指南

## 目录结构

```
your_project/
├── simulation_report.py      # 核心报告生成模块
├── jmatpro_automation.py     # JMatPro自动化集成示例
├── config.json              # 配置文件
├── requirements.txt         # 依赖包列表
├── output/                  # 输出目录
│   ├── reports/            # 生成的报告文件
│   └── data/              # 原始数据文件
└── examples/               # 示例脚本
```

## 安装依赖

```bash
pip install -r requirements.txt
```

### requirements.txt 内容：
```
python-docx>=0.8.11
openpyxl>=3.0.9
matplotlib>=3.5.0
pandas>=1.3.0
numpy>=1.21.0
Pillow>=8.4.0
```

## 快速开始

### 1. 基础使用示例

```python
from simulation_report import SimulationReportGenerator

# 创建报告生成器
report = SimulationReportGenerator()

# 设置基本信息
report.set_basic_info(
    title="316L不锈钢材料性能分析报告",
    author="材料研究部",
    company="XX科技有限公司"
)

# 添加材料成分
composition = {
    'C': 0.03,
    'Cr': 17.0,
    'Ni': 12.0,
    'Mo': 2.5,
    'Mn': 2.0,
    'Si': 0.75
}
report.add_composition(composition)

# 生成报告
report.generate_word_report("output/316L_report.docx")
report.generate_excel_report("output/316L_data.xlsx")
```

### 2. 配置文件示例 (config.json)

```json
{
    "project_name": "钢材性能分析项目",
    "material_configs": {
        "45_steel": {
            "material_name": "45号钢",
            "material_type": "General Steel",
            "standard": "GB/T 699-2015",
            "description": "中碳结构钢，广泛用于机械制造",
            "composition": {
                "C": 0.45,
                "Si": 0.25,
                "Mn": 0.65,
                "Cr": 0.25,
                "Ni": 0.30,
                "P": 0.035,
                "S": 0.035
            },
            "report_title": "45号钢综合性能分析报告",
            "report_subtitle": "基于JMatPro的材料仿真研究"
        },
        "316L_steel": {
            "material_name": "316L不锈钢",
            "material_type": "Stainless Steel",
            "standard": "ASTM A240/A240M",
            "description": "低碳奥氏体不锈钢，耐腐蚀性优异",
            "composition": {
                "C": 0.03,
                "Cr": 17.0,
                "Ni": 12.0,
                "Mo": 2.5,
                "Mn": 2.0,
                "Si": 0.75,
                "P": 0.045,
                "S": 0.030
            },
            "report_title": "316L不锈钢材料性能分析报告",
            "report_subtitle": "耐腐蚀性能与机械性能综合评估"
        }
    },
    "simulation_parameters": {
        "temperature_range": [200, 1200],
        "cooling_rates": [0.1, 1, 10, 100],
        "grain_size": 50,
        "austenitization_temp": 900,
        "holding_time": 30,
        "solidification_cooling_rate": 10
    },
    "report_settings": {
        "author": "张三",
        "company": "XX材料科技有限公司",
        "jmatpro_version": "v12.0",
        "include_raw_data": true,
        "generate_charts": true,
        "chart_dpi": 300
    }
}
```

### 3. 完整集成示例

```python
import json
from simulation_report import SimulationReportGenerator
from jmatpro_automation import JMatProAutomation

# 加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 选择材料
material_config = config['material_configs']['45_steel']
sim_params = config['simulation_parameters']
report_settings = config['report_settings']

# 合并配置
full_config = {
    **material_config,
    **sim_params,
    **report_settings
}

# 创建自动化实例
automation = JMatProAutomation(full_config)

# 设置材料
automation.setup_material(
    material_config['material_name'],
    material_config['composition']
)

# 运行分析
automation.run_comprehensive_analysis()

# 添加热处理建议
automation.add_heat_treatment_recommendations()

# 分析结果
automation.analyze_results()

# 生成报告
word_path, excel_path = automation.generate_reports('output/reports')

print(f"报告已生成：")
print(f"Word报告: {word_path}")
print(f"Excel数据: {excel_path}")
```

## 仿真报告内容说明

### 典型的材料仿真报告包含以下内容：

#### 1. **封面信息**
- 报告标题和副标题
- 材料名称
- 公司信息
- 作者和日期
- 报告编号

#### 2. **摘要**
- 仿真目的
- 使用的软件和模块
- 材料成分概览
- 关键发现总结

#### 3. **材料信息**
- 详细化学成分表
- 材料类型和标准
- 材料特性描述

#### 4. **仿真参数**
- 计算模块列表
- 温度范围
- 冷却速率
- 晶粒尺寸
- 其他关键参数

#### 5. **计算结果**
- **相平衡图**：不同温度下的相组成
- **TTT曲线**：等温转变动力学
- **CCT曲线**：连续冷却转变
- **凝固曲线**：液固相变过程
- **机械性能**：强度、硬度随温度变化

#### 6. **数据分析**
- 相变特征分析
- 组织演变规律
- 性能变化趋势
- 工艺窗口确定

#### 7. **工艺建议**
- 热处理工艺参数
- 冷却方式选择
- 预期组织和性能

#### 8. **结论与建议**
- 主要发现总结
- 应用建议
- 进一步研究方向

## 高级功能

### 1. 批量处理多种材料

```python
# 批量分析示例
materials = ['45_steel', '316L_steel', '20CrMnTi']

for material_key in materials:
    if material_key in config['material_configs']:
        material_config = config['material_configs'][material_key]
        
        # 创建独立的报告生成器
        report = SimulationReportGenerator()
        
        # 运行分析流程...
        # 生成报告...
```

### 2. 自定义图表样式

```python
# 自定义matplotlib样式
import matplotlib.pyplot as plt

# 创建自定义图表
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(data['x'], data['y'], 'b-', linewidth=3)
ax.set_xlabel('温度 (°C)', fontsize=14)
ax.set_ylabel('性能指标', fontsize=14)
ax.set_title('自定义性能曲线', fontsize=16, fontweight='bold')

# 添加到报告
report.add_custom_figure(
    figure=fig,
    name='custom_performance',
    caption='自定义性能分析图',
    description='展示特定条件下的性能变化'
)
```

### 3. 添加自定义数据表

```python
import pandas as pd

# 创建对比数据表
comparison_df = pd.DataFrame({
    '材料': ['45钢', '40Cr', '35CrMo'],
    '屈服强度 (MPa)': [355, 785, 835],
    '抗拉强度 (MPa)': [600, 980, 985],
    '延伸率 (%)': [16, 9, 12],
    '冲击功 (J)': [39, 47, 63]
})

report.add_custom_table(
    data=comparison_df,
    name='material_comparison',
    caption='不同材料性能对比'
)
```

### 4. 集成实际JMatPro API

```python
# 替换模拟API为实际API
import jmatpro_api  # 实际的JMatPro Python包

class RealJMatProAPI:
    def __init__(self):
        self.jmp = jmatpro_api.initialize()
        
    def calculate_phase_equilibrium(self, temp_range):
        self.jmp.SetModule("Equilibrium")
        # 实际API调用...
        return results
```

## 常见问题解决

### 1. 中文显示问题
```python
# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
```

### 2. 大量数据处理
```python
# 使用生成器处理大量数据
def process_large_dataset(data_generator):
    for batch in data_generator:
        # 处理批次数据
        results = analyze_batch(batch)
        # 逐步添加到报告
        report.add_custom_results(results)
```

### 3. 报告模板定制
```python
# 使用自定义Word模板
from docx import Document

template_doc = Document('templates/custom_template.docx')
# 基于模板创建报告...
```

## 最佳实践

1. **模块化设计**：将不同的分析功能封装成独立模块
2. **配置管理**：使用配置文件管理参数，避免硬编码
3. **错误处理**：添加适当的异常处理和日志记录
4. **版本控制**：对报告和数据进行版本管理
5. **自动化测试**：编写单元测试确保报告生成的正确性

## 扩展开发

### 添加新的分析模块
```python
class CustomAnalysisModule:
    def __init__(self, report_generator):
        self.report = report_generator
        
    def analyze_fatigue_properties(self, data):
        # 疲劳性能分析
        results = self._calculate_fatigue(data)
        
        # 生成图表
        fig = self._create_sn_curve(results)
        
        # 添加到报告
        self.report.add_custom_figure(
            fig, 'fatigue_curve', 
            'S-N疲劳曲线',
            '不同应力水平下的疲劳寿命'
        )
```

### 集成其他材料数据库
```python
# 集成Materials Project API
from pymatgen.ext.matproj import MPRester

def get_material_properties(formula):
    with MPRester("YOUR_API_KEY") as mpr:
        data = mpr.get_data(formula)
    return data
```

## 联系和支持

如有问题或需要定制开发，请联系：
- 技术支持邮箱：support@example.com
- 项目仓库：https://github.com/yourcompany/jmatpro-report

## 许可证

本模块遵循 MIT 许可证。详见 LICENSE 文件。
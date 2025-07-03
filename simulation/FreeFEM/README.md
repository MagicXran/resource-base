# FreeFEM轧辊应力场可视化系统

## 项目概述

本项目实现了基于FreeFEM的轧辊应力场模拟与可视化系统，集成了VTK数据导出和ParaviewWeb实时3D可视化功能。

## 核心特性

- **高性能计算**：使用FreeFEM++有限元求解器进行热力耦合分析
- **灵活的Python集成**：通过模板引擎动态生成FreeFEM脚本，避免了pyFreeFEM的限制
- **专业级可视化**：通过VTK格式导出和ParaviewWeb实现交互式3D可视化
- **实时进度监控**：WebSocket推送计算进度，提供良好的用户体验
- **容器化部署**：Docker Compose一键部署，适合生产环境
- **中文支持**：考虑GB2312编码要求，支持中文界面

## 技术架构

### 后端技术栈
- **FreeFEM++**：有限元求解核心
- **Python 3.8+**：系统集成语言
- **FastAPI**：高性能Web API框架
- **VTK**：科学数据可视化库
- **ParaView**：专业级可视化平台

### 前端技术栈
- **Vue.js 3**：响应式前端框架
- **WebSocket**：实时通信
- **ParaviewWeb**：3D可视化客户端

## 快速开始

### 1. 环境要求
- Windows 10/11 或 Linux (Ubuntu 20.04+)
- Python 3.8+
- FreeFEM++ 4.x
- Docker和Docker Compose（可选）

### 2. 安装步骤
```bash
# 克隆项目
git clone <repository>
cd FreeFEM

# 安装Python依赖
pip install -r requirements.txt

# 运行系统
python main_api.py
```

### 3. 访问系统
- 主界面：http://localhost/
- API文档：http://localhost:8000/docs
- ParaviewWeb：http://localhost:9000

## 文档结构

- `FreeFEM方案.md` - 原始技术方案文档
- `pyfreefem_paraviewweb_solution.md` - 初始PyFreeFEM方案
- `improved_freefem_vtk_paraviewweb_solution.md` - **改进的完整实现方案**
- `implementation_steps.md` - 详细实施步骤指南
- `rolling_simulation_pyfreefem.py` - PyFreeFEM示例代码（参考）
- `vtk_exporter.py` - VTK导出器实现（参考）

## 改进要点

### 1. FreeFEM集成优化
- 使用模板引擎替代pyFreeFEM，支持2D/3D问题
- 结构化数据交换，提高效率
- 完整的错误处理和进度监控

### 2. VTK导出增强
- 支持时间序列数据（PVD格式）
- 多物理场同时导出（温度、应力、位移）
- 网格质量分析报告

### 3. ParaviewWeb集成
- 完整的WebSocket协议实现
- 实时字段切换和时间步控制
- 自定义相机视角和渲染设置

### 4. 生产级部署
- Docker容器化支持
- Nginx反向代理配置
- 自动备份和日志轮转

## 使用示例

### API调用示例
```python
import requests

# 创建模拟任务
response = requests.post('http://localhost:8000/api/v1/simulation/create', json={
    'rolling_params': {
        'roll_radius': 0.5,
        'thickness_initial': 0.025,
        'thickness_final': 0.020,
        'roll_speed': 3.8,
        'temperature_initial': 1123
    }
})

task_id = response.json()['task_id']

# 获取结果
results = requests.get(f'http://localhost:8000/api/v1/simulation/{task_id}/results')
```

## 性能指标

- 典型2D模拟（10万网格）：2-5分钟
- VTK导出：< 10秒
- 可视化加载：< 3秒
- 并发支持：4-8个任务

## 扩展性

系统设计支持以下扩展：
- 3D轧制模拟
- 多道次轧制工艺
- 材料数据库集成
- 机器学习预测模型
- 云端部署支持

## 许可证

本项目基于MIT许可证开源。

## 联系方式

如有问题或建议，请提交Issue或联系开发团队。

---

**注意**：本系统专为Windows环境优化，支持GB2312编码。Linux部署请参考Docker配置。
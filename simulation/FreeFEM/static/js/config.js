// -*- coding: gb2312 -*-
/**
 * 配置文件 - API端点和全局设置
 */

const CONFIG = {
    // API配置
    API: {
        BASE_URL: window.location.origin,
        ENDPOINTS: {
            // 模拟相关
            CREATE_SIMULATION: '/api/v1/simulations',
            GET_SIMULATION: '/api/v1/simulations/{id}',
            LIST_SIMULATIONS: '/api/v1/simulations',
            CANCEL_SIMULATION: '/api/v1/simulations/{id}/cancel',
            DELETE_SIMULATION: '/api/v1/simulations/{id}',
            
            // 结果相关
            GET_RESULTS: '/api/v1/simulations/{id}/results',
            DOWNLOAD_VTK: '/api/v1/simulations/{id}/vtk',
            DOWNLOAD_REPORT: '/api/v1/simulations/{id}/report',
            
            // 系统相关
            HEALTH_CHECK: '/api/v1/health',
            SYSTEM_INFO: '/api/v1/system/info',
            
            // WebSocket
            WS_PROGRESS: '/ws/progress/{id}'
        },
        TIMEOUT: 30000, // 30秒
        RETRY_COUNT: 3,
        RETRY_DELAY: 1000 // 1秒
    },
    
    // ParaviewWeb配置
    PARAVIEW: {
        WS_URL: 'ws://localhost:9000/ws',
        LAUNCHER_URL: 'http://localhost:9000/paraview',
        CONNECTION_TIMEOUT: 10000,
        HEARTBEAT_INTERVAL: 5000
    },
    
    // 界面配置
    UI: {
        // 更新间隔
        REFRESH_INTERVAL: 2000, // 2秒
        NOTIFICATION_DURATION: 5000, // 5秒
        
        // 分页
        PAGE_SIZE: 10,
        MAX_PAGE_SIZE: 100,
        
        // 文件限制
        MAX_UPLOAD_SIZE: 100 * 1024 * 1024, // 100MB
        ALLOWED_FILE_TYPES: ['.edp', '.json', '.dat'],
        
        // 图表配置
        CHART_COLORS: [
            '#3498db', '#2ecc71', '#f39c12', '#e74c3c', 
            '#9b59b6', '#1abc9c', '#34495e', '#f1c40f'
        ],
        
        // 动画
        ANIMATION_DURATION: 300,
        LOADING_DELAY: 500
    },
    
    // 默认参数
    DEFAULTS: {
        // 几何参数
        ROLLING_PARAMS: {
            roll_radius: 0.5,
            initial_thickness: 0.03,
            final_thickness: 0.02,
            strip_width: 1.0,
            roll_speed: 5.0,
            friction_coefficient: 0.3,
            initial_temperature: 1273.15,
            roll_temperature: 373.15
        },
        
        // 材料属性
        MATERIAL_PROPS: {
            density: 7850,
            youngs_modulus: 210e9,
            poisson_ratio: 0.3,
            thermal_conductivity: 50,
            specific_heat: 450,
            thermal_expansion: 12e-6
        },
        
        // 求解器参数
        SOLVER_PARAMS: {
            num_steps: 100,
            output_interval: 10,
            tolerance: 1e-6,
            max_iterations: 1000,
            mesh_size: 0.005
        }
    },
    
    // 预设配置
    PRESETS: [
        {
            id: 'hot_rolling_steel',
            name: '热轧钢板',
            description: '典型的热轧钢板工艺参数',
            reduction: 33.3,
            temperature: 1100,
            parameters: {
                rolling_params: {
                    roll_radius: 0.5,
                    initial_thickness: 0.03,
                    final_thickness: 0.02,
                    strip_width: 1.0,
                    roll_speed: 5.0,
                    friction_coefficient: 0.3,
                    initial_temperature: 1373.15,
                    roll_temperature: 373.15
                }
            }
        },
        {
            id: 'cold_rolling_steel',
            name: '冷轧钢板',
            description: '典型的冷轧钢板工艺参数',
            reduction: 20,
            temperature: 25,
            parameters: {
                rolling_params: {
                    roll_radius: 0.3,
                    initial_thickness: 0.005,
                    final_thickness: 0.004,
                    strip_width: 1.2,
                    roll_speed: 10.0,
                    friction_coefficient: 0.1,
                    initial_temperature: 298.15,
                    roll_temperature: 298.15
                }
            }
        },
        {
            id: 'aluminum_rolling',
            name: '铝板轧制',
            description: '铝合金板材轧制参数',
            reduction: 25,
            temperature: 400,
            parameters: {
                rolling_params: {
                    roll_radius: 0.4,
                    initial_thickness: 0.02,
                    final_thickness: 0.015,
                    strip_width: 1.5,
                    roll_speed: 8.0,
                    friction_coefficient: 0.2,
                    initial_temperature: 673.15,
                    roll_temperature: 323.15
                },
                material_props: {
                    density: 2700,
                    youngs_modulus: 70e9,
                    poisson_ratio: 0.33,
                    thermal_conductivity: 200,
                    specific_heat: 900,
                    thermal_expansion: 23e-6
                }
            }
        }
    ],
    
    // 材料库
    MATERIALS: {
        carbon_steel: {
            name: '碳钢',
            properties: {
                density: 7850,
                youngs_modulus: 210e9,
                poisson_ratio: 0.3,
                thermal_conductivity: 50,
                specific_heat: 450,
                thermal_expansion: 12e-6
            }
        },
        stainless_steel: {
            name: '不锈钢',
            properties: {
                density: 7900,
                youngs_modulus: 200e9,
                poisson_ratio: 0.28,
                thermal_conductivity: 16,
                specific_heat: 500,
                thermal_expansion: 16e-6
            }
        },
        aluminum: {
            name: '铝合金',
            properties: {
                density: 2700,
                youngs_modulus: 70e9,
                poisson_ratio: 0.33,
                thermal_conductivity: 200,
                specific_heat: 900,
                thermal_expansion: 23e-6
            }
        }
    },
    
    // 验证规则
    VALIDATION: {
        ROLLING_PARAMS: {
            roll_radius: { min: 0.1, max: 2.0, step: 0.1 },
            initial_thickness: { min: 0.001, max: 0.1, step: 0.001 },
            final_thickness: { min: 0.0005, max: 0.1, step: 0.001 },
            strip_width: { min: 0.1, max: 5.0, step: 0.1 },
            roll_speed: { min: 0.1, max: 20.0, step: 0.1 },
            friction_coefficient: { min: 0, max: 1, step: 0.05 },
            initial_temperature: { min: 273.15, max: 1773.15, step: 10 },
            roll_temperature: { min: 273.15, max: 773.15, step: 10 }
        },
        SOLVER_PARAMS: {
            num_steps: { min: 10, max: 1000, step: 10 },
            output_interval: { min: 1, max: 100, step: 1 },
            mesh_size: { min: 0.0001, max: 0.01, step: 0.0001 },
            max_iterations: { min: 100, max: 10000, step: 100 }
        }
    },
    
    // 本地存储键
    STORAGE_KEYS: {
        LAST_PARAMETERS: 'freefem_last_parameters',
        USER_PREFERENCES: 'freefem_user_preferences',
        TASK_HISTORY: 'freefem_task_history',
        AUTH_TOKEN: 'freefem_auth_token'
    },
    
    // 调试模式
    DEBUG: false,
    
    // 语言设置
    LOCALE: 'zh-CN',
    
    // 时区
    TIMEZONE: 'Asia/Shanghai'
};

// 环境相关配置覆盖
if (window.location.hostname === 'localhost') {
    CONFIG.DEBUG = true;
} else if (window.location.hostname.includes('test')) {
    CONFIG.API.BASE_URL = 'https://test-api.example.com';
} else if (window.location.hostname.includes('prod')) {
    CONFIG.API.BASE_URL = 'https://api.example.com';
}

// 导出配置
window.CONFIG = CONFIG;
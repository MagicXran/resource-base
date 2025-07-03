// -*- coding: gb2312 -*-
/**
 * 主应用程序 - Vue.js应用
 */

// 确保中文字符正确显示
document.characterSet = 'GB2312';

const { createApp, ref, reactive, computed, watch, onMounted, onUnmounted } = Vue;

const app = createApp({
    setup() {
        // ============================================
        // 状态管理
        // ============================================
        
        // 系统状态
        const systemStatus = ref('online'); // online, offline, busy
        const currentTime = ref(Utils.formatDate(new Date()));
        const isLoading = ref(false);
        const loadingMessage = ref('正在处理...');
        
        // 参数状态
        const parameters = reactive({
            rolling_params: { ...CONFIG.DEFAULTS.ROLLING_PARAMS },
            material_props: { ...CONFIG.DEFAULTS.MATERIAL_PROPS },
            solver_params: { ...CONFIG.DEFAULTS.SOLVER_PARAMS }
        });
        
        // 界面状态
        const activeTab = ref('geometry');
        const selectedMaterial = ref('carbon_steel');
        const showVisualization = ref(false);
        const showPresetModal = ref(false);
        const visualizationUrl = ref('');
        
        // 任务状态
        const currentTask = ref(null);
        const results = ref(null);
        const taskHistory = ref([]);
        const historyFilter = reactive({
            status: '',
            search: ''
        });
        const currentPage = ref(1);
        const pageSize = ref(CONFIG.UI.PAGE_SIZE);
        const isRefreshing = ref(false);
        
        // 验证状态
        const validationErrors = ref([]);
        const isValid = computed(() => validationErrors.value.length === 0);
        
        // 通知列表
        const notifications = ref([]);
        let notificationId = 0;
        
        // 定时器
        let timeUpdateTimer = null;
        let statusCheckTimer = null;
        let progressWs = null;
        
        // ============================================
        // 计算属性
        // ============================================
        
        // 单位转换
        const thickness_initial_mm = computed({
            get: () => parameters.rolling_params.initial_thickness * 1000,
            set: (val) => { parameters.rolling_params.initial_thickness = val / 1000; }
        });
        
        const thickness_final_mm = computed({
            get: () => parameters.rolling_params.final_thickness * 1000,
            set: (val) => { parameters.rolling_params.final_thickness = val / 1000; }
        });
        
        const temperature_initial_c = computed({
            get: () => parameters.rolling_params.initial_temperature - 273.15,
            set: (val) => { parameters.rolling_params.initial_temperature = val + 273.15; }
        });
        
        const temperature_roll_c = computed({
            get: () => parameters.rolling_params.roll_temperature - 273.15,
            set: (val) => { parameters.rolling_params.roll_temperature = val + 273.15; }
        });
        
        const youngs_modulus_gpa = computed({
            get: () => parameters.material_props.youngs_modulus / 1e9,
            set: (val) => { parameters.material_props.youngs_modulus = val * 1e9; }
        });
        
        const mesh_size_mm = computed({
            get: () => parameters.solver_params.mesh_size * 1000,
            set: (val) => { parameters.solver_params.mesh_size = val / 1000; }
        });
        
        // 压下率计算
        const reductionRatio = computed(() => {
            const initial = parameters.rolling_params.initial_thickness;
            const final = parameters.rolling_params.final_thickness;
            if (initial > 0) {
                return ((initial - final) / initial * 100).toFixed(1);
            }
            return 0;
        });
        
        // 是否正在运行
        const isRunning = computed(() => 
            currentTask.value && ['pending', 'queued', 'running'].includes(currentTask.value.status)
        );
        
        // 已用时间
        const elapsedTime = ref(0);
        let startTime = null;
        let elapsedTimer = null;
        
        // 执行时间
        const executionTime = computed(() => {
            if (results.value && results.value.execution_time) {
                return results.value.execution_time;
            }
            return 0;
        });
        
        // 系统状态文本
        const systemStatusText = computed(() => {
            const statusMap = {
                'online': '系统正常',
                'offline': '系统离线',
                'busy': '系统繁忙'
            };
            return statusMap[systemStatus.value] || '未知';
        });
        
        // 参数标签
        const parameterTabs = [
            { key: 'geometry', name: '几何参数', icon: 'fas fa-shapes' },
            { key: 'process', name: '工艺参数', icon: 'fas fa-cogs' },
            { key: 'material', name: '材料参数', icon: 'fas fa-cube' },
            { key: 'solver', name: '计算设置', icon: 'fas fa-calculator' }
        ];
        
        // 预设列表
        const presets = CONFIG.PRESETS;
        
        // 过滤后的任务
        const filteredTasks = computed(() => {
            let tasks = taskHistory.value;
            
            if (historyFilter.status) {
                tasks = tasks.filter(task => task.status === historyFilter.status);
            }
            
            if (historyFilter.search) {
                const search = historyFilter.search.toLowerCase();
                tasks = tasks.filter(task => 
                    task.task_id.toLowerCase().includes(search)
                );
            }
            
            // 分页
            const start = (currentPage.value - 1) * pageSize.value;
            const end = start + pageSize.value;
            
            return tasks.slice(start, end);
        });
        
        // 总页数
        const totalPages = computed(() => {
            return Math.ceil(taskHistory.value.length / pageSize.value);
        });
        
        // ============================================
        // 方法
        // ============================================
        
        // 验证参数
        const validateParameters = () => {
            const errors = [];
            const { rolling_params, solver_params } = parameters;
            
            // 验证厚度
            if (rolling_params.initial_thickness <= rolling_params.final_thickness) {
                errors.push('初始厚度必须大于最终厚度');
            }
            
            // 验证压下率
            const reduction = parseFloat(reductionRatio.value);
            if (reduction > 80) {
                errors.push('压下率过大（建议小于80%）');
            }
            
            // 验证温度
            if (rolling_params.initial_temperature < rolling_params.roll_temperature) {
                errors.push('板材温度应高于轧辊温度');
            }
            
            // 验证网格
            if (solver_params.mesh_size > rolling_params.final_thickness / 2) {
                errors.push('网格尺寸过大，建议小于最终厚度的一半');
            }
            
            validationErrors.value = errors;
            return errors.length === 0;
        };
        
        // 开始模拟
        const startSimulation = async () => {
            if (!validateParameters()) {
                showNotification('请修正参数错误后再试', 'error');
                return;
            }
            
            isLoading.value = true;
            loadingMessage.value = '正在创建模拟任务...';
            
            try {
                // 保存参数到本地
                Utils.storage.set(CONFIG.STORAGE_KEYS.LAST_PARAMETERS, parameters);
                
                // 创建模拟任务
                const response = await API.simulation.createSimulation(parameters);
                
                // 更新当前任务
                currentTask.value = {
                    task_id: response.task_id,
                    status: 'queued',
                    progress: 0,
                    message: '任务已创建，等待执行...',
                    created_at: new Date().toISOString()
                };
                
                // 开始时间
                startTime = Date.now();
                startElapsedTimer();
                
                // 连接WebSocket
                connectProgressWebSocket(response.task_id);
                
                // 添加到历史
                taskHistory.value.unshift(currentTask.value);
                
                showNotification('模拟任务已创建', 'success');
                
            } catch (error) {
                showNotification(error.message || '创建任务失败', 'error');
            } finally {
                isLoading.value = false;
            }
        };
        
        // 取消模拟
        const cancelSimulation = async () => {
            if (!currentTask.value) return;
            
            if (!confirm('确定要取消当前任务吗？')) return;
            
            isLoading.value = true;
            loadingMessage.value = '正在取消任务...';
            
            try {
                await API.simulation.cancelSimulation(currentTask.value.task_id);
                currentTask.value.status = 'cancelled';
                showNotification('任务已取消', 'warning');
            } catch (error) {
                showNotification(error.message || '取消任务失败', 'error');
            } finally {
                isLoading.value = false;
                stopElapsedTimer();
            }
        };
        
        // 连接进度WebSocket
        const connectProgressWebSocket = (taskId) => {
            // 断开旧连接
            if (progressWs) {
                API.ws.disconnect(taskId);
            }
            
            // 建立新连接
            progressWs = API.ws.connect(taskId, {
                onMessage: (data) => {
                    if (currentTask.value && currentTask.value.task_id === taskId) {
                        currentTask.value.status = data.status;
                        currentTask.value.progress = data.progress;
                        currentTask.value.message = data.message;
                        
                        if (data.status === 'completed') {
                            stopElapsedTimer();
                            loadResults(taskId);
                            showNotification('模拟计算完成', 'success');
                        } else if (data.status === 'failed') {
                            stopElapsedTimer();
                            currentTask.value.error = data.error;
                            showNotification('模拟计算失败', 'error');
                        }
                    }
                },
                onClose: () => {
                    progressWs = null;
                },
                onError: (error) => {
                    console.error('WebSocket错误:', error);
                }
            });
        };
        
        // 加载结果
        const loadResults = async (taskId) => {
            try {
                const data = await API.simulation.getResults(taskId);
                results.value = data;
            } catch (error) {
                showNotification('加载结果失败', 'error');
            }
        };
        
        // 重置参数
        const resetParameters = () => {
            Object.assign(parameters.rolling_params, CONFIG.DEFAULTS.ROLLING_PARAMS);
            Object.assign(parameters.material_props, CONFIG.DEFAULTS.MATERIAL_PROPS);
            Object.assign(parameters.solver_params, CONFIG.DEFAULTS.SOLVER_PARAMS);
            selectedMaterial.value = 'carbon_steel';
            validateParameters();
            showNotification('参数已重置', 'info');
        };
        
        // 加载预设
        const loadPreset = () => {
            showPresetModal.value = true;
        };
        
        // 选择预设
        const selectPreset = (preset) => {
            if (preset.parameters.rolling_params) {
                Object.assign(parameters.rolling_params, preset.parameters.rolling_params);
            }
            if (preset.parameters.material_props) {
                Object.assign(parameters.material_props, preset.parameters.material_props);
            }
            if (preset.parameters.solver_params) {
                Object.assign(parameters.solver_params, preset.parameters.solver_params);
            }
            
            showPresetModal.value = false;
            validateParameters();
            showNotification(`已加载预设: ${preset.name}`, 'success');
        };
        
        // 关闭预设模态框
        const closePresetModal = () => {
            showPresetModal.value = false;
        };
        
        // 保存参数
        const saveParameters = () => {
            const data = JSON.stringify(parameters, null, 2);
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const filename = `rolling_params_${Utils.formatDate(new Date(), 'YYYY-MM-DD_HH-mm-ss')}.json`;
            
            Utils.downloadFile(url, filename);
            URL.revokeObjectURL(url);
            
            showNotification('参数已保存', 'success');
        };
        
        // 加载材料属性
        const loadMaterialProperties = () => {
            const material = CONFIG.MATERIALS[selectedMaterial.value];
            if (material && selectedMaterial.value !== 'custom') {
                Object.assign(parameters.material_props, material.properties);
                validateParameters();
            }
        };
        
        // 打开可视化
        const openVisualization = () => {
            if (!results.value) return;
            
            // 构建ParaviewWeb URL
            const baseUrl = CONFIG.PARAVIEW.LAUNCHER_URL;
            const simId = currentTask.value.task_id;
            visualizationUrl.value = `${baseUrl}?session=${simId}`;
            
            showVisualization.value = true;
        };
        
        // 关闭可视化
        const closeVisualization = () => {
            showVisualization.value = false;
            visualizationUrl.value = '';
        };
        
        // 全屏可视化
        const fullscreenVisualization = () => {
            const iframe = document.querySelector('.visualization-iframe');
            if (iframe.requestFullscreen) {
                iframe.requestFullscreen();
            } else if (iframe.webkitRequestFullscreen) {
                iframe.webkitRequestFullscreen();
            } else if (iframe.msRequestFullscreen) {
                iframe.msRequestFullscreen();
            }
        };
        
        // 下载VTK
        const downloadVTK = async () => {
            if (!currentTask.value) return;
            
            try {
                await API.simulation.downloadVTK(currentTask.value.task_id);
                showNotification('VTK文件下载成功', 'success');
            } catch (error) {
                showNotification('下载失败', 'error');
            }
        };
        
        // 查看报告
        const viewReport = async () => {
            if (!currentTask.value) return;
            
            try {
                await API.simulation.downloadReport(currentTask.value.task_id);
                showNotification('报告下载成功', 'success');
            } catch (error) {
                showNotification('下载失败', 'error');
            }
        };
        
        // 分享结果
        const shareResults = async () => {
            if (!currentTask.value) return;
            
            const shareUrl = `${window.location.origin}/results/${currentTask.value.task_id}`;
            
            try {
                await Utils.copyToClipboard(shareUrl);
                showNotification('分享链接已复制到剪贴板', 'success');
            } catch (error) {
                showNotification('复制失败', 'error');
            }
        };
        
        // 复制任务ID
        const copyTaskId = async () => {
            if (!currentTask.value) return;
            
            const success = await Utils.copyToClipboard(currentTask.value.task_id);
            if (success) {
                showNotification('任务ID已复制', 'success');
            }
        };
        
        // 刷新历史
        const refreshHistory = async () => {
            isRefreshing.value = true;
            
            try {
                const data = await API.simulation.listSimulations({
                    skip: 0,
                    limit: 100
                });
                taskHistory.value = data.items || [];
                showNotification('历史记录已更新', 'success');
            } catch (error) {
                showNotification('刷新失败', 'error');
            } finally {
                isRefreshing.value = false;
            }
        };
        
        // 加载任务
        const loadTask = async (taskId) => {
            isLoading.value = true;
            loadingMessage.value = '正在加载任务...';
            
            try {
                const task = await API.simulation.getSimulation(taskId);
                currentTask.value = task;
                
                if (task.status === 'completed') {
                    await loadResults(taskId);
                } else if (['pending', 'queued', 'running'].includes(task.status)) {
                    connectProgressWebSocket(taskId);
                }
                
                showNotification('任务已加载', 'success');
            } catch (error) {
                showNotification('加载任务失败', 'error');
            } finally {
                isLoading.value = false;
            }
        };
        
        // 跟踪任务
        const trackTask = (taskId) => {
            loadTask(taskId);
        };
        
        // 删除任务
        const deleteTask = async (taskId) => {
            if (!confirm('确定要删除这个任务吗？')) return;
            
            try {
                await API.simulation.deleteSimulation(taskId);
                taskHistory.value = taskHistory.value.filter(t => t.task_id !== taskId);
                
                if (currentTask.value && currentTask.value.task_id === taskId) {
                    currentTask.value = null;
                    results.value = null;
                }
                
                showNotification('任务已删除', 'success');
            } catch (error) {
                showNotification('删除失败', 'error');
            }
        };
        
        // 显示通知
        const showNotification = (message, type = 'info') => {
            const id = ++notificationId;
            const notification = { id, message, type };
            
            notifications.value.push(notification);
            
            // 自动移除
            setTimeout(() => {
                removeNotification(id);
            }, CONFIG.UI.NOTIFICATION_DURATION);
        };
        
        // 移除通知
        const removeNotification = (id) => {
            const index = notifications.value.findIndex(n => n.id === id);
            if (index > -1) {
                notifications.value.splice(index, 1);
            }
        };
        
        // 获取状态图标
        const getStatusIcon = (status) => {
            const icons = {
                'pending': 'fas fa-clock',
                'queued': 'fas fa-hourglass-half',
                'running': 'fas fa-spinner fa-spin',
                'completed': 'fas fa-check-circle',
                'failed': 'fas fa-times-circle',
                'cancelled': 'fas fa-ban'
            };
            return icons[status] || 'fas fa-question-circle';
        };
        
        // 获取状态文本
        const getStatusText = (status) => {
            const texts = {
                'pending': '等待中',
                'queued': '排队中',
                'running': '运行中',
                'completed': '已完成',
                'failed': '失败',
                'cancelled': '已取消'
            };
            return texts[status] || '未知';
        };
        
        // 获取通知图标
        const getNotificationIcon = (type) => {
            const icons = {
                'success': 'fas fa-check-circle',
                'error': 'fas fa-exclamation-circle',
                'warning': 'fas fa-exclamation-triangle',
                'info': 'fas fa-info-circle'
            };
            return icons[type] || 'fas fa-info-circle';
        };
        
        // 格式化方法
        const formatDate = Utils.formatDate;
        const formatDuration = Utils.formatDuration;
        const formatTemperature = Utils.formatTemperature;
        const formatStress = Utils.formatStress;
        const formatForce = Utils.formatForce;
        
        // 开始计时
        const startElapsedTimer = () => {
            stopElapsedTimer();
            elapsedTime.value = 0;
            
            elapsedTimer = setInterval(() => {
                if (startTime) {
                    elapsedTime.value = Math.floor((Date.now() - startTime) / 1000);
                }
            }, 1000);
        };
        
        // 停止计时
        const stopElapsedTimer = () => {
            if (elapsedTimer) {
                clearInterval(elapsedTimer);
                elapsedTimer = null;
            }
            startTime = null;
        };
        
        // 更新系统时间
        const updateSystemTime = () => {
            currentTime.value = Utils.formatDate(new Date());
        };
        
        // 检查系统状态
        const checkSystemStatus = async () => {
            try {
                const health = await API.system.healthCheck();
                systemStatus.value = health.status === 'healthy' ? 'online' : 'offline';
            } catch (error) {
                systemStatus.value = 'offline';
            }
        };
        
        // ============================================
        // 生命周期
        // ============================================
        
        onMounted(async () => {
            // 加载保存的参数
            const savedParams = Utils.storage.get(CONFIG.STORAGE_KEYS.LAST_PARAMETERS);
            if (savedParams) {
                Object.assign(parameters, savedParams);
            }
            
            // 检查URL参数
            const taskId = Utils.getQueryParam('task');
            if (taskId) {
                loadTask(taskId);
            }
            
            // 启动定时器
            timeUpdateTimer = setInterval(updateSystemTime, 1000);
            statusCheckTimer = setInterval(checkSystemStatus, 30000);
            
            // 初始检查
            checkSystemStatus();
            refreshHistory();
            
            // 键盘快捷键
            document.addEventListener('keydown', handleKeyboard);
        });
        
        onUnmounted(() => {
            // 清理定时器
            if (timeUpdateTimer) clearInterval(timeUpdateTimer);
            if (statusCheckTimer) clearInterval(statusCheckTimer);
            stopElapsedTimer();
            
            // 断开WebSocket
            API.ws.disconnectAll();
            
            // 移除事件监听
            document.removeEventListener('keydown', handleKeyboard);
        });
        
        // 键盘事件处理
        const handleKeyboard = (event) => {
            // Ctrl/Cmd + Enter: 开始模拟
            if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                if (!isRunning.value) {
                    startSimulation();
                }
            }
            
            // Escape: 关闭模态框
            if (event.key === 'Escape') {
                if (showVisualization.value) {
                    closeVisualization();
                }
                if (showPresetModal.value) {
                    closePresetModal();
                }
            }
        };
        
        // ============================================
        // 返回模板需要的数据和方法
        // ============================================
        
        return {
            // 状态
            systemStatus,
            systemStatusText,
            currentTime,
            isLoading,
            loadingMessage,
            
            // 参数
            parameters,
            activeTab,
            selectedMaterial,
            parameterTabs,
            presets,
            
            // 计算属性
            thickness_initial_mm,
            thickness_final_mm,
            temperature_initial_c,
            temperature_roll_c,
            youngs_modulus_gpa,
            mesh_size_mm,
            reductionRatio,
            isRunning,
            isValid,
            elapsedTime,
            executionTime,
            
            // 任务相关
            currentTask,
            results,
            taskHistory,
            historyFilter,
            filteredTasks,
            currentPage,
            totalPages,
            isRefreshing,
            
            // 验证
            validationErrors,
            
            // 通知
            notifications,
            
            // 模态框
            showVisualization,
            showPresetModal,
            visualizationUrl,
            
            // 方法
            validateParameters,
            startSimulation,
            cancelSimulation,
            resetParameters,
            loadPreset,
            selectPreset,
            closePresetModal,
            saveParameters,
            loadMaterialProperties,
            openVisualization,
            closeVisualization,
            fullscreenVisualization,
            downloadVTK,
            viewReport,
            shareResults,
            copyTaskId,
            refreshHistory,
            loadTask,
            trackTask,
            deleteTask,
            showNotification,
            removeNotification,
            getStatusIcon,
            getStatusText,
            getNotificationIcon,
            formatDate,
            formatDuration,
            formatTemperature,
            formatStress,
            formatForce
        };
    }
});

// 挂载应用
app.mount('#app');

// 全局错误处理
window.addEventListener('error', (event) => {
    console.error('全局错误:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('未处理的Promise拒绝:', event.reason);
});

// 页面卸载时清理
window.addEventListener('beforeunload', () => {
    API.ws.disconnectAll();
});
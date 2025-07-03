// -*- coding: gb2312 -*-
/**
 * API客户端 - 处理所有HTTP请求
 */

class APIClient {
    constructor(config) {
        this.config = config;
        this.baseURL = config.API.BASE_URL;
        this.timeout = config.API.TIMEOUT;
        this.retryCount = config.API.RETRY_COUNT;
        this.retryDelay = config.API.RETRY_DELAY;
        
        // 请求拦截器
        this.requestInterceptors = [];
        // 响应拦截器
        this.responseInterceptors = [];
        
        // 初始化axios实例
        this.axios = axios.create({
            baseURL: this.baseURL,
            timeout: this.timeout,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        
        this._setupInterceptors();
    }
    
    /**
     * 设置拦截器
     */
    _setupInterceptors() {
        // 请求拦截器
        this.axios.interceptors.request.use(
            config => {
                // 添加认证token
                const token = Utils.storage.get(this.config.STORAGE_KEYS.AUTH_TOKEN);
                if (token) {
                    config.headers['Authorization'] = `Bearer ${token}`;
                }
                
                // 添加时间戳防止缓存
                if (config.method === 'get') {
                    config.params = config.params || {};
                    config.params['_t'] = Date.now();
                }
                
                // 执行自定义拦截器
                this.requestInterceptors.forEach(interceptor => {
                    config = interceptor(config);
                });
                
                return config;
            },
            error => {
                return Promise.reject(error);
            }
        );
        
        // 响应拦截器
        this.axios.interceptors.response.use(
            response => {
                // 执行自定义拦截器
                this.responseInterceptors.forEach(interceptor => {
                    response = interceptor(response);
                });
                
                return response.data;
            },
            async error => {
                if (error.response) {
                    // 服务器返回错误
                    const { status, data } = error.response;
                    
                    switch (status) {
                        case 401:
                            // 未授权，清除token
                            Utils.storage.remove(this.config.STORAGE_KEYS.AUTH_TOKEN);
                            window.location.href = '/login';
                            break;
                        case 403:
                            throw new Error('没有权限访问该资源');
                        case 404:
                            throw new Error('请求的资源不存在');
                        case 429:
                            throw new Error('请求过于频繁，请稍后重试');
                        case 500:
                            throw new Error(data.detail || '服务器内部错误');
                        default:
                            throw new Error(data.detail || `请求失败 (${status})`);
                    }
                } else if (error.request) {
                    // 请求已发送但没有收到响应
                    throw new Error('网络连接失败，请检查网络设置');
                } else {
                    // 请求配置出错
                    throw new Error('请求配置错误');
                }
            }
        );
    }
    
    /**
     * 通用请求方法（带重试）
     */
    async _request(method, url, data = null, config = {}) {
        let lastError;
        
        for (let i = 0; i <= this.retryCount; i++) {
            try {
                const response = await this.axios({
                    method,
                    url,
                    data,
                    ...config
                });
                return response;
            } catch (error) {
                lastError = error;
                
                // 不重试的情况
                if (i === this.retryCount || 
                    (error.response && error.response.status < 500)) {
                    throw error;
                }
                
                // 等待后重试
                await Utils.delay(this.retryDelay * (i + 1));
            }
        }
        
        throw lastError;
    }
    
    /**
     * GET请求
     */
    async get(url, params = {}, config = {}) {
        return this._request('get', url, null, { params, ...config });
    }
    
    /**
     * POST请求
     */
    async post(url, data = {}, config = {}) {
        return this._request('post', url, data, config);
    }
    
    /**
     * PUT请求
     */
    async put(url, data = {}, config = {}) {
        return this._request('put', url, data, config);
    }
    
    /**
     * DELETE请求
     */
    async delete(url, config = {}) {
        return this._request('delete', url, null, config);
    }
    
    /**
     * 上传文件
     */
    async upload(url, file, onProgress) {
        const formData = new FormData();
        formData.append('file', file);
        
        return this.post(url, formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            },
            onUploadProgress: progressEvent => {
                if (onProgress) {
                    const percentCompleted = Math.round(
                        (progressEvent.loaded * 100) / progressEvent.total
                    );
                    onProgress(percentCompleted);
                }
            }
        });
    }
    
    /**
     * 下载文件
     */
    async download(url, filename) {
        const response = await this.get(url, {}, {
            responseType: 'blob'
        });
        
        const blob = new Blob([response]);
        const downloadUrl = window.URL.createObjectURL(blob);
        Utils.downloadFile(downloadUrl, filename);
        window.URL.revokeObjectURL(downloadUrl);
    }
}

/**
 * 模拟API
 */
class SimulationAPI {
    constructor(client) {
        this.client = client;
        this.endpoints = CONFIG.API.ENDPOINTS;
    }
    
    /**
     * 创建模拟任务
     */
    async createSimulation(params) {
        return this.client.post(this.endpoints.CREATE_SIMULATION, params);
    }
    
    /**
     * 获取模拟任务详情
     */
    async getSimulation(id) {
        const url = this.endpoints.GET_SIMULATION.replace('{id}', id);
        return this.client.get(url);
    }
    
    /**
     * 获取模拟任务列表
     */
    async listSimulations(params = {}) {
        return this.client.get(this.endpoints.LIST_SIMULATIONS, params);
    }
    
    /**
     * 取消模拟任务
     */
    async cancelSimulation(id) {
        const url = this.endpoints.CANCEL_SIMULATION.replace('{id}', id);
        return this.client.post(url);
    }
    
    /**
     * 删除模拟任务
     */
    async deleteSimulation(id) {
        const url = this.endpoints.DELETE_SIMULATION.replace('{id}', id);
        return this.client.delete(url);
    }
    
    /**
     * 获取模拟结果
     */
    async getResults(id) {
        const url = this.endpoints.GET_RESULTS.replace('{id}', id);
        return this.client.get(url);
    }
    
    /**
     * 下载VTK文件
     */
    async downloadVTK(id, filename) {
        const url = this.endpoints.DOWNLOAD_VTK.replace('{id}', id);
        return this.client.download(url, filename || `simulation_${id}.vtk`);
    }
    
    /**
     * 下载报告
     */
    async downloadReport(id, filename) {
        const url = this.endpoints.DOWNLOAD_REPORT.replace('{id}', id);
        return this.client.download(url, filename || `report_${id}.pdf`);
    }
}

/**
 * 系统API
 */
class SystemAPI {
    constructor(client) {
        this.client = client;
        this.endpoints = CONFIG.API.ENDPOINTS;
    }
    
    /**
     * 健康检查
     */
    async healthCheck() {
        return this.client.get(this.endpoints.HEALTH_CHECK);
    }
    
    /**
     * 获取系统信息
     */
    async getSystemInfo() {
        return this.client.get(this.endpoints.SYSTEM_INFO);
    }
}

/**
 * WebSocket管理器
 */
class WebSocketManager {
    constructor(config) {
        this.config = config;
        this.connections = new Map();
        this.reconnectAttempts = new Map();
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }
    
    /**
     * 连接WebSocket
     */
    connect(id, handlers = {}) {
        const url = `${this.config.API.BASE_URL.replace('http', 'ws')}${this.config.API.ENDPOINTS.WS_PROGRESS.replace('{id}', id)}`;
        
        // 如果已存在连接，先关闭
        if (this.connections.has(id)) {
            this.disconnect(id);
        }
        
        const ws = new WebSocket(url);
        
        ws.onopen = () => {
            console.log(`WebSocket connected for task ${id}`);
            this.reconnectAttempts.set(id, 0);
            if (handlers.onOpen) handlers.onOpen();
        };
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (handlers.onMessage) handlers.onMessage(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };
        
        ws.onerror = (error) => {
            console.error(`WebSocket error for task ${id}:`, error);
            if (handlers.onError) handlers.onError(error);
        };
        
        ws.onclose = (event) => {
            console.log(`WebSocket closed for task ${id}`);
            this.connections.delete(id);
            
            // 尝试重连
            if (!event.wasClean && this.shouldReconnect(id)) {
                this.reconnect(id, handlers);
            } else {
                this.reconnectAttempts.delete(id);
                if (handlers.onClose) handlers.onClose(event);
            }
        };
        
        this.connections.set(id, ws);
        return ws;
    }
    
    /**
     * 断开连接
     */
    disconnect(id) {
        const ws = this.connections.get(id);
        if (ws) {
            ws.close();
            this.connections.delete(id);
            this.reconnectAttempts.delete(id);
        }
    }
    
    /**
     * 断开所有连接
     */
    disconnectAll() {
        this.connections.forEach((ws, id) => {
            this.disconnect(id);
        });
    }
    
    /**
     * 发送消息
     */
    send(id, data) {
        const ws = this.connections.get(id);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(data));
            return true;
        }
        return false;
    }
    
    /**
     * 是否应该重连
     */
    shouldReconnect(id) {
        const attempts = this.reconnectAttempts.get(id) || 0;
        return attempts < this.maxReconnectAttempts;
    }
    
    /**
     * 重连
     */
    reconnect(id, handlers) {
        const attempts = this.reconnectAttempts.get(id) || 0;
        this.reconnectAttempts.set(id, attempts + 1);
        
        setTimeout(() => {
            console.log(`Attempting to reconnect WebSocket for task ${id} (attempt ${attempts + 1})`);
            this.connect(id, handlers);
        }, this.reconnectDelay * Math.pow(2, attempts));
    }
}

// 创建API实例
const apiClient = new APIClient(CONFIG);
const simulationAPI = new SimulationAPI(apiClient);
const systemAPI = new SystemAPI(apiClient);
const wsManager = new WebSocketManager(CONFIG);

// 导出API
window.API = {
    client: apiClient,
    simulation: simulationAPI,
    system: systemAPI,
    ws: wsManager
};
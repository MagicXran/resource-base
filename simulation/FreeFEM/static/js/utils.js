// -*- coding: gb2312 -*-
/**
 * 工具函数库
 */

const Utils = {
    /**
     * 格式化日期时间
     */
    formatDate(date, format = 'YYYY-MM-DD HH:mm:ss') {
        if (!date) return '';
        
        const d = new Date(date);
        const year = d.getFullYear();
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        const hours = String(d.getHours()).padStart(2, '0');
        const minutes = String(d.getMinutes()).padStart(2, '0');
        const seconds = String(d.getSeconds()).padStart(2, '0');
        
        return format
            .replace('YYYY', year)
            .replace('MM', month)
            .replace('DD', day)
            .replace('HH', hours)
            .replace('mm', minutes)
            .replace('ss', seconds);
    },
    
    /**
     * 格式化持续时间
     */
    formatDuration(seconds) {
        if (!seconds || seconds < 0) return '0秒';
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        const parts = [];
        if (hours > 0) parts.push(`${hours}小时`);
        if (minutes > 0) parts.push(`${minutes}分钟`);
        if (secs > 0 || parts.length === 0) parts.push(`${secs}秒`);
        
        return parts.join(' ');
    },
    
    /**
     * 格式化文件大小
     */
    formatFileSize(bytes) {
        if (!bytes || bytes === 0) return '0 B';
        
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        const k = 1024;
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return `${(bytes / Math.pow(k, i)).toFixed(2)} ${units[i]}`;
    },
    
    /**
     * 格式化数字
     */
    formatNumber(num, decimals = 2) {
        if (num === null || num === undefined) return '-';
        
        const fixed = Number(num).toFixed(decimals);
        const parts = fixed.split('.');
        parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        
        return parts.join('.');
    },
    
    /**
     * 格式化百分比
     */
    formatPercent(value, decimals = 1) {
        if (value === null || value === undefined) return '-';
        return `${(value * 100).toFixed(decimals)}%`;
    },
    
    /**
     * 格式化温度（开尔文转摄氏度）
     */
    formatTemperature(kelvin) {
        if (!kelvin) return '-';
        const celsius = kelvin - 273.15;
        return `${celsius.toFixed(1)}°C`;
    },
    
    /**
     * 格式化应力（Pa转MPa）
     */
    formatStress(pa) {
        if (!pa) return '-';
        const mpa = pa / 1e6;
        return `${mpa.toFixed(1)} MPa`;
    },
    
    /**
     * 格式化力（N转kN）
     */
    formatForce(newton) {
        if (!newton) return '-';
        const kn = newton / 1000;
        return `${kn.toFixed(1)} kN`;
    },
    
    /**
     * 深拷贝对象
     */
    deepClone(obj) {
        if (obj === null || typeof obj !== 'object') return obj;
        if (obj instanceof Date) return new Date(obj.getTime());
        if (obj instanceof Array) return obj.map(item => this.deepClone(item));
        
        const cloned = {};
        for (const key in obj) {
            if (obj.hasOwnProperty(key)) {
                cloned[key] = this.deepClone(obj[key]);
            }
        }
        return cloned;
    },
    
    /**
     * 防抖函数
     */
    debounce(func, wait = 300) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    /**
     * 节流函数
     */
    throttle(func, limit = 300) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },
    
    /**
     * 生成UUID
     */
    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    },
    
    /**
     * 复制到剪贴板
     */
    async copyToClipboard(text) {
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(text);
                return true;
            } else {
                // 降级方案
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'absolute';
                textArea.style.left = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                
                try {
                    document.execCommand('copy');
                    return true;
                } finally {
                    textArea.remove();
                }
            }
        } catch (err) {
            console.error('复制失败:', err);
            return false;
        }
    },
    
    /**
     * 下载文件
     */
    downloadFile(url, filename) {
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || 'download';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    },
    
    /**
     * 获取URL参数
     */
    getQueryParam(name) {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get(name);
    },
    
    /**
     * 设置URL参数
     */
    setQueryParam(name, value) {
        const url = new URL(window.location);
        if (value) {
            url.searchParams.set(name, value);
        } else {
            url.searchParams.delete(name);
        }
        window.history.pushState({}, '', url);
    },
    
    /**
     * 验证邮箱
     */
    isValidEmail(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    },
    
    /**
     * 验证手机号（中国）
     */
    isValidPhone(phone) {
        const re = /^1[3-9]\d{9}$/;
        return re.test(phone);
    },
    
    /**
     * 转义HTML
     */
    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },
    
    /**
     * 本地存储封装
     */
    storage: {
        get(key) {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : null;
            } catch (e) {
                console.error('读取存储失败:', e);
                return null;
            }
        },
        
        set(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (e) {
                console.error('存储失败:', e);
                return false;
            }
        },
        
        remove(key) {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (e) {
                console.error('删除存储失败:', e);
                return false;
            }
        },
        
        clear() {
            try {
                localStorage.clear();
                return true;
            } catch (e) {
                console.error('清空存储失败:', e);
                return false;
            }
        }
    },
    
    /**
     * 判断是否为移动设备
     */
    isMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    },
    
    /**
     * 获取浏览器信息
     */
    getBrowserInfo() {
        const ua = navigator.userAgent;
        let browser = 'Unknown';
        let version = 'Unknown';
        
        if (ua.indexOf('Firefox') > -1) {
            browser = 'Firefox';
            version = ua.match(/Firefox\/(\d+\.\d+)/)[1];
        } else if (ua.indexOf('Chrome') > -1) {
            browser = 'Chrome';
            version = ua.match(/Chrome\/(\d+\.\d+)/)[1];
        } else if (ua.indexOf('Safari') > -1) {
            browser = 'Safari';
            version = ua.match(/Version\/(\d+\.\d+)/)[1];
        } else if (ua.indexOf('Edge') > -1) {
            browser = 'Edge';
            version = ua.match(/Edge\/(\d+\.\d+)/)[1];
        }
        
        return { browser, version };
    },
    
    /**
     * 数组分块
     */
    chunk(array, size) {
        const chunks = [];
        for (let i = 0; i < array.length; i += size) {
            chunks.push(array.slice(i, i + size));
        }
        return chunks;
    },
    
    /**
     * 延迟函数
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    },
    
    /**
     * 重试函数
     */
    async retry(fn, retries = 3, delay = 1000) {
        for (let i = 0; i < retries; i++) {
            try {
                return await fn();
            } catch (error) {
                if (i === retries - 1) throw error;
                await this.delay(delay);
            }
        }
    },
    
    /**
     * 计算哈希值
     */
    async hash(str) {
        const msgUint8 = new TextEncoder().encode(str);
        const hashBuffer = await crypto.subtle.digest('SHA-256', msgUint8);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
        return hashHex;
    },
    
    /**
     * 验证参数
     */
    validateParams(params, rules) {
        const errors = [];
        
        for (const [key, rule] of Object.entries(rules)) {
            const value = params[key];
            
            if (rule.required && (value === undefined || value === null || value === '')) {
                errors.push(`${key} 是必填项`);
                continue;
            }
            
            if (value !== undefined && value !== null) {
                if (rule.type && typeof value !== rule.type) {
                    errors.push(`${key} 类型错误，应为 ${rule.type}`);
                }
                
                if (rule.min !== undefined && value < rule.min) {
                    errors.push(`${key} 不能小于 ${rule.min}`);
                }
                
                if (rule.max !== undefined && value > rule.max) {
                    errors.push(`${key} 不能大于 ${rule.max}`);
                }
                
                if (rule.pattern && !rule.pattern.test(value)) {
                    errors.push(`${key} 格式不正确`);
                }
                
                if (rule.custom && !rule.custom(value)) {
                    errors.push(`${key} 验证失败`);
                }
            }
        }
        
        return errors;
    },
    
    /**
     * 颜色工具
     */
    color: {
        // RGB转HEX
        rgbToHex(r, g, b) {
            return '#' + [r, g, b].map(x => {
                const hex = x.toString(16);
                return hex.length === 1 ? '0' + hex : hex;
            }).join('');
        },
        
        // HEX转RGB
        hexToRgb(hex) {
            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            return result ? {
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16)
            } : null;
        },
        
        // 获取渐变色
        getGradient(startColor, endColor, steps) {
            const start = this.hexToRgb(startColor);
            const end = this.hexToRgb(endColor);
            const colors = [];
            
            for (let i = 0; i < steps; i++) {
                const ratio = i / (steps - 1);
                const r = Math.round(start.r + (end.r - start.r) * ratio);
                const g = Math.round(start.g + (end.g - start.g) * ratio);
                const b = Math.round(start.b + (end.b - start.b) * ratio);
                colors.push(this.rgbToHex(r, g, b));
            }
            
            return colors;
        }
    }
};

// 导出工具函数
window.Utils = Utils;
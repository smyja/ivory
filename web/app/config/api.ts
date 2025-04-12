// API configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

// API endpoints (Paths relative to API_BASE_URL)
export const API_ENDPOINTS = {
    datasets: {
        base: '/datasets',
        download: '/datasets/download/',
        stream: (datasetId: number) => `/datasets/download/${datasetId}/stream`,
        huggingface: {
            info: '/datasets/huggingface/info',
            configs: '/datasets/huggingface/configs',
            splits: '/datasets/huggingface/splits',
            features: '/datasets/huggingface/features',
        },
        clustering: {
            start: '/datasets/clustering/start',
            status: (datasetId: number) => `/datasets/clustering/${datasetId}/status`,
            results: (datasetId: number, version?: number) =>
                `/datasets/clustering/${datasetId}/results${version ? `?version=${version}` : ''}`,
            history: (datasetId: number) => `/datasets/clustering/${datasetId}/history`,
        },
        details: (datasetId: number) => `/datasets/${datasetId}`,
    },
    auth: {
        login: '/auth/login',
        register: '/auth/register',
        me: '/auth/me',
    },
}; 
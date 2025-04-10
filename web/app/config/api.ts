// API configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// API endpoints
export const API_ENDPOINTS = {
    datasets: {
        base: '/datasets',
        huggingface: {
            configs: '/datasets/huggingface/configs',
            splits: '/datasets/huggingface/splits',
            features: '/datasets/huggingface/features',
        },
    },
}; 
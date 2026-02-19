import axios from 'axios'

// Always append /api to ensure consistent routing
const BACKEND_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5000'
const API_URL = `${BACKEND_BASE}/api`

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 180000, // 180 second timeout for long operations like training
})

// Add token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle token expiration
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Only redirect on 401 if we're not already on the login page
    if (error.response?.status === 401 && typeof window !== 'undefined') {
      const currentPath = window.location.pathname
      if (currentPath !== '/') {
        localStorage.removeItem('token')
        localStorage.removeItem('userRole')
        localStorage.removeItem('userId')
        window.location.href = '/'
      }
    }
    return Promise.reject(error)
  }
)

export default api

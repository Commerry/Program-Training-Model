import axios from 'axios'

/**
 * Shared axios instance.
 *
 * All requests go to /api and are proxied to the backend by the Vite dev
 * server, so they are same-origin and the session cookie is sent normally.
 */
const http = axios.create({
  baseURL: '/api',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' }
})

/**
 * Callback invoked when the backend reports the session is gone.
 * Set by the auth store so this module does not import the router or a store,
 * which would create an import cycle.
 */
let onUnauthorized = null
export const setUnauthorizedHandler = (handler) => { onUnauthorized = handler }

/** Pull the human-readable message out of whatever the failure looked like. */
export const errorMessage = (error, fallback = 'Something went wrong') => {
  if (error?.response?.data?.message) return error.response.data.message
  if (error?.response?.status === 413) return 'Upload is too large.'
  if (error?.code === 'ERR_NETWORK') return 'Cannot reach the backend. Is it running?'
  return error?.message || fallback
}

http.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status
    const url = error?.config?.url || ''
    // A 401 from /auth/me is just "not signed in yet" during startup and must
    // not trigger the session-expired path.
    if (status === 401 && !url.includes('/auth/me') && !url.includes('/auth/login')) {
      onUnauthorized?.()
    }
    return Promise.reject(error)
  }
)

export default http

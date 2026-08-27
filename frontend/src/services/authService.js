import http from './http'

export const authService = {
  register: (payload) =>
    http.post('/auth/register', payload).then((r) => r.data),

  login: (username, password, remember = false) =>
    http.post('/auth/login', { username, password, remember }).then((r) => r.data),

  logout: () =>
    http.post('/auth/logout').then((r) => r.data),

  me: () =>
    http.get('/auth/me').then((r) => r.data),

  /**
   * How this server is configured, including whether it wants anyone to sign
   * in at all. A deployment started with REQUIRE_AUTH=0 accepts every request,
   * and the UI has no business showing a login wall in front of it.
   */
  health: () =>
    http.get('/health').then((r) => r.data),

  updateProfile: (payload) =>
    http.put('/auth/profile', payload).then((r) => r.data)
}

export default authService

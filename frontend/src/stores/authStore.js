import { defineStore } from 'pinia'
import { authService, errorMessage, setUnauthorizedHandler } from '@/services'

/**
 * Authentication state.
 *
 * The browser only holds a display copy of the user; the real session lives in
 * the backend's cookie. `ready` distinguishes "not signed in" from "we have not
 * asked the server yet", which the router guard needs so it does not bounce a
 * signed-in user to /login on a page refresh.
 */
export const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null,
    ready: false,
    loading: false,
    error: null,
    // Whether this server wants anyone signed in. Defaults to true so that a
    // server which cannot be reached is treated as locked rather than open.
    authRequired: true
  }),

  getters: {
    isAuthenticated: (state) => state.user !== null,
    /** True when the router should let a page through without a session. */
    canBrowseAnonymously: (state) => !state.authRequired,
    displayName: (state) => state.user?.full_name || state.user?.username || ''
  },

  actions: {
    async register(payload) {
      this.loading = true
      this.error = null
      try {
        const result = await authService.register(payload)
        this.user = result.user
        return result
      } catch (error) {
        this.error = errorMessage(error)
        throw new Error(this.error)
      } finally {
        this.loading = false
      }
    },

    async login(username, password, remember = false) {
      this.loading = true
      this.error = null
      try {
        const result = await authService.login(username, password, remember)
        this.user = result.user
        return result
      } catch (error) {
        this.error = errorMessage(error)
        throw new Error(this.error)
      } finally {
        this.loading = false
      }
    },

    async logout() {
      try {
        await authService.logout()
      } finally {
        this.user = null
      }
    },

    /**
     * Ask the backend who we are, and whether it cares. Called once on start.
     *
     * Both questions matter: a server run with REQUIRE_AUTH=0 answers /auth/me
     * with no user, and reading only that would send every visitor to a login
     * form the server does not want and the guard would never let them past.
     */
    async restore() {
      try {
        const health = await authService.health()
        this.authRequired = health.auth_required !== false
      } catch {
        this.authRequired = true
      }
      try {
        const result = await authService.me()
        this.user = result.user ?? null
      } catch {
        this.user = null
      } finally {
        this.ready = true
      }
    },

    async updateProfile(payload) {
      this.loading = true
      this.error = null
      try {
        const result = await authService.updateProfile(payload)
        this.user = result.user
        return result
      } catch (error) {
        this.error = errorMessage(error)
        throw new Error(this.error)
      } finally {
        this.loading = false
      }
    },

    /** Called by the HTTP layer when the backend rejects the session. */
    handleSessionExpired() {
      this.user = null
    }
  }
})

/** Wire the 401 interceptor to the store. Called once from main.js. */
export const installAuthInterceptor = (router) => {
  const store = useAuthStore()
  setUnauthorizedHandler(() => {
    store.handleSessionExpired()
    // On an open server there is no session to expire and no login to send
    // anyone to, so a stray 401 must not throw the user out of the page.
    if (!store.authRequired) return
    if (router.currentRoute.value.name !== 'Login') {
      router.push({ name: 'Login', query: { expired: '1' } })
    }
  })
}

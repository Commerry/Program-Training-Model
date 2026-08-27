<template>
  <div class="auth-page">
    <div class="auth-container">
      <div class="auth-card card">
        <div class="auth-header">
          <div class="auth-logo">
            <Icon name="box" size="lg" />
            <h1>Vision Training</h1>
          </div>
          <h2>Welcome Back</h2>
          <p>Sign in to your account to continue</p>
        </div>

        <form @submit.prevent="handleLogin" class="auth-form">
          <div v-if="sessionExpired && !error" class="error-message">
            Your session expired. Please sign in again.
          </div>
          <div v-if="error" class="error-message">
            {{ error }}
          </div>

          <div class="form-group">
            <label class="form-label">Username</label>
            <input
              v-model="formData.username"
              type="text"
              class="form-input"
              placeholder="Enter your username"
              autocomplete="username"
              required
              autofocus
            />
          </div>

          <div class="form-group">
            <label class="form-label">Password</label>
            <input
              v-model="formData.password"
              type="password"
              class="form-input"
              placeholder="Enter your password"
              autocomplete="current-password"
              required
            />
          </div>

          <div class="form-group">
            <label class="form-checkbox">
              <input type="checkbox" v-model="formData.remember" />
              <span>Remember me</span>
            </label>
          </div>

          <button type="submit" class="btn btn-primary btn-block" :disabled="loading">
            <span v-if="loading">Signing in...</span>
            <span v-else>Sign In</span>
          </button>
        </form>

        <div class="auth-footer">
          <p>
            Don't have an account?
            <router-link to="/register" class="auth-link">Sign up</router-link>
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import Icon from '@/components/Icon.vue'
import { useAuthStore } from '@/stores/authStore'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const formData = ref({ username: '', password: '', remember: false })
const loading = ref(false)
const error = ref('')

// The router guard adds ?redirect= when it bounces an unauthenticated user,
// and the 401 interceptor adds ?expired=1 when a session runs out.
const sessionExpired = computed(() => route.query.expired === '1')

const handleLogin = async () => {
  loading.value = true
  error.value = ''
  try {
    await authStore.login(
      formData.value.username,
      formData.value.password,
      formData.value.remember
    )
    const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : null
    // Only follow same-site paths, so a crafted link cannot bounce the user
    // off to another origin after signing in.
    router.push(redirect && redirect.startsWith('/') && !redirect.startsWith('//')
      ? redirect
      : { name: 'Dashboard' })
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.auth-page {
  min-height:100vh;
  display:flex;
  align-items:center;
  justify-content:center;
  background:var(--grad-surface) 100%);
  padding:2rem;
}

.auth-container {
  width:100%;
  max-width:420px;
}

.auth-card {
  padding:2.5rem;
}

.auth-header {
  text-align:center;
  margin-bottom:2rem;
}

.auth-logo {
  display:flex;
  align-items:center;
  justify-content:center;
  gap:0.75rem;
  color:var(--primary-600);
  margin-bottom:1.5rem;
}

.auth-logo h1 {
  font-size:1.5rem;
  font-weight:700;
}

.auth-header h2 {
  font-size:1.5rem;
  font-weight:600;
  margin-bottom:0.5rem;
}

.auth-header p {
  color:var(--text-secondary);
  font-size:0.875rem;
}

.auth-form {
  margin-bottom:1.5rem;
}

.form-checkbox {
  display:flex;
  align-items:center;
  gap:0.5rem;
  cursor:pointer;
  font-size:0.875rem;
}

.form-checkbox input[type="checkbox"] {
  cursor:pointer;
}

.btn-block {
  width:100%;
}

.auth-footer {
  text-align:center;
  padding-top:1.5rem;
  border-top:1px solid var(--border-color);
}

.auth-footer p {
  color:var(--text-secondary);
  font-size:0.875rem;
}

.auth-link {
  color:var(--primary-600);
  text-decoration:none;
  font-weight:500;
}

.auth-link:hover {
  text-decoration:underline;
}
</style>

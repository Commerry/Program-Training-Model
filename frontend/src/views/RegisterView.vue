<template>
  <div class="auth-page">
    <div class="auth-container">
      <div class="auth-card card">
        <div class="auth-header">
          <div class="auth-logo">
            <Icon name="box" size="lg" />
            <h1>Vision Training</h1>
          </div>
          <h2>Create Account</h2>
          <p>Sign up to start training your models</p>
        </div>

        <form @submit.prevent="handleRegister" class="auth-form">
          <div v-if="error" class="error-message">
            {{ error }}
          </div>

          <div class="form-group">
            <label class="form-label">Username *</label>
            <input
              v-model="formData.username"
              type="text"
              class="form-input"
              placeholder="Choose a username"
              required
              autofocus
            />
          </div>

          <div class="form-group">
            <label class="form-label">Email *</label>
            <input
              v-model="formData.email"
              type="email"
              class="form-input"
              placeholder="your.email@example.com"
              required
            />
          </div>

          <div class="form-group">
            <label class="form-label">Full Name</label>
            <input
              v-model="formData.fullName"
              type="text"
              class="form-input"
              placeholder="Your full name"
            />
          </div>

          <div class="form-group">
            <label class="form-label">Password *</label>
            <input
              v-model="formData.password"
              type="password"
              class="form-input"
              placeholder="Create a password"
              required
              minlength="8"
            />
            <p class="form-hint">At least 8 characters</p>
          </div>

          <div class="form-group">
            <label class="form-label">Confirm Password *</label>
            <input
              v-model="formData.confirmPassword"
              type="password"
              class="form-input"
              placeholder="Confirm your password"
              required
            />
          </div>

          <button type="submit" class="btn btn-primary btn-block" :disabled="loading">
            <span v-if="loading">Creating account...</span>
            <span v-else>Create Account</span>
          </button>
        </form>

        <div class="auth-footer">
          <p>
            Already have an account?
            <router-link to="/login" class="auth-link">Sign in</router-link>
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import Icon from '@/components/Icon.vue'
import { useAuthStore } from '@/stores/authStore'

// Matches the backend's minimum, so the form cannot accept something the
// server will reject.
const MIN_PASSWORD_LENGTH = 8

const router = useRouter()
const authStore = useAuthStore()

const formData = ref({
  username: '',
  email: '',
  fullName: '',
  password: '',
  confirmPassword: ''
})

const loading = ref(false)
const error = ref('')

const handleRegister = async () => {
  error.value = ''

  if (formData.value.password !== formData.value.confirmPassword) {
    error.value = 'Passwords do not match'
    return
  }
  if (formData.value.password.length < MIN_PASSWORD_LENGTH) {
    error.value = `Password must be at least ${MIN_PASSWORD_LENGTH} characters`
    return
  }

  loading.value = true
  try {
    await authStore.register({
      username: formData.value.username.trim(),
      email: formData.value.email.trim(),
      password: formData.value.password,
      full_name: formData.value.fullName.trim()
    })
    router.push({ name: 'Dashboard' })
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

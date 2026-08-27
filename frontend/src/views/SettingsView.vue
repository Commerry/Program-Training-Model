<template>
  <div class="settings-view">
    <div class="settings-container">
      <!-- Profile -->
      <div class="settings-section card">
        <div class="section-header">
          <h2>Profile</h2>
          <p>Your account details</p>
        </div>

        <div class="profile-avatar-section">
          <div class="avatar-preview">
            <Icon name="user" size="xl" />
          </div>
          <div class="avatar-actions">
            <strong>{{ authStore.displayName || authStore.user?.username }}</strong>
            <p class="avatar-hint">
              {{ authStore.user?.is_admin ? 'Administrator' : 'Member' }}
              &middot; joined {{ formatDate(authStore.user?.created_at) }}
            </p>
          </div>
        </div>

        <form @submit.prevent="updateProfile" class="settings-form">
          <div class="form-row">
            <div class="form-group">
              <label class="form-label">Username</label>
              <input
                :value="authStore.user?.username"
                type="text"
                class="form-input"
                disabled
              />
              <p class="form-hint">Usernames cannot be changed.</p>
            </div>

            <div class="form-group">
              <label class="form-label">Email *</label>
              <input
                v-model="profileForm.email"
                type="email"
                class="form-input"
                placeholder="your.email@example.com"
                required
              />
            </div>
          </div>

          <div class="form-group">
            <label class="form-label">Full name</label>
            <input
              v-model="profileForm.fullName"
              type="text"
              class="form-input"
              placeholder="Your full name"
            />
          </div>

          <p v-if="profileError" class="error-message">{{ profileError }}</p>
          <p v-if="profileNotice" class="success-message">{{ profileNotice }}</p>

          <div class="form-actions">
            <button type="button" class="btn btn-secondary" @click="resetProfileForm">
              Cancel
            </button>
            <button type="submit" class="btn btn-primary" :disabled="savingProfile">
              <Icon name="check" size="sm" />
              {{ savingProfile ? 'Saving...' : 'Save changes' }}
            </button>
          </div>
        </form>
      </div>

      <!-- Password -->
      <div class="settings-section card">
        <div class="section-header">
          <h2>Change password</h2>
          <p>Use a password you do not use anywhere else</p>
        </div>

        <form @submit.prevent="changePassword" class="settings-form">
          <div class="form-group">
            <label class="form-label">Current password *</label>
            <input
              v-model="passwordForm.currentPassword"
              type="password"
              class="form-input"
              autocomplete="current-password"
              required
            />
          </div>

          <div class="form-group">
            <label class="form-label">New password *</label>
            <input
              v-model="passwordForm.newPassword"
              type="password"
              class="form-input"
              autocomplete="new-password"
              required
              minlength="8"
            />
            <p class="form-hint">At least 8 characters.</p>
          </div>

          <div class="form-group">
            <label class="form-label">Confirm new password *</label>
            <input
              v-model="passwordForm.confirmPassword"
              type="password"
              class="form-input"
              autocomplete="new-password"
              required
            />
          </div>

          <p v-if="passwordError" class="error-message">{{ passwordError }}</p>
          <p v-if="passwordNotice" class="success-message">{{ passwordNotice }}</p>

          <div class="form-actions">
            <button type="button" class="btn btn-secondary" @click="resetPasswordForm">
              Cancel
            </button>
            <button type="submit" class="btn btn-primary" :disabled="savingPassword">
              <Icon name="check" size="sm" />
              {{ savingPassword ? 'Updating...' : 'Update password' }}
            </button>
          </div>
        </form>
      </div>

      <!-- Preferences: stored in this browser only -->
      <div class="settings-section card">
        <div class="section-header">
          <h2>Preferences</h2>
          <p>Saved in this browser</p>
        </div>

        <div class="preferences-list">
          <div class="preference-item">
            <div class="preference-info">
              <h4>Auto-save annotations</h4>
              <p>Save a box as soon as you finish drawing it</p>
            </div>
            <label class="toggle-switch">
              <input v-model="preferences.autoSave" type="checkbox" />
              <span class="toggle-slider"></span>
            </label>
          </div>

          <div class="preference-item">
            <div class="preference-info">
              <h4>Confirm before leaving unsaved work</h4>
              <p>Warn when navigating away from unsaved annotations</p>
            </div>
            <label class="toggle-switch">
              <input v-model="preferences.warnUnsaved" type="checkbox" />
              <span class="toggle-slider"></span>
            </label>
          </div>
        </div>

        <div class="form-actions">
          <button class="btn btn-primary" @click="savePreferences">
            <Icon name="check" size="sm" />
            Save preferences
          </button>
          <span v-if="preferencesNotice" class="success-message inline">
            {{ preferencesNotice }}
          </span>
        </div>
      </div>

      <!-- Server -->
      <div class="settings-section card">
        <div class="section-header">
          <h2>Server</h2>
          <p>Where this instance keeps its data</p>
        </div>

        <dl class="server-info">
          <div>
            <dt>API version</dt>
            <dd>{{ health.version || '-' }}</dd>
          </div>
          <div>
            <dt>Projects directory</dt>
            <dd class="text-mono">{{ health.projects_root || '-' }}</dd>
          </div>
          <div>
            <dt>Authentication</dt>
            <dd>{{ health.auth_required ? 'Required' : 'Disabled' }}</dd>
          </div>
        </dl>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import Icon from '@/components/Icon.vue'
import { errorMessage, http } from '@/services'
import { useAuthStore } from '@/stores/authStore'
import { formatDate } from '@/utils/format'

const PREFERENCES_KEY = 'vision-training.preferences'

const authStore = useAuthStore()

const profileForm = ref({ email: '', fullName: '' })
const passwordForm = ref({ currentPassword: '', newPassword: '', confirmPassword: '' })
const preferences = ref({ autoSave: false, warnUnsaved: true })
const health = ref({})

const savingProfile = ref(false)
const savingPassword = ref(false)
const profileError = ref(null)
const profileNotice = ref(null)
const passwordError = ref(null)
const passwordNotice = ref(null)
const preferencesNotice = ref(null)

const resetProfileForm = () => {
  profileForm.value = {
    email: authStore.user?.email || '',
    fullName: authStore.user?.full_name || ''
  }
  profileError.value = null
  profileNotice.value = null
}

const resetPasswordForm = () => {
  passwordForm.value = { currentPassword: '', newPassword: '', confirmPassword: '' }
  passwordError.value = null
  passwordNotice.value = null
}

onMounted(async () => {
  resetProfileForm()

  // Preferences are per-browser, not part of the account, so they live in
  // localStorage rather than being sent to a server that has nowhere to put
  // them. Reading is guarded because private-mode browsers can throw here.
  try {
    const stored = window.localStorage.getItem(PREFERENCES_KEY)
    if (stored) preferences.value = { ...preferences.value, ...JSON.parse(stored) }
  } catch { /* keep the defaults */ }

  try {
    health.value = (await http.get('/health')).data
  } catch { /* the panel simply shows dashes */ }
})

const updateProfile = async () => {
  savingProfile.value = true
  profileError.value = null
  profileNotice.value = null
  try {
    await authStore.updateProfile({
      email: profileForm.value.email,
      full_name: profileForm.value.fullName
    })
    profileNotice.value = 'Profile updated.'
  } catch (error) {
    profileError.value = error.message
  } finally {
    savingProfile.value = false
  }
}

const changePassword = async () => {
  passwordError.value = null
  passwordNotice.value = null

  if (passwordForm.value.newPassword !== passwordForm.value.confirmPassword) {
    passwordError.value = 'The new passwords do not match.'
    return
  }
  if (passwordForm.value.newPassword.length < 8) {
    passwordError.value = 'The new password must be at least 8 characters.'
    return
  }

  savingPassword.value = true
  try {
    // Sends the real request; the previous version only showed a success
    // dialog and never changed anything.
    await authStore.updateProfile({
      current_password: passwordForm.value.currentPassword,
      new_password: passwordForm.value.newPassword
    })
    resetPasswordForm()
    passwordNotice.value = 'Password updated.'
  } catch (error) {
    passwordError.value = error.message
  } finally {
    savingPassword.value = false
  }
}

const savePreferences = () => {
  try {
    window.localStorage.setItem(PREFERENCES_KEY, JSON.stringify(preferences.value))
    preferencesNotice.value = 'Saved.'
  } catch (error) {
    preferencesNotice.value = errorMessage(error, 'Could not save preferences')
  }
  window.setTimeout(() => { preferencesNotice.value = null }, 4000)
}
</script>

<style scoped>
.settings-view {
  min-height:100vh;
}

.page-header {
  background:var(--surface);
  border-bottom:1px solid var(--border-color);
  padding:0.375rem 0;
  margin-bottom:0.5rem;
}

.header-content {
  max-width:900px;
  margin:0 auto;
  padding:0 2rem;
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
}

.header-text {
  flex:1;
}

.page-title {
  font-size:1rem;
  font-weight:600;
  margin:0;
  color:var(--text-primary);
  letter-spacing:-0.02em;
}

.page-subtitle {
  font-size:0.9375rem;
  margin:0;
  color:var(--text-secondary);
  font-weight:400;
}

.settings-container {
  max-width:900px;
  margin:0 auto;
  padding:0 2rem 2rem 2rem;
  display:flex;
  flex-direction:column;
  gap:2rem;
}

.settings-section {
  padding:2rem;
}

.section-header {
  margin-bottom:2rem;
  padding-bottom:1rem;
  border-bottom:2px solid var(--border-color);
}

.section-header h2 {
  font-size:1.5rem;
  font-weight:600;
  margin-bottom:0.5rem;
}

.section-header p {
  color:var(--text-secondary);
  font-size:0.875rem;
}

.profile-avatar-section {
  display:flex;
  align-items:center;
  gap:2rem;
  margin-bottom:2rem;
  padding:1.5rem;
  background:var(--gray-50);
  border-radius:var(--radius-lg);
}

.avatar-preview {
  width:100px;
  height:100px;
  border-radius:50%;
  background:linear-gradient(135deg, var(--primary-100), var(--primary-200));
  color:var(--primary-600);
  display:flex;
  align-items:center;
  justify-content:center;
  flex-shrink:0;
}

.avatar-hint {
  font-size:0.75rem;
  color:var(--text-secondary);
  margin-top:0.5rem;
}

.settings-form {
  display:flex;
  flex-direction:column;
  gap:1.5rem;
}

.form-row {
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:1.5rem;
}

.form-actions {
  display:flex;
  justify-content:flex-end;
  gap:1rem;
  margin-top:1rem;
}

.preferences-list {
  display:flex;
  flex-direction:column;
  gap:1.5rem;
  margin-bottom:2rem;
}

.preference-item {
  display:flex;
  justify-content:space-between;
  align-items:center;
  padding:1.5rem;
  background:var(--gray-50);
  border-radius:var(--radius-lg);
  gap:1rem;
}

.preference-info h4 {
  font-weight:600;
  margin-bottom:0.25rem;
}

.preference-info p {
  font-size:0.875rem;
  color:var(--text-secondary);
}

/* Toggle Switch */
.toggle-switch {
  position:relative;
  display:inline-block;
  width:52px;
  height:28px;
  flex-shrink:0;
}

.toggle-switch input {
  opacity:0;
  width:0;
  height:0;
}

.toggle-slider {
  position:absolute;
  cursor:pointer;
  top:0;
  left:0;
  right:0;
  bottom:0;
  background-color:var(--gray-300);
  transition:.3s;
  border-radius:34px;
}

.toggle-slider:before {
  position:absolute;
  content:"";
  height:20px;
  width:20px;
  left:4px;
  bottom:4px;
  background-color:var(--surface);
  transition:.3s;
  border-radius:50%;
}

.toggle-switch input:checked + .toggle-slider {
  background-color:var(--primary-500);
}

.toggle-switch input:checked + .toggle-slider:before {
  transform:translateX(24px);
}

/* Danger Zone */
.danger-zone {
  border:2px solid var(--danger-200);
  background:var(--danger-50);
}

.danger-zone .section-header {
  border-bottom-color:var(--danger-200);
}

.danger-zone .section-header h2 {
  color:var(--danger-700);
}

.danger-item {
  display:flex;
  justify-content:space-between;
  align-items:center;
  padding:1.5rem;
  background:var(--surface);
  border-radius:var(--radius-lg);
  gap:1rem;
}

.danger-info h4 {
  font-weight:600;
  margin-bottom:0.25rem;
  color:var(--danger-700);
}

.danger-info p {
  font-size:0.875rem;
  color:var(--danger-600);
}

/* Responsive */
@media (max-width:768px) {
  .header-content h1 {
    font-size:2rem;
  }
  
  .form-row {
    grid-template-columns:1fr;
  }
  
  .profile-avatar-section {
    flex-direction:column;
    text-align:center;
  }
  
  .preference-item,
  .danger-item {
    flex-direction:column;
    align-items:flex-start;
    gap:1rem;
  }
  
  .form-actions {
    flex-direction:column;
  }
  
  .form-actions button {
    width:100%;
  }
}
</style>

<style scoped>
.server-info {
  display:grid;
  gap:0.75rem;
  margin:0;
}

.server-info > div {
  display:grid;
  grid-template-columns:minmax(140px, 200px) 1fr;
  gap:1rem;
  align-items:baseline;
}

.server-info dt {
  font-size:0.8125rem;
  color:var(--text-secondary);
}

.server-info dd {
  margin:0;
  font-size:0.8125rem;
  color:var(--text-primary);
  overflow-wrap:anywhere;
}

.success-message.inline {
  margin:0 0 0 0.75rem;
  padding:0;
  background:none;
}

.form-input:disabled {
  background:var(--gray-100);
  color:var(--text-tertiary);
  cursor:not-allowed;
}

@media (max-width:640px) {
  .server-info > div {
    grid-template-columns:1fr;
    gap:0.125rem;
  }
}
</style>

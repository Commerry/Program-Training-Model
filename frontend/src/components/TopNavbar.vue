<template>
  <div class="top-navbar">
    <div class="navbar-container">
      <!-- Left: Brand & Search -->
      <div class="navbar-left">
        <div class="navbar-brand">
          <Icon name="box" size="lg" />
          <span class="brand-text">Vision Training</span>
        </div>
        
        <form class="navbar-search" @submit.prevent="submitSearch">
          <Icon name="search" size="sm" />
          <input
            type="search"
            placeholder="Search projects..."
            class="search-input"
            v-model="searchQuery"
          />
        </form>
      </div>

      <!-- Right: Actions & User -->
      <div class="navbar-right">
        <!-- Quick Actions -->
        <button class="navbar-icon-btn" title="New project" @click="newProject">
          <Icon name="plus" size="sm" />
        </button>

        <!-- User Menu -->
        <div class="navbar-user" @click="toggleUserMenu">
          <div class="user-avatar">
            <Icon name="user" size="sm" />
          </div>
          <div class="user-info" v-if="authStore.user">
            <div class="user-name">{{ authStore.user.full_name || authStore.user.username }}</div>
            <div class="user-role">{{ authStore.user.is_admin ? 'Administrator' : 'Member' }}</div>
          </div>
          <Icon name="chevron-down" size="sm" class="dropdown-icon" />

          <!-- Dropdown Menu -->
          <div class="user-dropdown" v-if="showUserMenu">
            <router-link to="/settings" class="dropdown-item">
              <Icon name="settings" size="sm" />
              <span>Settings</span>
            </router-link>
            <div class="dropdown-divider"></div>
            <button @click="handleLogout" class="dropdown-item logout">
              <Icon name="logout" size="sm" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/authStore'
import Icon from './Icon.vue'

const router = useRouter()
const authStore = useAuthStore()

const searchQuery = ref('')
const showUserMenu = ref(false)

const toggleUserMenu = () => { showUserMenu.value = !showUserMenu.value }

/** Projects page reads ?q= and filters its list. */
const submitSearch = () => {
  router.push({ name: 'Projects', query: searchQuery.value ? { q: searchQuery.value } : {} })
}

/** Projects page opens its create dialog when ?new=1 is present. */
const newProject = () => {
  router.push({ name: 'Projects', query: { new: '1' } })
}

const handleLogout = async () => {
  showUserMenu.value = false
  await authStore.logout()
  router.push({ name: 'Login' })
}

// Registered on mount and removed on unmount; the previous version added a
// listener at module scope that was never cleaned up.
const closeOnOutsideClick = (event) => {
  if (!event.target.closest('.navbar-user')) showUserMenu.value = false
}
onMounted(() => window.addEventListener('click', closeOnOutsideClick))
onBeforeUnmount(() => window.removeEventListener('click', closeOnOutsideClick))
</script>

<style scoped>
/* ──────────────────────────────────────────────────────────────────────────
   Fixed top bar. Translucent with a blur so content scrolling underneath is
   suggested rather than hidden behind a solid slab.
   ────────────────────────────────────────────────────────────────────────── */

.top-navbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 100;
  height: var(--navbar-h);
  background: rgba(14, 16, 23, 0.82);
  backdrop-filter: blur(14px) saturate(140%);
  border-bottom: 1px solid var(--border);
}

.navbar-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
  height: 100%;
  padding: 0 1.25rem;
}

.navbar-left {
  display: flex;
  align-items: center;
  gap: 1.25rem;
  min-width: 0;
  flex: 1;
}

.navbar-right {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* ── Brand ─────────────────────────────────────────────────────────────── */

.navbar-brand {
  display: flex;
  align-items: center;
  gap: 0.625rem;
  flex-shrink: 0;
  color: var(--text);
}

.navbar-brand :deep(svg) {
  color: var(--accent-hover);
}

.brand-text {
  font-size: var(--fs-md);
  font-weight: 650;
  letter-spacing: -0.02em;
  background: var(--grad-accent);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

/* ── Search ────────────────────────────────────────────────────────────── */

.navbar-search {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
  max-width: 420px;
  height: 34px;
  padding: 0 0.75rem;
  border: 1px solid var(--border-strong);
  border-radius: var(--r);
  background: var(--bg);
  color: var(--text-3);
  transition: border-color var(--t-fast), box-shadow var(--t-fast);
}

.navbar-search:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-ring);
}

.search-input {
  flex: 1;
  min-width: 0;
  border: none;
  background: transparent;
  color: var(--text);
  font-family: inherit;
  font-size: var(--fs-base);
  outline: none;
}

.search-input::placeholder { color: var(--text-3); }

/* Chrome's clear button is invisible against a dark field. */
.search-input::-webkit-search-cancel-button {
  -webkit-appearance: none;
  height: 12px;
  width: 12px;
  background: var(--text-3);
  mask: url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12'%3E%3Cpath d='M1 1l10 10M11 1L1 11' stroke='black' stroke-width='2'/%3E%3C/svg%3E") center/contain no-repeat;
  cursor: pointer;
}

/* ── Actions ───────────────────────────────────────────────────────────── */

.navbar-icon-btn {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border: 1px solid var(--border-strong);
  border-radius: var(--r);
  background: var(--surface-2);
  color: var(--text-2);
  cursor: pointer;
  transition: all var(--t-fast);
}

.navbar-icon-btn:hover {
  background: var(--surface-hover);
  border-color: var(--accent-muted);
  color: var(--text);
}

/* ── User menu ─────────────────────────────────────────────────────────── */

.navbar-user {
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.625rem;
  height: 38px;
  padding: 0 0.5rem 0 0.375rem;
  border: 1px solid transparent;
  border-radius: var(--r);
  cursor: pointer;
  transition: background var(--t-fast), border-color var(--t-fast);
}

.navbar-user:hover {
  background: var(--surface-2);
  border-color: var(--border);
}

.user-avatar {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: var(--r-full);
  background: var(--grad-accent-2);
  color: #fff;
  flex-shrink: 0;
}

.user-info { min-width: 0; line-height: 1.25; }

.user-name {
  font-size: var(--fs-sm);
  font-weight: 550;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 150px;
}

.user-role {
  font-size: var(--fs-xs);
  color: var(--text-3);
}

.dropdown-icon {
  color: var(--text-3);
  flex-shrink: 0;
}

.user-dropdown {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  z-index: 10;
  min-width: 190px;
  padding: 0.3125rem;
  background: var(--surface-2);
  border: 1px solid var(--border-strong);
  border-radius: var(--r-md);
  box-shadow: var(--shadow-lg);
  animation: dropdown-in 160ms var(--ease-out);
}

@keyframes dropdown-in {
  from { opacity: 0; transform: translateY(-6px); }
  to   { opacity: 1; transform: translateY(0); }
}

.dropdown-item {
  display: flex;
  align-items: center;
  gap: 0.625rem;
  width: 100%;
  padding: 0.5rem 0.625rem;
  border: none;
  border-radius: var(--r-sm);
  background: transparent;
  color: var(--text-2);
  font-family: inherit;
  font-size: var(--fs-base);
  text-align: left;
  text-decoration: none;
  cursor: pointer;
  transition: background var(--t-fast), color var(--t-fast);
}

.dropdown-item:hover {
  background: var(--surface-3);
  color: var(--text);
}

.dropdown-item.logout:hover {
  background: var(--danger-soft);
  color: var(--danger);
}

.dropdown-divider {
  height: 1px;
  margin: 0.3125rem 0;
  background: var(--border);
}

@media (max-width: 900px) {
  .navbar-search { display: none; }
}

@media (max-width: 640px) {
  .user-info { display: none; }
  .brand-text { display: none; }
}
</style>

<template>
  <div class="sidebar">
    <nav class="sidebar-nav">
      <router-link to="/dashboard" class="nav-item" active-class="active">
        <div class="nav-icon"><Icon name="layout" size="sm" /></div>
        <span class="nav-label">Dashboard</span>
      </router-link>

      <router-link to="/projects" class="nav-item" active-class="active">
        <div class="nav-icon"><Icon name="folder" size="sm" /></div>
        <span class="nav-label">Projects</span>
        <span v-if="projectCount > 0" class="nav-badge">{{ projectCount }}</span>
      </router-link>

      <router-link to="/datasets" class="nav-item" active-class="active">
        <div class="nav-icon"><Icon name="database" size="sm" /></div>
        <span class="nav-label">Datasets</span>
      </router-link>

      <router-link to="/models" class="nav-item" active-class="active">
        <div class="nav-icon"><Icon name="box" size="sm" /></div>
        <span class="nav-label">Models</span>
      </router-link>

      <router-link to="/history" class="nav-item" active-class="active">
        <div class="nav-icon"><Icon name="clock" size="sm" /></div>
        <span class="nav-label">History</span>
      </router-link>

      <router-link to="/analytics" class="nav-item" active-class="active">
        <div class="nav-icon"><Icon name="chart-bar" size="sm" /></div>
        <span class="nav-label">Analytics</span>
      </router-link>

      <router-link to="/test-model" class="nav-item" active-class="active">
        <div class="nav-icon"><Icon name="rocket" size="sm" /></div>
        <span class="nav-label">Model Test</span>
      </router-link>
    </nav>

    <div class="sidebar-bottom">
      <router-link to="/settings" class="nav-item" active-class="active">
        <div class="nav-icon"><Icon name="settings" size="sm" /></div>
        <span class="nav-label">Settings</span>
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import Icon from './Icon.vue'

const projectStore = useProjectStore()
const projectCount = computed(() => projectStore.projects?.length || 0)
</script>

<style scoped>
/* ──────────────────────────────────────────────────────────────────────────
   Rail that widens on hover. Collapsed it shows icons only; the labels fade
   in as it opens rather than appearing instantly, so the transition reads as
   one motion instead of two.
   ────────────────────────────────────────────────────────────────────────── */

.sidebar {
  position: fixed;
  top: var(--navbar-h);
  left: 0;
  z-index: 90;
  display: flex;
  flex-direction: column;
  width: var(--sidebar-w);
  height: calc(100vh - var(--navbar-h));
  background: var(--bg-subtle);
  border-right: 1px solid var(--border);
  overflow: hidden;
  transition: width var(--t-slow);
}

.sidebar::after {
  /* Faint accent wash so the rail separates from the page without a hard
     colour block. */
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, var(--accent-softer), transparent 40%);
  pointer-events: none;
}

.sidebar:hover { width: var(--sidebar-w-open); }

.sidebar-nav {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 0.75rem 0.5rem;
  overflow-x: hidden;
  overflow-y: auto;
}

.sidebar-bottom {
  padding: 0.5rem 0.5rem 0.875rem;
  border-top: 1px solid var(--border-subtle);
}

/* ── Items ─────────────────────────────────────────────────────────────── */

.nav-item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  height: 42px;
  padding: 0 0.625rem;
  border-radius: var(--r);
  color: var(--text-2);
  text-decoration: none;
  white-space: nowrap;
  transition: background var(--t-fast), color var(--t-fast);
}

.nav-item:hover {
  background: var(--surface-2);
  color: var(--text);
}

.nav-item.active {
  background: var(--accent-soft);
  color: var(--text);
}

.nav-item.active::before {
  /* Accent bar on the active item, anchored to the rail edge so it stays put
     while the sidebar widens. */
  content: '';
  position: absolute;
  left: -0.5rem;
  top: 50%;
  transform: translateY(-50%);
  width: 3px;
  height: 22px;
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  background: var(--grad-accent);
}

.nav-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 24px;
  width: 24px;
  height: 24px;
}

.nav-item.active .nav-icon { color: var(--accent-hover); }

.nav-label {
  flex: 1;
  font-size: var(--fs-base);
  font-weight: 500;
  opacity: 0;
  transform: translateX(-4px);
  transition: opacity var(--t) 40ms, transform var(--t) 40ms;
}

.sidebar:hover .nav-label {
  opacity: 1;
  transform: translateX(0);
}

.nav-badge {
  min-width: 20px;
  padding: 0.0625rem 0.375rem;
  border-radius: var(--r-full);
  background: var(--accent);
  color: #fff;
  font-size: var(--fs-xs);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  text-align: center;
  opacity: 0;
  transition: opacity var(--t) 40ms;
}

.sidebar:hover .nav-badge { opacity: 1; }

@media (max-width: 768px) {
  .sidebar { display: none; }
}
</style>

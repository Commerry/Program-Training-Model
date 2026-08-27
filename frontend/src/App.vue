<template>
  <div id="app">
    <template v-if="showChrome">
      <TopNavbar />
      <Sidebar
        @mouseenter="sidebarExpanded = true"
        @mouseleave="sidebarExpanded = false"
      />
    </template>

    <main
      class="app-main"
      :class="{ 'with-chrome': showChrome }"
      :style="showChrome ? { marginLeft: sidebarExpanded ? 'var(--sidebar-w-open)' : 'var(--sidebar-w)' } : null"
    >
      <RouterView v-if="auth.ready" v-slot="{ Component }">
        <component :is="Component" />
      </RouterView>
      <div v-else class="app-booting">
        <span class="spinner" />
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { RouterView } from 'vue-router'
import Sidebar from '@/components/Sidebar.vue'
import TopNavbar from '@/components/TopNavbar.vue'
import { useAuthStore } from '@/stores/authStore'

const auth = useAuthStore()
const sidebarExpanded = ref(false)

// Chrome is hidden until the session check finishes, so a signed-in user
// refreshing a page never sees the layout flash from signed-out to signed-in.
const showChrome = computed(() => auth.ready && auth.isAuthenticated)
</script>

<style scoped>
#app {
  min-height:100vh;
}

.app-main {
  min-height:100vh;
  background: transparent;
  overflow-y:auto;
}

.app-main.with-chrome {
  padding-top: var(--navbar-h);
  transition:margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.app-booting {
  display:flex;
  align-items:center;
  justify-content:center;
  min-height:100vh;
}

@media (max-width:768px) {
  .app-main {
    margin-left:0 !important;
  }
}
</style>

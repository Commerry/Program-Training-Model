<template>
  <div class="not-found">
    <h1>404</h1>
    <p>That page does not exist.</p>
    <RouterLink class="btn btn-primary" :to="{ name: 'Dashboard' }">
      Back to dashboard
    </RouterLink>
  </div>
</template>

<script setup>
import { RouterLink } from 'vue-router'
</script>

<style scoped>
.not-found {
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  gap:1rem;
  min-height:60vh;
  color:var(--text-secondary);
}

.not-found h1 {
  font-size:4rem;
  font-weight:700;
  color:var(--text-primary);
  margin:0;
}
</style>

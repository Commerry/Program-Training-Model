<template>
  <div class="history-view">
    <!-- Summary -->
    <div class="stats-row">
      <div class="stat-box card">
        <div class="stat-icon success"><Icon name="check" size="lg" /></div>
        <div class="stat-info">
          <div class="stat-value">{{ counts.completed }}</div>
          <div class="stat-label">Completed</div>
        </div>
      </div>

      <div class="stat-box card">
        <div class="stat-icon warning"><Icon name="clock" size="lg" /></div>
        <div class="stat-info">
          <div class="stat-value">{{ counts.stopped }}</div>
          <div class="stat-label">Stopped</div>
        </div>
      </div>

      <div class="stat-box card">
        <div class="stat-icon danger"><Icon name="x" size="lg" /></div>
        <div class="stat-info">
          <div class="stat-value">{{ counts.failed }}</div>
          <div class="stat-label">Failed</div>
        </div>
      </div>

      <div class="stat-box card">
        <div class="stat-icon primary"><Icon name="zap" size="lg" /></div>
        <div class="stat-info">
          <div class="stat-value">{{ runs.length }}</div>
          <div class="stat-label">Total runs</div>
        </div>
      </div>
    </div>

    <!-- Runs -->
    <div class="history-container">
      <div class="table-header">
        <h2>Training runs</h2>
        <div class="header-actions">
          <div class="search-box">
            <Icon name="search" size="sm" />
            <input v-model="searchQuery" type="search" placeholder="Search..." />
          </div>
          <select v-model="filterStatus" class="filter-select">
            <option value="all">All status</option>
            <option value="completed">Completed</option>
            <option value="stopped">Stopped</option>
            <option value="failed">Failed</option>
          </select>
          <button class="btn-icon" title="Refresh" :disabled="loading" @click="load">
            <Icon name="refresh" size="sm" />
          </button>
        </div>
      </div>

      <div v-if="error" class="empty-state">
        <Icon name="x" size="xl" />
        <h3>Could not load history</h3>
        <p>{{ error }}</p>
        <button class="btn btn-primary" @click="load">Try again</button>
      </div>

      <div v-else-if="loading" class="empty-state">
        <p>Loading…</p>
      </div>

      <div v-else-if="filteredRuns.length === 0" class="empty-state">
        <Icon name="clock" size="xl" />
        <h3>{{ runs.length ? 'No runs match this filter' : 'No training history yet' }}</h3>
        <p>{{ runs.length ? 'Try a different search or status.' : 'Train a model and it will appear here.' }}</p>
        <router-link v-if="!runs.length" to="/projects" class="btn btn-primary">
          <Icon name="zap" size="sm" />
          Go to projects
        </router-link>
      </div>

      <div v-else class="table-wrapper">
        <table class="history-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Project</th>
              <th>Status</th>
              <th>mAP@50</th>
              <th>Epochs</th>
              <th>Duration</th>
              <th>Started</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="run in filteredRuns" :key="run.key" class="table-row">
              <td class="model-cell">
                <div class="model-info">
                  <Icon name="box" size="sm" />
                  <div class="model-text">
                    <span class="model-name">{{ run.model_name }}</span>
                    <span class="model-type">{{ run.model_type }}</span>
                  </div>
                </div>
              </td>
              <td>{{ run.project_name }}</td>
              <td>
                <span class="badge" :class="'badge-' + statusVariant(run.status)">
                  {{ run.status }}
                </span>
              </td>
              <td class="accuracy-cell">
                <div v-if="run.map50 !== null" class="accuracy-bar">
                  <div class="accuracy-fill" :style="{ width: metricPercent(run.map50) + '%' }"></div>
                  <span class="accuracy-text">{{ formatMetric(run.map50) }}</span>
                </div>
                <span v-else class="muted">—</span>
              </td>
              <td>{{ run.completed_epochs ?? '—' }} / {{ run.epochs ?? '—' }}</td>
              <td>{{ formatDuration(run.started_at, run.completed_at) }}</td>
              <td>{{ formatDateTime(run.started_at) }}</td>
              <td>
                <div class="action-buttons">
                  <router-link
                    class="btn-icon"
                    title="Open training page"
                    :to="{ name: 'Train', params: { name: run.project_name } }"
                  >
                    <Icon name="search" size="sm" />
                  </router-link>
                  <a
                    v-if="run.best_model"
                    class="btn-icon"
                    title="Download weights"
                    :href="downloadUrl(run)"
                  >
                    <Icon name="download" size="sm" />
                  </a>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <p v-if="lastFailure" class="failure-note">
        Last failure — <strong>{{ lastFailure.model_name }}</strong>: {{ lastFailure.error }}
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import Icon from '@/components/Icon.vue'
import { errorMessage, trainingService } from '@/services'
import {
  formatDateTime, formatDuration, formatMetric, metricPercent, statusVariant
} from '@/utils/format'

const runs = ref([])
const loading = ref(true)
const error = ref(null)
const searchQuery = ref('')
const filterStatus = ref('all')

const load = async () => {
  loading.value = true
  error.value = null
  try {
    const { history } = await trainingService.allHistory()
    runs.value = history.map((run, index) => ({
      ...run,
      // Runs have no server-assigned id, so the list key is built from the
      // fields that together identify one.
      key: run.project_name + '/' + run.model_name + '/' + (run.started_at || index),
      map50: run.metrics?.mAP50 ?? null
    }))
  } catch (e) {
    error.value = errorMessage(e)
  } finally {
    loading.value = false
  }
}

onMounted(load)

const counts = computed(() => ({
  completed: runs.value.filter((r) => r.status === 'completed').length,
  stopped: runs.value.filter((r) => r.status === 'stopped').length,
  failed: runs.value.filter((r) => r.status === 'failed').length
}))

const filteredRuns = computed(() => {
  const query = searchQuery.value.trim().toLowerCase()
  return runs.value.filter((run) => {
    if (filterStatus.value !== 'all' && run.status !== filterStatus.value) return false
    if (!query) return true
    return [run.model_name, run.project_name, run.model_type]
      .filter(Boolean)
      .some((field) => String(field).toLowerCase().includes(query))
  })
})

const lastFailure = computed(() =>
  runs.value.find((run) => run.status === 'failed' && run.error) || null
)

const downloadUrl = (run) =>
  trainingService.downloadUrl(run.project_name, run.best_model)
</script>

<style scoped>
.history-view {
  min-height:100vh;
}

.page-header {
  background:var(--surface);
  border-bottom:1px solid var(--border-color);
  padding:0.375rem 0;
  margin-bottom:0.5rem;
}

.header-content {
  max-width:1400px;
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

.stats-row {
  max-width:1400px;
  margin:0 auto 3rem auto;
  padding:0 2rem;
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));
  gap:1.5rem;
}

.stat-box {
  padding:1.5rem;
  display:flex;
  align-items:center;
  gap:1rem;
  transition:all 0.3s;
}

.stat-box:hover {
  transform:translateY(-4px);
  box-shadow: var(--shadow);
}

.stat-icon {
  width:56px;
  height:56px;
  border-radius:var(--radius-lg);
  display:flex;
  align-items:center;
  justify-content:center;
}

.stat-icon.success {
  background:var(--success-100);
  color:var(--success-600);
}

.stat-icon.warning {
  background:var(--warning-100);
  color:var(--warning-600);
}

.stat-icon.danger {
  background:var(--danger-100);
  color:var(--danger-600);
}

.stat-icon.primary {
  background:var(--primary-100);
  color:var(--primary-600);
}

.stat-value {
  font-size:2rem;
  font-weight:700;
}

.stat-label {
  color:var(--text-secondary);
  font-size:0.875rem;
}

.history-container {
  max-width:1400px;
  margin:0 auto;
  padding:0 2rem 2rem 2rem;
}

.table-header {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:1.5rem;
  flex-wrap:wrap;
  gap:1rem;
}

.table-header h2 {
  font-size:1.5rem;
  font-weight:600;
}

.header-actions {
  display:flex;
  gap:1rem;
  flex-wrap:wrap;
}

.search-box {
  display:flex;
  align-items:center;
  gap:0.5rem;
  padding:0.5rem 1rem;
  background:var(--bg);
  border:1px solid var(--border-color);
  border-radius:var(--radius-md);
  transition:all 0.2s;
}

.search-box:focus-within {
  border-color:var(--primary-500);
  box-shadow: var(--shadow);
}

.search-box input {
  border:none;
  outline:none;
  font-size:0.875rem;
  width:200px;
}

.filter-select {
  padding:0.5rem 1rem;
  border:1px solid var(--border-color);
  border-radius:var(--radius-md);
  background:var(--surface);
  font-size:0.875rem;
  cursor:pointer;
}

.empty-state {
  text-align:center;
  padding:5rem 2rem;
  background:var(--surface);
  border-radius:var(--radius-lg);
  border:2px dashed var(--border-color);
}

.empty-state .icon {
  color:var(--text-tertiary);
  margin-bottom:1.5rem;
}

.empty-state h3 {
  font-size:1.5rem;
  font-weight:600;
  margin-bottom:0.5rem;
}

.empty-state p {
  color:var(--text-secondary);
  margin-bottom:2rem;
}

.table-wrapper {
  background:var(--surface);
  border-radius:var(--radius-lg);
  overflow:hidden;
  box-shadow: var(--shadow);
}

.history-table {
  width:100%;
  border-collapse:collapse;
}

.history-table thead {
  background:var(--gray-50);
}

.history-table th {
  padding:1rem;
  text-align:left;
  font-weight:600;
  font-size:0.875rem;
  color:var(--text-secondary);
  border-bottom:1px solid var(--border-color);
}

.table-row {
  transition:all 0.2s;
}

.table-row:hover {
  background:var(--gray-50);
}

.history-table td {
  padding:1rem;
  border-bottom:1px solid var(--border-color);
}

.model-info {
  display:flex;
  align-items:center;
  gap:0.5rem;
}

.model-name {
  font-weight:500;
}

.accuracy-cell {
  min-width:150px;
}

.accuracy-bar {
  position:relative;
  width:100%;
  height:28px;
  background:var(--bg-subtle);
  border-radius:var(--radius-md);
  overflow:hidden;
}

.accuracy-fill {
  position:absolute;
  left:0;
  top:0;
  height:100%;
  background:linear-gradient(90deg, var(--success-500), var(--success-400));
  transition:width 0.5s ease;
}

.accuracy-text {
  position:relative;
  display:block;
  text-align:center;
  line-height:28px;
  font-weight:600;
  font-size:0.875rem;
  color:var(--text-primary);
  z-index:1;
}

.action-buttons {
  display:flex;
  gap:0.5rem;
}

/* Responsive */
@media (max-width:1024px) {
  .table-wrapper {
    overflow-x:auto;
  }
  
  .history-table {
    min-width:900px;
  }
}

@media (max-width:768px) {
  .header-content h1 {
    font-size:2rem;
  }
  
  .stats-row {
    grid-template-columns:repeat(2, 1fr);
  }
  
  .table-header {
    flex-direction:column;
    align-items:flex-start;
  }
  
  .header-actions {
    width:100%;
    flex-direction:column;
  }
  
  .search-box input {
    width:100%;
  }
  
  .filter-select {
    width:100%;
  }
}
</style>

<style scoped>
.model-text {
  display:flex;
  flex-direction:column;
  gap:0.125rem;
}

.model-type {
  font-size:0.75rem;
  color:var(--text-tertiary);
}

.failure-note {
  margin:0;
  padding:0.75rem 1.5rem;
  border-top:1px solid var(--border-color);
  font-size:0.8125rem;
  color:var(--danger);
  background:var(--danger-soft);
}

.header-actions .btn-icon:disabled {
  opacity:0.5;
  cursor:not-allowed;
}
</style>

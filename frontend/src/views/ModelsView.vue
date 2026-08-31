<template>
  <div class="models-view">

    <!-- Toolbar -->
    <div class="page-topbar">
      <div class="topbar-left">
        <h2 class="topbar-title">Trained Models</h2>
        <span class="topbar-count">{{ filteredModels.length }}</span>
      </div>
      <div class="topbar-right">
        <div class="search-box">
          <Icon name="search" size="sm" />
          <input v-model="searchQuery" type="text" placeholder="Search models..." class="search-input" />
        </div>
        <select v-model="filterStatus" class="filter-select">
          <option value="all">All Status</option>
          <option value="ready">Ready</option>
          <option value="training">Training</option>
          <option value="stopped">Stopped</option>
          <option value="failed">Failed</option>
        </select>
        <select v-model="sortBy" class="filter-select">
          <option value="date">Latest first</option>
          <option value="accuracy">Best mAP@50</option>
          <option value="name">Name</option>
        </select>
        <button class="btn-icon" title="Refresh" :disabled="loading" @click="load">
          <Icon name="refresh" size="sm" />
        </button>
        <router-link to="/projects" class="btn btn-primary">
          <Icon name="zap" size="sm" />
          Start Training
        </router-link>
      </div>
    </div>

    <!-- Models Grid -->
    <div class="models-container">
      <div v-if="error" class="empty-state">
        <Icon name="x" size="xl" />
        <h3>Could not load models</h3>
        <p>{{ error }}</p>
        <button class="btn btn-primary" @click="load">Try again</button>
      </div>

      <div v-else-if="loading" class="empty-state">
        <Icon name="clock" size="xl" />
        <h3>Loading models…</h3>
      </div>

      <div v-else-if="allModels.length === 0" class="empty-state">
        <Icon name="box" size="xl" />
        <h3>No Trained Models Yet</h3>
        <p>Train a YOLO model from a project to see it here</p>
        <router-link to="/projects" class="btn btn-primary">
          <Icon name="zap" size="sm" />
          Go to Projects
        </router-link>
      </div>

      <div v-else-if="filteredModels.length === 0" class="empty-state">
        <Icon name="search" size="xl" />
        <h3>No Results</h3>
        <p>No models match the current filters</p>
      </div>

      <div v-else class="models-grid">
        <div v-for="model in filteredModels" :key="model.key" class="model-card card">
          <div class="model-header">
            <div class="model-icon" :class="`status-${model.status}`">
              <Icon name="box" size="lg" />
            </div>
            <span class="badge" :class="badgeClass(model.status)">
              {{ model.status }}
            </span>
          </div>

          <h3 class="model-name" :title="model.name">{{ model.name }}</h3>
          <p class="model-project">
            <Icon name="folder" size="xs" />
            {{ model.project }}
            <span v-if="model.modelType" class="model-arch">· {{ model.modelType }}</span>
          </p>

          <div class="model-metrics">
            <div class="metric">
              <Icon name="image" size="sm" />
              <span class="metric-label">Train / val:</span>
              <span class="metric-value">
                {{ formatNumber(model.trainImages) }} / {{ formatNumber(model.valImages) }}
              </span>
            </div>
            <div class="metric">
              <Icon name="layers" size="sm" />
              <span class="metric-label">Classes:</span>
              <span class="metric-value">{{ model.classes }}</span>
            </div>
            <div class="metric">
              <Icon name="clock" size="sm" />
              <span class="metric-label">Epochs:</span>
              <span class="metric-value">{{ model.epochs || '—' }} / {{ model.totalEpochs || '—' }}</span>
            </div>
            <div v-if="model.map50 !== null" class="metric">
              <Icon name="check" size="sm" />
              <span class="metric-label">mAP@50:</span>
              <span class="metric-value">{{ formatMetric(model.map50) }}</span>
            </div>
          </div>

          <div class="model-exports" v-if="model.formats.length">
            <span v-for="fmt in model.formats" :key="fmt" class="fmt-badge">{{ fmt }}</span>
          </div>

          <p v-if="model.error" class="model-error">{{ model.error }}</p>

          <div class="model-footer">
            <div class="model-date">
              <Icon name="clock" size="sm" />
              {{ formatDate(model.createdAt) }}
            </div>
            <div class="model-actions">
              <a v-if="model.downloadUrl" :href="model.downloadUrl"
                 class="btn-icon" title="Download weights" download>
                <Icon name="download" size="sm" />
              </a>
              <router-link :to="`/projects/${model.project}`" class="btn-icon" title="Open Project">
                <Icon name="folder" size="sm" />
              </router-link>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import Icon from '@/components/Icon.vue'
import { errorMessage, trainingService } from '@/services'
import { formatDate, formatMetric, formatNumber } from '@/utils/format'

const loading = ref(true)
const error = ref(null)
const filterStatus = ref('all')
const sortBy = ref('date')
const searchQuery = ref('')
const allModels = ref([])

const load = async () => {
  loading.value = true
  error.value = null
  try {
    // One request for every finished run across every project, plus one for
    // anything currently training. The previous version issued two requests
    // per project and could only ever show the most recent run of each.
    const [historyResult, overview] = await Promise.all([
      trainingService.allHistory(),
      trainingService.overview()
    ])

    const models = historyResult.history.map((run) => ({
      key: `${run.project_name}/${run.model_name}/${run.started_at || ''}`,
      name: run.model_name,
      project: run.project_name,
      modelType: run.model_type,
      status: { completed: 'ready', failed: 'failed', stopped: 'stopped' }[run.status] || run.status,
      trainImages: run.train_images,
      valImages: run.val_images,
      classes: (run.classes || []).length,
      epochs: run.completed_epochs || run.epochs,
      totalEpochs: run.epochs,
      map50: run.metrics?.mAP50 ?? null,
      createdAt: run.completed_at || run.started_at,
      error: run.error,
      formats: Object.entries(run.exported_models || {})
        .filter(([, path]) => path)
        .map(([format]) => format.toUpperCase()),
      downloadUrl: run.best_model
        ? trainingService.downloadUrl(run.project_name, run.best_model)
        : null
    }))

    // Runs still in progress are not in history yet, so they are added here.
    for (const active of overview.active_runs || []) {
      models.unshift({
        key: `active/${active.project_name}`,
        name: active.model_name || 'Training…',
        project: active.project_name,
        modelType: '',
        status: 'training',
        classes: 0,
        epochs: active.current_epoch,
        totalEpochs: active.total_epochs,
        map50: null,
        createdAt: null,
        formats: [],
        downloadUrl: null
      })
    }

    allModels.value = models
  } catch (e) {
    error.value = errorMessage(e)
  } finally {
    loading.value = false
  }
}

onMounted(load)

const badgeClass = (status) => ({
  ready: 'badge-success',
  training: 'badge-warning',
  stopped: 'badge-neutral',
  failed: 'badge-danger'
}[status] || 'badge-neutral')

const filteredModels = computed(() => {
  let result = allModels.value

  if (filterStatus.value !== 'all') {
    result = result.filter((model) => model.status === filterStatus.value)
  }

  const query = searchQuery.value.trim().toLowerCase()
  if (query) {
    result = result.filter((model) =>
      [model.name, model.project, model.modelType]
        .filter(Boolean)
        .some((field) => field.toLowerCase().includes(query))
    )
  }

  return [...result].sort((a, b) => {
    if (sortBy.value === 'name') return a.name.localeCompare(b.name)
    if (sortBy.value === 'accuracy') return (b.map50 ?? -1) - (a.map50 ?? -1)
    return String(b.createdAt || '').localeCompare(String(a.createdAt || ''))
  })
})
</script>

<style scoped>
.models-view {
  min-height:100vh;
  background:var(--grad-surface) 100%);
}

/* ── Toolbar ── */
.page-topbar {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:1rem;
  padding:0.75rem 2rem;
  background:var(--surface);
  border-bottom:1px solid var(--border-color);
  position:sticky;
  top:0;
  z-index:50;
  flex-wrap:wrap;
}

.topbar-left { display:flex; align-items:center; gap:0.625rem; }

.topbar-title { font-size:1.125rem; font-weight:600; margin:0; color:var(--text-primary); }

.topbar-count {
  display:inline-flex; align-items:center; justify-content:center;
  min-width:24px; height:24px; padding:0 6px;
  background:var(--primary-100); color:var(--primary-700);
  border-radius:9999px; font-size:0.8125rem; font-weight:600;
}

.topbar-right { display:flex; align-items:center; gap:0.625rem; flex-wrap:wrap; }

.search-box {
  display:flex; align-items:center; gap:0.5rem;
  padding:0.4375rem 0.875rem;
  background:var(--bg); border:1px solid var(--border-color);
  border-radius:10px; color:var(--text-secondary); transition:border-color 0.2s;
}

.search-box:focus-within { border-color:var(--primary-400); background:var(--surface); }

.search-input {
  border:none; background:transparent; outline:none;
  font-size:0.875rem; color:var(--text-primary); width:150px;
}

.filter-select {
  padding:0.5rem 0.875rem;
  border:1px solid var(--border-color);
  border-radius:10px;
  background:var(--gray-50);
  font-size:0.875rem;
  cursor:pointer;
  transition:border-color 0.2s;
}

.filter-select:focus { outline:none; border-color:var(--primary-400); }

/* ── Container ── */
.models-container {
  max-width:1400px;
  margin:0 auto;
  padding:2rem;
}

.empty-state {
  text-align:center;
  padding:5rem 2rem;
  background:var(--surface);
  border-radius:16px;
  border:2px dashed var(--border-color);
  color:var(--text-secondary);
}

.empty-state h3 { font-size:1.4rem; font-weight:600; margin:1rem 0 0.5rem; color:var(--text-primary); }
.empty-state p { margin-bottom:2rem; }

/* ── Grid ── */
.models-grid {
  display:grid;
  grid-template-columns:repeat(auto-fill, minmax(320px, 1fr));
  gap:1.5rem;
}

.model-card {
  padding:1.5rem;
  background:var(--surface);
  border-radius:16px;
  border:1px solid var(--border);
  box-shadow: var(--shadow-sm);
  transition:all 0.3s ease;
}

.model-card:hover { transform:translateY(-4px); box-shadow: var(--shadow-lg); }

.model-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem; }

.model-icon {
  width:52px; height:52px; border-radius:14px;
  display:flex; align-items:center; justify-content:center;
}

.model-icon.status-ready { background:var(--grad-success); color:var(--success); }
.model-icon.status-training { background:linear-gradient(135deg, var(--warning), var(--amber)); color:var(--warning); }
.model-icon.status-stopped {
  background: linear-gradient(135deg, var(--surface-3), var(--surface-hover));
  color: var(--text-2);
}

.model-icon.status-failed { background:var(--grad-danger); color:var(--danger); }

.badge {
  padding:0.25rem 0.75rem; border-radius:9999px;
  font-size:0.75rem; font-weight:600;
}

.badge-success { background:var(--success-soft); color:var(--success); }
.badge-warning { background:var(--warning-soft); color:var(--warning); }
.badge-danger { background:var(--danger-soft); color:var(--danger); }

.model-name { font-size:1rem; font-weight:600; margin:0 0 0.25rem; color:var(--text-primary); }

.model-project {
  display:flex; align-items:center; gap:0.3rem;
  color:var(--text-secondary); font-size:0.875rem; margin-bottom:1rem;
}

.model-metrics {
  background:var(--gray-50); border-radius:10px;
  padding:0.875rem; margin-bottom:0.75rem;
  display:flex; flex-direction:column; gap:0.5rem;
}

.metric {
  display:flex; align-items:center; gap:0.5rem;
  font-size:0.8125rem;
}

.metric-label { color:var(--text-secondary); }
.metric-value { margin-left:auto; font-weight:600; color:var(--text-primary); }

.model-exports { display:flex; gap:0.25rem; flex-wrap:wrap; margin-bottom:0.75rem; }

.fmt-badge {
  padding:0.15rem 0.5rem;
  background:var(--accent-soft); color:var(--accent);
  border-radius:6px; font-size:0.75rem; font-weight:600;
}

.model-footer {
  display:flex; justify-content:space-between; align-items:center;
  padding-top:1rem; border-top:1px solid var(--border-color);
}

.model-date { display:flex; align-items:center; gap:0.375rem; font-size:0.8125rem; color:var(--text-secondary); }
.model-actions { display:flex; gap:0.5rem; }

.btn-icon {
  width:34px; height:34px; border:none;
  background:var(--gray-100); color:var(--gray-600);
  border-radius:8px; cursor:pointer;
  display:flex; align-items:center; justify-content:center;
  text-decoration:none; transition:all 0.2s;
}

.btn-icon:hover { background:var(--primary-100); color:var(--primary-600); }

.btn {
  display:inline-flex; align-items:center; gap:0.375rem;
  padding:0.5rem 1.125rem; border:none; border-radius:10px;
  font-size:0.875rem; font-weight:600; cursor:pointer;
  text-decoration:none; transition:all 0.2s;
}

.btn-primary { background:var(--primary-600); color:var(--text); }
.btn-primary:hover { background:var(--primary-700); }
</style>

<style scoped>
.model-arch {
  color:var(--text-tertiary);
}

.model-error {
  margin:0 0 0.75rem;
  padding:0.5rem 0.625rem;
  border-radius:6px;
  background:var(--danger-soft);
  color:var(--danger);
  font-size:0.75rem;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}
</style>

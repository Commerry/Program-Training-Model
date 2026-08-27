<template>
  <div class="analytics-view">
    <div class="content-wrapper">
      <div v-if="error" class="chart-card">
        <h3 class="chart-title">Could not load analytics</h3>
        <p class="stat-label">{{ error }}</p>
        <button class="btn btn-primary" @click="load">Try again</button>
      </div>

      <template v-else>
        <!-- Stats -->
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-icon" style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);">
              <Icon name="folder" size="lg" />
            </div>
            <div class="stat-content">
              <div class="stat-label">Projects</div>
              <div class="stat-value">{{ overview.project_count ?? 0 }}</div>
              <div class="stat-change neutral">
                {{ formatNumber(overview.total_images) }} images
              </div>
            </div>
          </div>

          <div class="stat-card">
            <div class="stat-icon" style="background: linear-gradient(135deg, #10b981 0%, #34d399 100%);">
              <Icon name="check" size="lg" />
            </div>
            <div class="stat-content">
              <div class="stat-label">Annotated images</div>
              <div class="stat-value">{{ formatNumber(overview.annotated_images) }}</div>
              <div class="stat-change" :class="annotationCoverage >= 80 ? 'positive' : 'neutral'">
                {{ annotationCoverage }}% of all images
              </div>
            </div>
          </div>

          <div class="stat-card">
            <div class="stat-icon" style="background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);">
              <Icon name="chart-bar" size="lg" />
            </div>
            <div class="stat-content">
              <div class="stat-label">Average mAP@50</div>
              <div class="stat-value">{{ formatMetric(overview.average_map50) }}</div>
              <div class="stat-change neutral">
                across {{ overview.completed_runs ?? 0 }} completed runs
              </div>
            </div>
          </div>

          <div class="stat-card">
            <div class="stat-icon" style="background: linear-gradient(135deg, #ec4899 0%, #f472b6 100%);">
              <Icon name="zap" size="lg" />
            </div>
            <div class="stat-content">
              <div class="stat-label">Training runs</div>
              <div class="stat-value">{{ overview.training_runs ?? 0 }}</div>
              <div class="stat-change" :class="overview.failed_runs ? 'negative' : 'positive'">
                {{ overview.failed_runs ?? 0 }} failed
              </div>
            </div>
          </div>
        </div>

        <!-- Active runs -->
        <div v-if="overview.active_runs?.length" class="chart-card active-runs">
          <h3 class="chart-title">Running now</h3>
          <div v-for="run in overview.active_runs" :key="run.project_name" class="active-run">
            <span class="active-run-name">
              {{ run.project_name }} · {{ run.model_name }}
            </span>
            <div class="active-run-bar">
              <div
                class="active-run-fill"
                :style="{ width: epochPercent(run) + '%' }"
              ></div>
            </div>
            <span class="active-run-epoch">
              epoch {{ run.current_epoch ?? 0 }} / {{ run.total_epochs ?? '?' }}
            </span>
          </div>
        </div>

        <!-- Charts -->
        <div class="charts-grid">
          <div class="chart-card">
            <div class="chart-header">
              <h3 class="chart-title">Runs completed per day</h3>
              <div class="chart-legend">
                <span class="legend-item">
                  <span class="legend-dot" style="background: var(--primary-500);"></span>
                  Last {{ activityDays.length }} days
                </span>
              </div>
            </div>
            <div v-if="maxRunsPerDay === 0" class="chart-empty">No runs yet</div>
            <div v-else class="chart-placeholder">
              <div class="chart-bar-group">
                <div
                  v-for="day in activityDays"
                  :key="day.label"
                  class="chart-bar"
                  :style="{ height: (day.count ? Math.max(6, (day.count / maxRunsPerDay) * 100) : 2) + '%' }"
                  :class="{ 'is-empty': !day.count }"
                  :title="`${day.label}: ${day.count} run(s)`"
                ></div>
              </div>
            </div>
          </div>

          <div class="chart-card">
            <div class="chart-header">
              <h3 class="chart-title">mAP@50 over time</h3>
              <div class="chart-legend">
                <span class="legend-item">
                  <span class="legend-dot" style="background: #16a34a;"></span>
                  Completed runs
                </span>
              </div>
            </div>
            <div v-if="accuracySeries.length === 0" class="chart-empty">
              No completed runs with metrics yet
            </div>
            <div v-else-if="accuracySeries.length === 1" class="chart-single">
              <span class="chart-single-value text-gradient">
                {{ formatMetric(accuracySeries[0].value) }}
              </span>
              <span class="chart-single-label">
                one completed run so far — a trend needs at least two
              </span>
            </div>
            <div v-else class="chart-placeholder">
              <div class="line-chart">
                <svg width="100%" height="100%" viewBox="0 0 400 200" preserveAspectRatio="none">
                  <line
                    v-for="tick in [0, 50, 100, 150, 200]"
                    :key="tick"
                    x1="0" :y1="tick" x2="400" :y2="tick"
                    stroke="var(--border-color)" stroke-width="1"
                  />
                  <polyline
                    fill="none" stroke="#16a34a" stroke-width="2"
                    :points="accuracyPoints"
                  />
                  <circle
                    v-for="(point, index) in accuracyCoords"
                    :key="index"
                    :cx="point.x" :cy="point.y" r="3" fill="#16a34a"
                  >
                    <title>{{ point.label }}</title>
                  </circle>
                </svg>
              </div>
            </div>
          </div>
        </div>

        <!-- Recent runs -->
        <div class="recent-sessions">
          <div class="section-header">
            <h2 class="section-title">Recent training runs</h2>
            <router-link to="/history" class="link-view-all">
              View all
              <Icon name="arrow-right" size="xs" />
            </router-link>
          </div>

          <div v-if="!overview.recent_runs?.length" class="chart-empty">
            Nothing to show yet.
          </div>
          <div v-else class="sessions-table">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Project</th>
                  <th>mAP@50</th>
                  <th>Duration</th>
                  <th>Status</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="run in overview.recent_runs" :key="runKey(run)">
                  <td class="cell-model">
                    <Icon name="box" size="sm" />
                    <span>{{ run.model_name }}</span>
                  </td>
                  <td>{{ run.project_name }}</td>
                  <td class="cell-accuracy">{{ formatMetric(run.metrics?.mAP50) }}</td>
                  <td>{{ formatDuration(run.started_at, run.completed_at) }}</td>
                  <td>
                    <span class="status-badge" :class="run.status">{{ run.status }}</span>
                  </td>
                  <td class="cell-date">{{ formatDate(run.completed_at || run.started_at) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import Icon from '@/components/Icon.vue'
import { errorMessage, trainingService } from '@/services'
import {
  formatDate, formatDuration, formatMetric, formatNumber
} from '@/utils/format'

const ACTIVITY_DAYS = 14

const overview = ref({})
const history = ref([])
const error = ref(null)

const load = async () => {
  error.value = null
  try {
    const [overviewResult, historyResult] = await Promise.all([
      trainingService.overview(),
      trainingService.allHistory()
    ])
    overview.value = overviewResult
    history.value = historyResult.history
  } catch (e) {
    error.value = errorMessage(e)
  }
}

onMounted(load)

const runKey = (run) =>
  `${run.project_name}/${run.model_name}/${run.started_at || ''}`

const annotationCoverage = computed(() => {
  const total = overview.value.total_images || 0
  if (!total) return 0
  return Math.round(((overview.value.annotated_images || 0) / total) * 100)
})

const epochPercent = (run) => {
  if (!run.total_epochs) return 0
  return Math.min(100, Math.round(((run.current_epoch || 0) / run.total_epochs) * 100))
}

/** One bucket per day for the last ACTIVITY_DAYS days, oldest first. */
const activityDays = computed(() => {
  const buckets = new Map()
  const today = new Date()
  today.setHours(0, 0, 0, 0)

  for (let offset = ACTIVITY_DAYS - 1; offset >= 0; offset -= 1) {
    const day = new Date(today)
    day.setDate(day.getDate() - offset)
    buckets.set(day.toDateString(), { label: formatDate(day), count: 0 })
  }

  for (const run of history.value) {
    const stamp = run.completed_at || run.started_at
    if (!stamp) continue
    const day = new Date(stamp)
    if (Number.isNaN(day.getTime())) continue
    day.setHours(0, 0, 0, 0)
    const bucket = buckets.get(day.toDateString())
    if (bucket) bucket.count += 1
  }

  return [...buckets.values()]
})

const maxRunsPerDay = computed(() =>
  activityDays.value.reduce((max, day) => Math.max(max, day.count), 0)
)

/** Completed runs that reported a mAP50, oldest first. */
const accuracySeries = computed(() =>
  history.value
    .filter((run) => run.status === 'completed' && run.metrics?.mAP50 != null)
    .slice()
    .reverse()
    .map((run) => ({
      value: Number(run.metrics.mAP50),
      label: `${run.model_name}: ${formatMetric(run.metrics.mAP50)}`
    }))
)

/** SVG viewBox is 400x200 with y inverted, so 1.0 maps to y=0. */
const accuracyCoords = computed(() => {
  const series = accuracySeries.value
  if (!series.length) return []
  const step = series.length === 1 ? 0 : 400 / (series.length - 1)
  return series.map((point, index) => ({
    x: series.length === 1 ? 200 : index * step,
    y: 200 - Math.max(0, Math.min(1, point.value)) * 200,
    label: point.label
  }))
})

const accuracyPoints = computed(() =>
  accuracyCoords.value.map((point) => `${point.x},${point.y}`).join(' ')
)
</script>

<style scoped>
.analytics-view {
  min-height:100vh;
  background:var(--grad-surface) 100%);
}

/* Simple Clean Header */
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
  font-size:0.875rem;
  margin:0;
  color:var(--text-secondary);
  font-weight:400;
}

/* Content Wrapper */
.content-wrapper {
  max-width:1400px;
  margin:0 auto;
  padding:0 2rem 2rem;
}

.header-actions {
  display:flex;
  gap:0.75rem;
  align-items:center;
}

.period-select {
  padding:0.5rem 1rem;
  border:1px solid var(--border-color);
  border-radius:8px;
  background:var(--surface);
  color:var(--text-primary);
  font-size:0.875rem;
  cursor:pointer;
  transition:all 0.2s;
}

.period-select:hover {
  border-color:var(--primary-400);
}

.period-select:focus {
  outline:none;
  border-color:var(--primary-500);
  box-shadow: var(--shadow);
}

.btn-export {
  display:flex;
  align-items:center;
  gap:0.5rem;
  padding:0.5rem 1rem;
  background:var(--surface);
  border:1px solid var(--border-color);
  border-radius:8px;
  color:var(--text-primary);
  font-size:0.875rem;
  font-weight:500;
  cursor:pointer;
  transition:all 0.2s;
}

.btn-export:hover {
  background:var(--gray-50);
  border-color:var(--primary-400);
  color:var(--primary-600);
  transform:translateY(-1px);
  box-shadow: var(--shadow);
}

.btn-export:active {
  transform:translateY(0) scale(0.98);
}

/* Stats Grid */
.stats-grid {
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(250px, 1fr));
  gap:1.5rem;
  margin-bottom:2rem;
}

.stat-card {
  background:var(--surface);
  border:1px solid var(--border-color);
  border-radius:12px;
  padding:1.5rem;
  display:flex;
  gap:1rem;
  transition:all 0.2s;
}

.stat-card:hover {
  transform:translateY(-2px);
  box-shadow: var(--shadow);
}

.stat-icon {
  width:56px;
  height:56px;
  border-radius:12px;
  display:flex;
  align-items:center;
  justify-content:center;
  color:var(--text);
  flex-shrink:0;
}

.stat-content {
  flex:1;
}

.stat-label {
  font-size:0.875rem;
  color:var(--text-secondary);
  margin-bottom:0.25rem;
}

.stat-value {
  font-size:1.875rem;
  font-weight:700;
  color:var(--text-primary);
  margin-bottom:0.25rem;
}

.stat-change {
  font-size:0.8125rem;
  font-weight:500;
}

.stat-change.positive {
  color:var(--success-600);
}

.stat-change.negative {
  color:var(--error-600);
}

.stat-change.neutral {
  color:var(--text-tertiary);
}

/* Charts Grid */
.charts-grid {
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(400px, 1fr));
  gap:1.5rem;
  margin-bottom:2rem;
}

.chart-card {
  background:var(--surface);
  border:1px solid var(--border-color);
  border-radius:12px;
  padding:1.5rem;
}

.chart-header {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:1.5rem;
}

.chart-title {
  font-size:1rem;
  font-weight:600;
  color:var(--text-primary);
  margin:0;
}

.chart-legend {
  display:flex;
  gap:1rem;
}

.legend-item {
  display:flex;
  align-items:center;
  gap:0.5rem;
  font-size:0.8125rem;
  color:var(--text-secondary);
}

.legend-dot {
  width:8px;
  height:8px;
  border-radius:50%;
}

.chart-placeholder {
  height:200px;
  display:flex;
  align-items:flex-end;
  justify-content:space-around;
  padding:1rem 0;
}

.chart-bar-group {
  display:flex;
  align-items:flex-end;
  justify-content:space-around;
  gap:1rem;
  width:100%;
  height:100%;
}

.chart-bar {
  flex:1;
  background:linear-gradient(180deg, var(--primary-500) 0%, var(--primary-600) 100%);
  border-radius:4px 4px 0 0;
  min-height:20px;
  transition:all 0.3s;
  cursor:pointer;
}

.chart-bar:hover {
  background:linear-gradient(180deg, var(--primary-400) 0%, var(--primary-500) 100%);
  transform:scaleY(1.05);
}

.line-chart {
  width:100%;
  height:100%;
}

/* Recent Sessions */
.recent-sessions {
  background:var(--surface);
  border:1px solid var(--border-color);
  border-radius:12px;
  padding:1.5rem;
}

.section-header {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:1.5rem;
}

.section-title {
  font-size:1rem;
  font-weight:600;
  color:var(--text-primary);
  margin:0;
}

.link-view-all {
  color:var(--primary-600);
  text-decoration:none;
  font-size:0.875rem;
  font-weight:500;
  transition:all 0.2s;
}

.link-view-all:hover {
  color:var(--primary-700);
  transform:translateX(4px);
}

.sessions-table {
  overflow-x:auto;
}

table {
  width:100%;
  border-collapse:collapse;
}

thead {
  background:var(--gray-50);
  border-radius:8px;
}

th {
  padding:0.75rem 1rem;
  text-align:left;
  font-size:0.8125rem;
  font-weight:600;
  color:var(--text-secondary);
  text-transform:uppercase;
  letter-spacing:0.05em;
}

tbody tr {
  border-bottom:1px solid var(--border-color);
  transition:all 0.2s;
}

tbody tr:hover {
  background:var(--gray-50);
}

td {
  padding:1rem;
  font-size:0.875rem;
  color:var(--text-primary);
}

.cell-model {
  display:flex;
  align-items:center;
  gap:0.5rem;
  font-weight:500;
}

.cell-accuracy {
  color:var(--success-600);
  font-weight:600;
}

.cell-date {
  color:var(--text-secondary);
}

.status-badge {
  display:inline-block;
  padding:0.25rem 0.75rem;
  border-radius:9999px;
  font-size:0.75rem;
  font-weight:600;
  text-transform:capitalize;
}

.status-badge.completed {
  background: var(--success-soft);
  color: var(--success);
  border: 1px solid rgba(52, 211, 153, 0.25);
}

.status-badge.in-progress,
.status-badge.running,
.status-badge.stopped {
  background: var(--warning-soft);
  color: var(--warning);
  border: 1px solid rgba(251, 191, 36, 0.25);
}

.status-badge.failed {
  background: var(--danger-soft);
  color: var(--danger);
  border: 1px solid rgba(248, 113, 113, 0.25);
}

/* Responsive */
@media (max-width:768px) {
  .analytics-view {
    padding:1rem;
  }

  .page-header {
    flex-direction:column;
    gap:1rem;
  }

  .header-actions {
    width:100%;
    flex-direction:column;
  }

  .period-select,
  .btn-export {
    width:100%;
    justify-content:center;
  }

  .stats-grid {
    grid-template-columns:1fr;
  }

  .charts-grid {
    grid-template-columns:1fr;
  }
}
</style>

<style scoped>
.link-view-all {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

.chart-single {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.375rem;
  min-height: 180px;
}

.chart-single-value {
  font-size: var(--fs-3xl);
  font-weight: 700;
  letter-spacing: -0.03em;
}

.chart-single-label {
  font-size: var(--fs-sm);
  color: var(--text-3);
}

.chart-bar.is-empty {
  background: var(--border);
  opacity: 0.7;
}

.chart-empty {
  display:flex;
  align-items:center;
  justify-content:center;
  min-height:180px;
  color:var(--text-tertiary);
  font-size:0.875rem;
}

.active-runs {
  margin-bottom:1.5rem;
}

.active-run {
  display:grid;
  grid-template-columns:minmax(0, 1fr) 2fr auto;
  align-items:center;
  gap:1rem;
  padding:0.5rem 0;
}

.active-run-name {
  font-size:0.875rem;
  font-weight:500;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}

.active-run-bar {
  height:8px;
  border-radius:9999px;
  background:var(--bg-subtle);
  overflow:hidden;
}

.active-run-fill {
  height:100%;
  background:var(--primary-500);
  transition:width 0.4s ease;
}

.active-run-epoch {
  font-size:0.75rem;
  color:var(--text-secondary);
  white-space:nowrap;
}

.chart-bar {
  transition:height 0.3s ease;
}
</style>

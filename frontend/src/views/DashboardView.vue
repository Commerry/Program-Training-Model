<template>
  <div class="dashboard">
    <div class="content-wrapper">
      <div class="stats-grid">
        <div class="stat-card card">
          <div class="stat-icon" style="background-color: var(--primary-100); color: var(--primary-600);">
            <Icon name="folder" size="lg" />
          </div>
          <div class="stat-content">
            <h3 class="stat-value">{{ overview.project_count ?? 0 }}</h3>
            <p class="stat-label">Projects</p>
          </div>
        </div>

        <div class="stat-card card">
          <div class="stat-icon" style="background-color: var(--success-100); color: var(--success-600);">
            <Icon name="image" size="lg" />
          </div>
          <div class="stat-content">
            <h3 class="stat-value">{{ formatNumber(overview.annotated_images) }}</h3>
            <p class="stat-label">
              Annotated images
              <span class="stat-sub">of {{ formatNumber(overview.total_images) }}</span>
            </p>
          </div>
        </div>

        <div class="stat-card card">
          <div class="stat-icon" style="background-color: var(--warning-100); color: var(--warning-600);">
            <Icon name="zap" size="lg" />
          </div>
          <div class="stat-content">
            <h3 class="stat-value">{{ overview.training_runs ?? 0 }}</h3>
            <p class="stat-label">
              Training runs
              <span class="stat-sub">{{ overview.completed_runs ?? 0 }} completed</span>
            </p>
          </div>
        </div>

        <div class="stat-card card">
          <div class="stat-icon" style="background-color: var(--info-100); color: var(--info-600);">
            <Icon name="chart-bar" size="lg" />
          </div>
          <div class="stat-content">
            <h3 class="stat-value">{{ formatMetric(overview.average_map50) }}</h3>
            <p class="stat-label">Average mAP@50</p>
          </div>
        </div>
      </div>

      <!-- A run in progress is the thing most worth surfacing first. -->
      <div v-if="overview.active_runs?.length" class="active-banner card">
        <Icon name="zap" size="sm" />
        <span>
          Training in progress:
          <strong v-for="(run, index) in overview.active_runs" :key="run.project_name">
            <template v-if="index > 0">, </template>{{ run.project_name }}
            (epoch {{ run.current_epoch ?? 0 }}/{{ run.total_epochs ?? '?' }})
          </strong>
        </span>
        <router-link
          class="btn btn-sm btn-primary"
          :to="{ name: 'Train', params: { name: overview.active_runs[0].project_name } }"
        >
          Open
        </router-link>
      </div>

      <p v-if="error" class="error-message">{{ error }}</p>

      <div class="dashboard-content">
        <div class="recent-projects">
          <div class="section-header">
            <h2 class="section-title">
              <Icon name="clock" size="sm" />
              Recent projects
            </h2>
            <router-link to="/projects" class="btn btn-secondary btn-sm">
              View all
            </router-link>
          </div>

          <div v-if="loading" class="empty-state">
            <p>Loading…</p>
          </div>

          <div v-else-if="projects.length === 0" class="empty-state">
            <Icon name="folder" size="xl" />
            <h3>No projects yet</h3>
            <p>Create your first project to get started</p>
            <router-link to="/projects" class="btn btn-primary">
              <Icon name="plus" size="sm" />
              Create project
            </router-link>
          </div>

          <div v-else class="projects-list">
            <div
              v-for="project in recentProjects"
              :key="project.name"
              class="project-item card"
              @click="openProject(project.name)"
            >
              <div class="project-info">
                <h3>{{ project.name }}</h3>
                <p class="project-desc">{{ project.description || 'No description' }}</p>
                <div class="project-meta">
                  <span class="badge badge-primary">
                    {{ Object.keys(project.tags || {}).length }} classes
                  </span>
                  <span class="badge badge-gray">
                    {{ formatNumber(project.total_images) }} images
                  </span>
                  <span class="badge" :class="readinessBadge(project)">
                    {{ project.annotated_images || 0 }} annotated
                  </span>
                </div>
              </div>
              <div class="project-arrow">
                <Icon name="chevron-right" />
              </div>
            </div>
          </div>
        </div>

        <div class="quick-actions">
          <h2 class="section-title">
            <Icon name="zap" size="sm" />
            Quick actions
          </h2>
          <div class="actions-grid">
            <router-link :to="{ name: 'Projects', query: { new: '1' } }" class="action-card card">
              <Icon name="plus" size="lg" />
              <h3>New project</h3>
              <p>Create a training project</p>
            </router-link>

            <router-link to="/datasets" class="action-card card">
              <Icon name="upload" size="lg" />
              <h3>Import dataset</h3>
              <p>Upload an exported dataset</p>
            </router-link>

            <router-link to="/test-model" class="action-card card">
              <Icon name="rocket" size="lg" />
              <h3>Test a model</h3>
              <p>Run a model on sample images</p>
            </router-link>

            <router-link to="/history" class="action-card card">
              <Icon name="clock" size="lg" />
              <h3>Training history</h3>
              <p>Review past runs</p>
            </router-link>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import Icon from '@/components/Icon.vue'
import { errorMessage, trainingService } from '@/services'
import { useProjectStore } from '@/stores/projectStore'
import { formatMetric, formatNumber } from '@/utils/format'

const router = useRouter()
const projectStore = useProjectStore()

const overview = ref({})
const loading = ref(true)
const error = ref(null)

const projects = computed(() => projectStore.projects)

const recentProjects = computed(() =>
  [...projects.value]
    .sort((a, b) => String(b.updated_at || b.created_at || '')
      .localeCompare(String(a.updated_at || a.created_at || '')))
    .slice(0, 5)
)

const readinessBadge = (project) => {
  const total = project.total_images || 0
  const annotated = project.annotated_images || 0
  if (!total) return 'badge-gray'
  if (annotated >= total) return 'badge-success'
  return annotated > 0 ? 'badge-warning' : 'badge-gray'
}

const openProject = (name) => {
  router.push({ name: 'ProjectDetail', params: { name } })
}

onMounted(async () => {
  loading.value = true
  try {
    // Project list and aggregates come from different endpoints; fetching them
    // together keeps the dashboard to a single round trip.
    const [, overviewResult] = await Promise.all([
      projectStore.fetchProjects(),
      trainingService.overview()
    ])
    overview.value = overviewResult
  } catch (e) {
    error.value = errorMessage(e)
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.dashboard {
  min-height:100vh;
  background:var(--grad-surface) 100%);
}

/* Compact Header */
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
  font-size:0.75rem;
  margin:0;
  color:var(--text-secondary);
  font-weight:400;
}

.page-subtitle strong {
  font-weight:600;
  color:var(--text-primary);
}

/* Content Wrapper */
.content-wrapper {
  max-width:1400px;
  margin:0 auto;
  padding:0 2rem 2rem;
}

/* Stats Grid */
.stats-grid {
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(260px, 1fr));
  gap:1.25rem;
  margin-bottom:2rem;
}

.stat-card {
  background:var(--surface);
  padding:1.75rem;
  border-radius:16px;
  display:flex;
  align-items:center;
  gap:1.25rem;
  transition:all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border:1px solid var(--border);
  box-shadow: var(--shadow-sm);
  position:relative;
}

.stat-card::after {
  content:'';
  position:absolute;
  inset:0;
  border-radius:16px;
  padding:2px;
  background:linear-gradient(135deg, var(--primary-500), var(--primary-300));
  -webkit-mask:linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite:xor;
  mask-composite:exclude;
  opacity:0;
  transition:opacity 0.3s ease;
}

.stat-card:hover {
  transform:translateY(-4px);
  box-shadow: var(--shadow-lg);
  border-color:var(--primary-200);
}

.stat-card:hover::after {
  opacity:1;
}

.stat-icon {
  width:64px;
  height:64px;
  border-radius:14px;
  display:flex;
  align-items:center;
  justify-content:center;
  flex-shrink:0;
  box-shadow: var(--shadow);
}

.stat-content {
  flex:1;
}

.stat-value {
  font-size:2rem;
  font-weight:700;
  margin:0 0 0.25rem 0;
  line-height:1;
  color:var(--text-primary);
}

.stat-label {
  color:var(--text-secondary);
  font-size:0.9375rem;
  margin:0;
  font-weight:500;
}

/* Dashboard Content */
.dashboard-content {
  display:grid;
  grid-template-columns:2fr 1fr;
  gap:1.5rem;
  margin-bottom:2rem;
}

.recent-projects,
.quick-actions {
  background:var(--surface);
  border-radius:16px;
  padding:1.75rem;
  border:1px solid var(--border);
  box-shadow: var(--shadow-sm);
}

.section-header {
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom:1.5rem;
  padding-bottom:1rem;
  border-bottom:1px solid var(--border-color);
}

.section-title {
  font-size:1.25rem;
  font-weight:600;
  margin:0;
  display:flex;
  align-items:center;
  gap:0.625rem;
  color:var(--text-primary);
}

.empty-state {
  text-align:center;
  padding:3rem 2rem;
  background:var(--grad-surface) 100%);
  border-radius:12px;
  border:2px dashed var(--border-color);
}

.empty-state .icon {
  color:var(--primary-300);
  margin-bottom:1.25rem;
}

.empty-state h3 {
  font-size:1.125rem;
  font-weight:600;
  margin:0 0 0.5rem 0;
  color:var(--text-primary);
}

.empty-state p {
  color:var(--text-secondary);
  margin:0 0 1.5rem 0;
  font-size:0.9375rem;
}

.projects-list {
  display:flex;
  flex-direction:column;
  gap:1rem;
}

.project-item {
  padding:1.25rem;
  border:1px solid var(--border-color);
  border-radius:12px;
  cursor:pointer;
  transition:all 0.2s ease;
  display:flex;
  align-items:center;
  justify-content:space-between;
  background:var(--bg-subtle);
}

.project-item:hover {
  border-color:var(--primary-300);
  background:var(--surface);
  transform:translateX(4px);
  box-shadow: var(--shadow);
}

.project-info {
  flex:1;
}

.project-info h3 {
  font-weight:600;
  color:var(--text-primary);
  margin:0 0 0.25rem 0;
  font-size:1rem;
}

.project-desc {
  font-size:0.875rem;
  color:var(--text-secondary);
  margin:0 0 0.5rem 0;
}

.project-meta {
  display:flex;
  gap:0.5rem;
  flex-wrap:wrap;
}

.project-arrow {
  color:var(--text-secondary);
  transition:transform 0.2s ease;
}

.project-item:hover .project-arrow {
  transform:translateX(4px);
  color:var(--primary-600);
}

.quick-actions {
  display:flex;
  flex-direction:column;
}

.quick-actions .section-title {
  margin-bottom:1.5rem;
}

.actions-grid {
  display:grid;
  grid-template-columns:repeat(2, 1fr);
  gap:1rem;
}

.action-card {
  background:var(--surface);
  padding:1.5rem;
  border-radius:12px;
  text-align:center;
  cursor:pointer;
  transition:all 0.2s ease;
  text-decoration:none;
  color:inherit;
  border:1px solid var(--border-color);
  display:flex;
  flex-direction:column;
  align-items:center;
  gap:0.75rem;
}

.action-card:hover {
  transform:translateY(-2px);
  box-shadow: var(--shadow);
  border-color:var(--primary-300);
  background:var(--grad-surface);
}

.action-card .icon {
  color:var(--primary-600);
}

.action-card h3 {
  font-size:1rem;
  font-weight:600;
  margin:0;
  color:var(--text-primary);
}

.action-card p {
  color:var(--text-secondary);
  font-size:0.875rem;
  margin:0;
}

/* Responsive Design */
@media (max-width:1200px) {
  .dashboard-content {
    grid-template-columns:1fr;
  }
  
  .stats-grid {
    grid-template-columns:repeat(2, 1fr);
  }
}

@media (max-width:1024px) {
  .stats-grid {
    grid-template-columns:repeat(2, 1fr);
  }
  
  .stat-card {
    padding:1.25rem;
  }

  .recent-projects,
  .quick-actions {
    padding:1.25rem;
  }
}

@media (max-width:768px) {
  .dashboard-content {
    grid-template-columns:1fr;
  }
  
  .stats-grid {
    grid-template-columns:1fr;
  }
  
  .actions-grid {
    grid-template-columns:1fr;
  }
  
  .header-content {
    flex-direction:column;
    gap:1rem;
  }
  
  .page-header {
    padding:1.5rem 0;
  }
  
  .content-wrapper {
    padding:0 1rem 1rem;
  }
}

@media (max-width:480px) {
  .page-title {
    font-size:1.5rem;
  }
  
  .stat-value {
    font-size:1.5rem;
  }
  
  .stat-icon {
    width:48px;
    height:48px;
  }
  
  .project-item {
    padding:1rem;
  }
  
  .action-card {
    padding:1.25rem;
  }
}
</style>

<style scoped>
.stat-sub {
  display:block;
  font-size:0.75rem;
  color:var(--text-tertiary);
  margin-top:0.125rem;
}

.active-banner {
  display:flex;
  align-items:center;
  gap:0.75rem;
  padding:0.75rem 1rem;
  margin-bottom:1.25rem;
  border-left:3px solid var(--primary-500);
  font-size:0.875rem;
}

.active-banner > span {
  flex:1;
}

.active-banner strong {
  font-weight:600;
}
</style>

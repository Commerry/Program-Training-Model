<template>
  <div class="projects-view">

    <!-- Toolbar -->
    <div class="page-topbar">
      <div class="topbar-left">
        <h2 class="topbar-title">Projects</h2>
        <span class="topbar-count">{{ store.projects.length }}</span>
      </div>
      <div class="topbar-right">
        <div class="search-box">
          <Icon name="search" size="sm" />
          <input v-model="searchQuery" type="text" placeholder="Search projects..." class="search-input" />
        </div>
        <button @click="openCreateModal" class="btn btn-primary">
          <Icon name="plus" size="sm" />
          New Project
        </button>
      </div>
    </div>

    <!-- Main Content -->
    <div class="content-wrapper">

    <div v-if="store.loading" class="loading">
      Loading projects...
    </div>

    <div v-else-if="store.error" class="error-message">
      {{ store.error }}
    </div>

    <div v-else-if="store.projects.length === 0" class="empty-state">
      <Icon name="folder" size="xl" />
      <h3>No Projects Yet</h3>
      <p>Create your first project to start training models</p>
      <button @click="openCreateModal" class="btn btn-primary">
        <Icon name="plus" size="sm" />
        Create Project
      </button>
    </div>

    <div v-else-if="filteredProjects.length === 0" class="empty-state">
      <Icon name="search" size="xl" />
      <h3>No Results</h3>
      <p>No projects match "{{ searchQuery }}"</p>
    </div>

    <div v-else class="projects-grid">
      <div
        v-for="project in filteredProjects"
        :key="project.name"
        class="project-card card"
        @click="goToProject(project.name)"
      >
        <div class="project-header">
          <div class="project-icon">
            <Icon name="folder" size="lg" />
          </div>
          <div class="project-actions" @click.stop>
            <button @click="deleteProjectHandler(project.name)" class="btn-icon" title="Delete">
              <Icon name="trash" size="sm" />
            </button>
          </div>
        </div>

        <h3 class="project-name">{{ project.name }}</h3>
        <p class="project-description">{{ project.description || 'No description' }}</p>
        
        <div class="project-stats">
          <div class="stat">
            <Icon name="image" size="sm" />
            <span class="stat-value">{{ project.total_images || 0 }}</span>
            <span class="stat-label">images</span>
          </div>
          <div class="stat">
            <Icon name="check" size="sm" />
            <span class="stat-value">{{ project.annotated_images || 0 }}</span>
            <span class="stat-label">annotated</span>
          </div>
          <div class="stat">
            <Icon name="box" size="sm" />
            <span class="stat-value">{{ project.total_annotations || 0 }}</span>
            <span class="stat-label">boxes</span>
          </div>
        </div>

        <div class="project-tags" v-if="project.tags && Object.keys(project.tags).length > 0">
          <span
            v-for="(stats, tag) in project.tags"
            :key="tag"
            class="badge badge-primary"
          >
            {{ tag }}: {{ stats.boxes }}
          </span>
        </div>
      </div>
    </div>

    <!-- Create Project Modal -->
    <div v-if="showCreateModal" class="modal-overlay" @click="showCreateModal = false">
      <div class="modal" @click.stop>
        <div class="modal-header">
          <h3 class="modal-title">Create new project</h3>
          <button @click="showCreateModal = false" class="modal-close">
            <Icon name="x" />
          </button>
        </div>
        
        <div class="modal-body">
          <form @submit.prevent="createProject">
            <div class="form-group">
              <label class="form-label">Project name *</label>
              <input
                v-model="newProject.name"
                type="text"
                class="form-input"
                placeholder="Enter project name"
                maxlength="64"
                autofocus
                required
              />
              <p class="form-hint">
                Letters, digits, spaces, dots, underscores and hyphens. This
                becomes the folder name on disk.
              </p>
            </div>
            
            <div class="form-group">
              <label class="form-label">Description</label>
              <textarea
                v-model="newProject.description"
                class="form-textarea"
                rows="3"
                placeholder="Enter project description"
              ></textarea>
            </div>

            <p v-if="createError" class="error-message">{{ createError }}</p>
          </form>
        </div>
        
        <div class="modal-footer">
          <button @click="showCreateModal = false" class="btn btn-secondary">
            Cancel
          </button>
          <button
            class="btn btn-primary"
            :disabled="creating || !newProject.name.trim()"
            @click="createProject"
          >
            <Icon name="plus" size="sm" />
            {{ creating ? 'Creating…' : 'Create project' }}
          </button>
        </div>
      </div>
    </div>
    </div><!-- End content-wrapper -->
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import Icon from '@/components/Icon.vue'
import { useProjectStore } from '@/stores/projectStore'

const route = useRoute()
const router = useRouter()
const store = useProjectStore()

const showCreateModal = ref(false)
const searchQuery = ref('')
const creating = ref(false)
const createError = ref(null)
const newProject = ref({ name: '', description: '' })

const filteredProjects = computed(() => {
  const query = searchQuery.value.trim().toLowerCase()
  if (!query) return store.projects
  return store.projects.filter((project) =>
    project.name.toLowerCase().includes(query) ||
    (project.description || '').toLowerCase().includes(query)
  )
})

// The navbar search box and its "+" button link here with query parameters,
// so this page has to honour them on load and on every later navigation.
const applyQuery = () => {
  searchQuery.value = String(route.query.q || '')
  if (route.query.new === '1') {
    showCreateModal.value = true
    router.replace({ name: 'Projects', query: { ...route.query, new: undefined } })
  }
}

onMounted(() => {
  store.fetchProjects()
  applyQuery()
})
watch(() => route.query, applyQuery)

const openCreateModal = () => {
  createError.value = null
  newProject.value = { name: '', description: '' }
  showCreateModal.value = true
}

const createProject = async () => {
  if (creating.value) return
  createError.value = null
  creating.value = true
  try {
    await store.createProject(newProject.value.name.trim(), newProject.value.description)
    showCreateModal.value = false
    newProject.value = { name: '', description: '' }
  } catch (error) {
    createError.value = error.message
  } finally {
    creating.value = false
  }
}

const goToProject = (projectName) => {
  router.push({ name: 'ProjectDetail', params: { name: projectName } })
}

const deleteProjectHandler = async (projectName) => {
  const confirmation = window.prompt(
    `Deleting "${projectName}" permanently removes its images, annotations and ` +
    'trained models. Type the project name to confirm:'
  )
  // A typed confirmation rather than a yes/no dialog: this deletes a dataset
  // that may represent days of annotation work.
  if (confirmation !== projectName) return
  try {
    await store.deleteProject(projectName)
  } catch (error) {
    window.alert('Could not delete project: ' + error.message)
  }
}
</script>

<style scoped>
.projects-view {
  min-height:100vh;
  background:var(--grad-surface) 100%);
}

/* โ”€โ”€ Toolbar โ”€โ”€ */
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
}

.topbar-left {
  display:flex;
  align-items:center;
  gap:0.625rem;
}

.topbar-title {
  font-size:1.125rem;
  font-weight:600;
  margin:0;
  color:var(--text-primary);
}

.topbar-count {
  display:inline-flex;
  align-items:center;
  justify-content:center;
  min-width:24px;
  height:24px;
  padding:0 6px;
  background:var(--primary-100);
  color:var(--primary-700);
  border-radius:9999px;
  font-size:0.8125rem;
  font-weight:600;
}

.topbar-right {
  display:flex;
  align-items:center;
  gap:0.75rem;
}

.search-box {
  display:flex;
  align-items:center;
  gap:0.5rem;
  padding:0.4375rem 0.875rem;
  background:var(--bg);
  border:1px solid var(--border-color);
  border-radius:10px;
  color:var(--text-secondary);
  transition:border-color 0.2s;
}

.search-box:focus-within {
  border-color:var(--primary-400);
  background:var(--surface);
}

.search-input {
  border:none;
  background:transparent;
  outline:none;
  font-size:0.875rem;
  color:var(--text-primary);
  width:180px;
}

/* Content Wrapper */
.content-wrapper {
  max-width:1400px;
  margin:0 auto;
  padding:1.5rem 2rem 2rem;
}

/* Loading / Error */
.loading {
  text-align:center;
  padding:4rem 2rem;
  background:var(--surface);
  border-radius:16px;
  color:var(--text-secondary);
  box-shadow: var(--shadow-sm);
  border:1px solid var(--border);
}

.error-message {
  padding:1rem 1.5rem;
  background-color:var(--danger-50, var(--danger-soft));
  color:var(--danger-700, var(--danger));
  border:1px solid rgba(248, 113, 113, 0.35);
  border-radius:12px;
}

/* Empty State */
.empty-state {
  text-align:center;
  padding:4rem 2rem;
  background:var(--surface);
  border-radius:16px;
  box-shadow: var(--shadow-sm);
  border:1px solid var(--border);
}

.empty-state h3 {
  font-size:1.5rem;
  font-weight:600;
  margin:1rem 0 0.75rem;
  color:var(--text-primary);
}

.empty-state p {
  color:var(--text-secondary);
  margin:0 0 2rem;
  font-size:1rem;
}

/* Projects Grid */
.projects-grid {
  display:grid;
  grid-template-columns:repeat(auto-fill, minmax(320px, 1fr));
  gap:1.25rem;
}

.project-card {
  background:var(--surface);
  cursor:pointer;
  padding:1.75rem;
  transition:all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position:relative;
  overflow:hidden;
  border:1px solid var(--border);
  border-radius:16px;
  box-shadow: var(--shadow-sm);
}

.project-card:hover {
  transform:translateY(-4px);
  box-shadow: var(--shadow-lg);
  border-color:var(--primary-200);
}

.project-header {
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
  margin-bottom:1.25rem;
}

.project-icon {
  width:56px;
  height:56px;
  background:linear-gradient(135deg, var(--primary-500), var(--primary-600));
  color:var(--text);
  border-radius:14px;
  display:flex;
  align-items:center;
  justify-content:center;
  box-shadow: var(--shadow);
  transition:transform 0.3s ease;
}

.project-card:hover .project-icon {
  transform:scale(1.05) rotate(5deg);
}

.project-actions {
  display:flex;
  gap:0.5rem;
  opacity:0;
  transition:opacity 0.3s ease;
}

.project-card:hover .project-actions {
  opacity:1;
}

.btn-icon {
  width:36px;
  height:36px;
  border:none;
  background-color:var(--gray-100);
  color:var(--gray-600);
  border-radius:10px;
  cursor:pointer;
  display:flex;
  align-items:center;
  justify-content:center;
  transition:all 0.2s;
}

.btn-icon:hover {
  background-color:var(--danger-100, var(--danger-soft));
  color:var(--danger-600, var(--danger));
  transform:scale(1.1);
}

.project-name {
  font-size:1.25rem;
  font-weight:600;
  margin:0 0 0.5rem;
  color:var(--text-primary);
}

.project-description {
  color:var(--text-secondary);
  font-size:0.9375rem;
  margin:0 0 1.25rem;
  line-height:1.6;
  display:-webkit-box;
  -webkit-line-clamp:2;
  line-clamp:2;
  -webkit-box-orient:vertical;
  overflow:hidden;
}

.project-stats {
  display:flex;
  gap:2rem;
  padding:1rem 0;
  border-top:1px solid var(--border-color);
  margin-bottom:1rem;
}

.stat {
  display:flex;
  align-items:center;
  gap:0.5rem;
  color:var(--text-secondary);
  font-size:0.875rem;
}

.stat-value {
  font-weight:700;
  color:var(--text-primary);
  font-size:1rem;
}

.project-tags {
  display:flex;
  flex-wrap:wrap;
  gap:0.5rem;
}

.badge {
  padding:0.25rem 0.75rem;
  border-radius:9999px;
  font-size:0.8125rem;
  font-weight:500;
  background-color:var(--primary-100);
  color:var(--primary-700);
  border:1px solid var(--primary-200);
}
</style>

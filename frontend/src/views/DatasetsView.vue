<template>
  <div class="datasets-view">

    <!-- Toolbar -->
    <div class="page-topbar">
      <div class="topbar-left">
        <h2 class="topbar-title">Datasets</h2>
        <span class="topbar-count">{{ datasets.length }}</span>
        <span class="format-hint">
          <Icon name="info" size="xs" />
          Import: a .zip exported from this platform
        </span>
      </div>
      <div class="topbar-right">
        <div class="search-box">
          <Icon name="search" size="sm" />
          <input v-model="searchQuery" type="text" placeholder="Search datasets..." class="search-input" />
        </div>
        <button @click="showUploadModal = true" class="btn btn-primary">
          <Icon name="upload" size="sm" />
          Import Dataset
        </button>
      </div>
    </div>

    <!-- Datasets Grid -->
    <div class="datasets-container">
      <p v-if="pageError" class="error-message">{{ pageError }}</p>
      <p v-if="notice" class="success-message">{{ notice }}</p>

      <div v-if="loading" class="loading-state">
        <Icon name="clock" size="xl" />
        <p>Loading datasets...</p>
      </div>

      <div v-else-if="datasets.length === 0" class="empty-state">
        <Icon name="box" size="xl" />
        <h3>No Datasets Yet</h3>
        <p>Export a dataset from a project, or import a ZIP file</p>
        <div class="empty-actions">
          <router-link to="/projects" class="btn btn-primary">
            <Icon name="folder" size="sm" />
            Go to Projects
          </router-link>
          <button class="btn btn-secondary" @click="showUploadModal = true">
            <Icon name="upload" size="sm" />
            Import Dataset (.zip)
          </button>
        </div>
      </div>

      <div v-else-if="filteredDatasets.length === 0" class="empty-state">
        <Icon name="search" size="xl" />
        <h3>No Results</h3>
        <p>No datasets match "{{ searchQuery }}"</p>
      </div>

      <div v-else class="datasets-grid">
        <div v-for="dataset in filteredDatasets" :key="dataset.project" class="dataset-card card">
          <div class="dataset-header">
            <div class="dataset-icon">
              <Icon name="box" size="lg" />
            </div>
            <div class="dataset-badges">
              <span class="badge badge-primary">{{ dataset.format }}</span>
              <span class="badge" :class="dataset.ready ? 'badge-success' : 'badge-warning'">
                {{ dataset.annotatedImages }} / {{ dataset.images }} annotated
              </span>
            </div>
          </div>

          <h3 class="dataset-name">{{ dataset.name }}</h3>
          <p class="dataset-project">
            <Icon name="folder" size="xs" />
            {{ dataset.project }}
          </p>

          <div class="dataset-stats">
            <div class="stat-item">
              <Icon name="image" size="sm" />
              <span>{{ dataset.images }} images</span>
            </div>
            <div class="stat-item">
              <Icon name="check" size="sm" />
              <span>{{ dataset.annotations }} boxes</span>
            </div>
            <div class="stat-item">
              <Icon name="layers" size="sm" />
              <span>{{ dataset.classes }} classes</span>
            </div>
          </div>

          <div class="dataset-footer">
            <div class="dataset-date">
              <Icon name="clock" size="sm" />
              {{ formatDate(dataset.created_at) }}
            </div>
            <div class="dataset-actions">
              <button
                class="btn-icon"
                title="Export as .zip"
                :disabled="exportingProject === dataset.project"
                @click="exportDataset(dataset.project)"
              >
                <Icon name="download" size="sm" />
              </button>
              <router-link
                class="btn-icon"
                title="Open project"
                :to="{ name: 'ProjectDetail', params: { name: dataset.project } }"
              >
                <Icon name="folder" size="sm" />
              </router-link>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Import Dataset Modal -->
    <div v-if="showUploadModal" class="modal-overlay" @click="showUploadModal = false">
      <div class="modal" @click.stop>
        <div class="modal-header">
          <h3 class="modal-title">Import Dataset</h3>
          <button @click="showUploadModal = false" class="modal-close">
            <Icon name="x" />
          </button>
        </div>

        <div class="modal-body">
          <!-- Format info card -->
          <div class="format-info-card">
            <Icon name="info" size="sm" />
            <div>
              <strong>Supported format: ZIP archive</strong>
              <p>The archive must be one produced by Export: it contains
                 <code>dataset.json</code> and an <code>images/</code> folder.
                 Its images and annotations are added to the project you choose
                 below; nothing already in that project is removed.</p>
            </div>
          </div>

          <div class="upload-area" @click="fileInput.click()" @drop.prevent="handleDrop" @dragover.prevent>
            <input type="file" ref="fileInput" @change="handleFileSelect" accept=".zip" style="display:none" />
            <div class="upload-content">
              <Icon name="upload" size="4xl" />
              <p>Click or drag &amp; drop ZIP file here</p>
              <p class="upload-hint">Required: .zip file only</p>
            </div>
          </div>

          <div v-if="uploadForm.file" class="selected-file-info">
            <Icon name="check-circle" size="sm" />
            {{ uploadForm.file.name }}
            <span class="file-size">({{ formatFileSize(uploadForm.file.size) }})</span>
          </div>

          <p v-if="uploadError" class="error-message">{{ uploadError }}</p>

          <div class="form-group" style="margin-top:1rem">
            <label class="form-label">Import into project *</label>
            <select v-model="uploadForm.projectName" class="form-input">
              <option value="">โ€” Select project โ€”</option>
              <option v-for="p in projects" :key="p.name" :value="p.name">{{ p.name }}</option>
            </select>
          </div>
        </div>

        <div class="modal-footer">
          <button @click="showUploadModal = false" class="btn btn-secondary">Cancel</button>
          <button @click="submitUpload" :disabled="uploading || !uploadForm.file || !uploadForm.projectName" class="btn btn-primary">
            {{ uploading ? 'Importing...' : 'Import dataset' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import Icon from '@/components/Icon.vue'
import { errorMessage, projectService } from '@/services'
import { formatBytes, formatDate } from '@/utils/format'

const showUploadModal = ref(false)
const loading = ref(true)
const uploading = ref(false)
const searchQuery = ref('')
const fileInput = ref(null)
const projects = ref([])
const pageError = ref(null)
const uploadError = ref(null)
const notice = ref(null)
const exportingProject = ref(null)

const uploadForm = ref({ projectName: '', file: null })

/**
 * Every project is its own dataset. There is no separate dataset store: the
 * project directory holds the images and annotations, and Export just packages
 * them into a zip. The old version listed rows it could neither export nor
 * delete for real.
 */
const datasets = computed(() =>
  projects.value
    .filter((project) => (project.total_images || 0) > 0)
    .map((project) => ({
      project: project.name,
      name: project.name + ' dataset',
      format: 'YOLO',
      images: project.total_images || 0,
      annotatedImages: project.annotated_images || 0,
      annotations: project.total_annotations || 0,
      classes: Object.keys(project.tags || {}).length,
      ready: (project.annotated_images || 0) >= (project.total_images || 0),
      created_at: project.updated_at || project.created_at
    }))
)

const filteredDatasets = computed(() => {
  const query = searchQuery.value.trim().toLowerCase()
  if (!query) return datasets.value
  return datasets.value.filter((dataset) =>
    dataset.name.toLowerCase().includes(query) ||
    dataset.project.toLowerCase().includes(query)
  )
})

const load = async () => {
  loading.value = true
  pageError.value = null
  try {
    projects.value = (await projectService.list()).projects
  } catch (error) {
    pageError.value = errorMessage(error)
  } finally {
    loading.value = false
  }
}

onMounted(load)

const flash = (message) => {
  notice.value = message
  window.setTimeout(() => { notice.value = null }, 6000)
}

const acceptFile = (file) => {
  if (!file) return
  if (!file.name.toLowerCase().endsWith('.zip')) {
    uploadError.value = 'Only .zip archives can be imported'
    return
  }
  uploadError.value = null
  uploadForm.value.file = file
}

const handleDrop = (event) => acceptFile(event.dataTransfer.files[0])
const handleFileSelect = (event) => acceptFile(event.target.files[0])

const submitUpload = async () => {
  if (!uploadForm.value.file || !uploadForm.value.projectName || uploading.value) return
  uploading.value = true
  uploadError.value = null
  try {
    const result = await projectService.importDataset(
      uploadForm.value.projectName, uploadForm.value.file
    )
    showUploadModal.value = false
    uploadForm.value = { projectName: '', file: null }
    if (fileInput.value) fileInput.value.value = ''
    flash(result.message)
    await load()
  } catch (error) {
    uploadError.value = errorMessage(error)
  } finally {
    uploading.value = false
  }
}

const exportDataset = async (projectName) => {
  if (exportingProject.value) return
  exportingProject.value = projectName
  pageError.value = null
  try {
    const blob = await projectService.exportDataset(projectName)
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = projectName + '_dataset.zip'
    document.body.appendChild(link)
    link.click()
    link.remove()
    // Revoking straight away can cancel the download in some browsers.
    window.setTimeout(() => window.URL.revokeObjectURL(url), 10000)
  } catch (error) {
    pageError.value = errorMessage(error)
  } finally {
    exportingProject.value = null
  }
}

const formatFileSize = formatBytes
</script>

<style scoped>
.datasets-view {
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
  flex-wrap:wrap;
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

.format-hint {
  display:flex;
  align-items:center;
  gap:0.25rem;
  font-size:0.8rem;
  color:var(--text-secondary);
  background:var(--gray-100);
  border:1px solid var(--border-color);
  border-radius:8px;
  padding:0.25rem 0.625rem;
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
  width:160px;
}

/* โ”€โ”€ Content โ”€โ”€ */
.datasets-container {
  max-width:1400px;
  margin:0 auto;
  padding:2rem;
}

.loading-state, .empty-state {
  text-align:center;
  padding:5rem 2rem;
  background:var(--surface);
  border-radius:16px;
  border:2px dashed var(--border-color);
  color:var(--text-secondary);
}

.empty-state h3 {
  font-size:1.5rem;
  font-weight:600;
  margin:1rem 0 0.5rem;
  color:var(--text-primary);
}

.empty-state p { margin-bottom:2rem; }

.empty-actions {
  display:flex;
  gap:1rem;
  justify-content:center;
  flex-wrap:wrap;
}

/* โ”€โ”€ Grid โ”€โ”€ */
.datasets-grid {
  display:grid;
  grid-template-columns:repeat(auto-fill, minmax(320px, 1fr));
  gap:1.5rem;
}

.dataset-card {
  padding:1.5rem;
  background:var(--surface);
  border-radius:16px;
  border:1px solid var(--border);
  box-shadow: var(--shadow-sm);
  transition:all 0.3s ease;
}

.dataset-card:hover {
  transform:translateY(-4px);
  box-shadow: var(--shadow-lg);
}

.dataset-header {
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
  margin-bottom:1rem;
}

.dataset-icon {
  width:52px;
  height:52px;
  background:linear-gradient(135deg, var(--accent-soft), var(--accent-soft));
  color:var(--accent);
  border-radius:14px;
  display:flex;
  align-items:center;
  justify-content:center;
}

.dataset-badges { display:flex; gap:0.5rem; flex-wrap:wrap; }

.badge {
  padding:0.25rem 0.75rem;
  border-radius:9999px;
  font-size:0.75rem;
  font-weight:600;
}

.badge-primary { background:var(--primary-100); color:var(--primary-700); }
.badge-secondary { background:var(--gray-100); color:var(--gray-700); }

.dataset-name {
  font-size:1.0625rem;
  font-weight:600;
  margin:0 0 0.25rem;
  color:var(--text-primary);
}

.dataset-project {
  display:flex;
  align-items:center;
  gap:0.3rem;
  color:var(--text-secondary);
  font-size:0.875rem;
  margin-bottom:1rem;
}

.dataset-stats {
  background:var(--gray-50);
  border-radius:10px;
  padding:0.875rem;
  margin-bottom:1rem;
  display:flex;
  flex-direction:column;
  gap:0.5rem;
}

.stat-item {
  display:flex;
  align-items:center;
  gap:0.5rem;
  font-size:0.875rem;
  color:var(--text-secondary);
}

.dataset-footer {
  display:flex;
  justify-content:space-between;
  align-items:center;
  padding-top:1rem;
  border-top:1px solid var(--border-color);
}

.dataset-date {
  display:flex;
  align-items:center;
  gap:0.375rem;
  font-size:0.8125rem;
  color:var(--text-secondary);
}

.dataset-actions { display:flex; gap:0.5rem; }

.btn-icon {
  width:34px;
  height:34px;
  border:none;
  background:var(--gray-100);
  color:var(--gray-600);
  border-radius:8px;
  cursor:pointer;
  display:flex;
  align-items:center;
  justify-content:center;
  text-decoration:none;
  transition:all 0.2s;
}

.btn-icon:hover { background:var(--primary-100); color:var(--primary-600); }
.btn-icon.danger:hover { background:var(--danger-100); color:var(--danger-600); }

/* โ”€โ”€ Modal โ”€โ”€ */
.modal-overlay {
  position:fixed;
  inset:0;
  background:rgba(0,0,0,0.55);
  display:flex;
  align-items:center;
  justify-content:center;
  z-index:1000;
  backdrop-filter:blur(4px);
}

.modal {
  background:var(--surface);
  border-radius:20px;
  box-shadow: var(--shadow-lg);
  width:90%;
  max-width:540px;
  max-height:90vh;
  overflow-y:auto;
}

.modal-header {
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:1.5rem 1.75rem;
  border-bottom:1px solid var(--border-color);
}

.modal-title { font-size:1.25rem; font-weight:600; margin:0; }

.modal-close {
  background:var(--gray-100);
  border:none;
  cursor:pointer;
  padding:0.4rem;
  border-radius:8px;
  display:flex;
  align-items:center;
  transition:all 0.2s;
}

.modal-close:hover { background:var(--danger-100); color:var(--danger-600); }

.modal-body { padding:1.5rem 1.75rem; }

.modal-footer {
  display:flex;
  justify-content:flex-end;
  gap:0.75rem;
  padding:1.25rem 1.75rem;
  border-top:1px solid var(--border-color);
}

/* โ”€โ”€ Format info โ”€โ”€ */
.format-info-card {
  display:flex;
  gap:0.75rem;
  padding:1rem;
  background:var(--info-soft);
  border: 1px solid rgba(96, 165, 250, 0.3);
  border-radius:10px;
  font-size:0.875rem;
  color:var(--info);
  margin-bottom:1rem;
}

.format-info-card strong { display:block; margin-bottom:0.25rem; }
.format-info-card p { margin:0.25rem 0; }

.format-list {
  margin:0.25rem 0 0 1rem;
  padding:0;
  list-style:disc;
}

.format-list li { margin-bottom:0.2rem; }

code {
  background:var(--border);
  padding:0.1rem 0.3rem;
  border-radius:4px;
  font-family:monospace;
  font-size:0.8125rem;
}

/* โ”€โ”€ Upload area โ”€โ”€ */
.upload-area {
  border:2px dashed var(--border-color);
  border-radius:12px;
  padding:2.5rem 2rem;
  text-align:center;
  cursor:pointer;
  transition:all 0.2s;
  background:var(--gray-50);
}

.upload-area:hover {
  border-color:var(--primary-500);
  background:var(--primary-50);
}

.upload-content { pointer-events:none; }
.upload-content p { color:var(--text-secondary); margin:0.5rem 0 0; }
.upload-hint { font-size:0.8125rem; }

.selected-file-info {
  display:flex;
  align-items:center;
  gap:0.5rem;
  padding:0.625rem 1rem;
  background:var(--success-50);
  border:1px solid var(--success-200);
  border-radius:8px;
  font-size:0.875rem;
  color:var(--success-700);
  margin-top:0.75rem;
}

.file-size { color:var(--text-secondary); font-size:0.8125rem; }

/* โ”€โ”€ Form โ”€โ”€ */
.form-group { margin-bottom:1rem; }
.form-label { display:block; font-size:0.875rem; font-weight:500; margin-bottom:0.375rem; color:var(--text-primary); }
.form-input {
  width:100%;
  padding:0.625rem 0.875rem;
  border:1px solid var(--border-color);
  border-radius:8px;
  font-size:0.875rem;
  outline:none;
  transition:border-color 0.2s;
  box-sizing:border-box;
}
.form-input:focus { border-color:var(--primary-400); }

.btn {
  display:inline-flex;
  align-items:center;
  gap:0.375rem;
  padding:0.5rem 1.125rem;
  border:none;
  border-radius:10px;
  font-size:0.875rem;
  font-weight:600;
  cursor:pointer;
  text-decoration:none;
  transition:all 0.2s;
}

.btn-primary { background:var(--primary-600); color:var(--text); }
.btn-primary:hover { background:var(--primary-700); }
.btn-primary:disabled { opacity:0.5; cursor:not-allowed; }
.btn-secondary { background:var(--gray-100); color:var(--text-primary); border:1px solid var(--border-color); }
.btn-secondary:hover { background:var(--gray-200); }
</style>

<template>
  <div class="project-detail-view">
    <div v-if="store.loading" class="loading">Loading project</div>

    <div v-else-if="store.current">
      <!-- Compact top bar -->
      <div class="project-topbar">
        <button @click="$router.back()" class="btn-back">
          <Icon name="arrow-left" size="sm" />
          Back
        </button>
        <div class="project-title-block">
          <h2 class="project-name-title">{{ store.current.name }}</h2>
          <span class="project-desc-small">{{ store.current.description || 'No description' }}</span>
        </div>
        <div class="header-actions">
          <button @click="showImportImages = true" class="btn btn-secondary">
            <Icon name="upload" size="sm" />
            <span>Import</span>
          </button>
          <button @click="showImportDataset = true" class="btn btn-secondary">
            <Icon name="package" size="sm" />
            <span>Dataset</span>
          </button>
          <button @click="exportDataset" class="btn btn-secondary">
            <Icon name="download" size="sm" />
            <span>Export</span>
          </button>
          <button @click="goToTrain" class="btn btn-primary">
            <Icon name="zap" size="sm" />
            <span>Train</span>
          </button>
        </div>
      </div>

      <!-- Main Content -->
      <div class="content-wrapper">

      <!-- Stats -->
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-icon primary">
            <Icon name="image" size="lg" />
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ store.current.total_images || 0 }}</div>
            <div class="stat-label">Total Images</div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="stat-icon success">
            <Icon name="check-circle" size="lg" />
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ store.current.annotated_images || 0 }}</div>
            <div class="stat-label">Annotated</div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="stat-icon warning">
            <Icon name="tag" size="lg" />
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ store.current.total_annotations || 0 }}</div>
            <div class="stat-label">Annotations</div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="stat-icon danger">
            <Icon name="layers" size="lg" />
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ Object.keys(store.current.tags || {}).length }}</div>
            <div class="stat-label">Classes</div>
          </div>
        </div>
      </div>

      <!-- The counters come from a cache; if the files behind them are gone,
           say so rather than showing 2232 images beside an empty gallery. -->
      <div v-if="imagesMissing" class="images-missing">
        <Icon name="alert-triangle" size="sm" />
        <div>
          <strong>The image files for this project are not where the app expects them.</strong>
          <p>
            The counters above come from <code>project.json</code> and still show
            {{ store.current.total_images }} images, but nothing is readable at
            <code class="text-mono">{{ store.summary?.images_path || 'the images folder' }}</code>.
            Annotating and training will not work until the files are back.
            If the dataset lives somewhere else, point <code>PROJECTS_ROOT</code>
            at it in <code>.env</code> and restart the backend.
          </p>
          <div class="images-missing-actions">
            <button class="btn btn-secondary btn-sm" :disabled="rescanning" @click="rescanProject()">
              <Icon name="refresh" size="xs" />
              {{ rescanning ? 'Checking…' : 'Check again' }}
            </button>
            <button class="btn btn-ghost btn-sm" :disabled="rescanning" @click="clearStaleCounts">
              Clear the counts
            </button>
            <span v-if="rescanMessage" class="images-missing-result">{{ rescanMessage }}</span>
          </div>
        </div>
      </div>

      <!-- Auto-label with a trained model -->
      <div class="autolabel-section">
        <h3 class="section-title">
          <Icon name="command" size="sm" />
          Auto-label with a trained model
          <span class="section-tip">ให้โมเดลที่เทรนแล้วตีกรอบให้ก่อน แล้วค่อยแก้</span>
        </h3>
        <p class="augment-desc">
          Runs a trained model over every image that has no annotations yet and
          writes its predictions in. Correcting a drawn box is far faster than
          drawing one, and everything it writes stays fully editable.
        </p>

        <div class="augment-controls">
          <label class="augment-field">
            <span>Confidence threshold</span>
            <input v-model.number="autoLabelThreshold" type="range" min="0.1" max="0.9" step="0.05" />
            <small>
              {{ autoLabelThreshold.toFixed(2) }} —
              {{ autoLabelThreshold < 0.35 ? 'more boxes, more to delete'
                 : autoLabelThreshold > 0.6 ? 'fewer boxes, more to add'
                 : 'balanced' }}
            </small>
          </label>

          <div class="augment-field">
            <span>Will process</span>
            <small class="autolabel-count">
              {{ unannotatedCount }} unannotated image{{ unannotatedCount === 1 ? '' : 's' }}
            </small>
          </div>

          <label class="augment-field">
            <span>Model</span>
            <select v-model="autoLabelModel" class="form-input">
              <option :value="''">This project's best run</option>
              <optgroup v-if="otherProjectModels.length" label="From another project">
                <option
                  v-for="model in otherProjectModels"
                  :key="model.path"
                  :value="model.path"
                >{{ model.project }} / {{ model.label }}</option>
              </optgroup>
            </select>
            <small>
              A model from a similar job usually gets the boxes close enough to
              be worth correcting, which is what makes a brand-new project
              worth pre-labelling at all.
            </small>
          </label>
        </div>

        <div v-if="autoLabelJob && autoLabelJob.status === 'running'" class="autolabel-progress">
          <div class="upload-progress">
            <div
              class="upload-progress-bar"
              :style="{ width: autoLabelPercent + '%' }"
            ></div>
            <span>{{ autoLabelJob.processed }} / {{ autoLabelJob.total }}</span>
          </div>
          <p class="autolabel-live">
            {{ autoLabelJob.labelled }} labelled &middot;
            {{ autoLabelJob.boxes }} boxes &middot;
            {{ autoLabelJob.skipped }} with nothing found
          </p>
        </div>

        <div v-else-if="autoLabelJob && autoLabelJob.status === 'completed'" class="autolabel-done">
          <Icon name="check-circle" size="sm" />
          Labelled {{ autoLabelJob.labelled }} images with
          {{ autoLabelJob.boxes }} boxes using
          <strong>{{ autoLabelJob.model_name }}</strong>.
          <span v-if="autoLabelJob.skipped">
            {{ autoLabelJob.skipped }} image(s) had no confident detection — lower
            the threshold or label those by hand.
          </span>
        </div>

        <p v-else-if="autoLabelJob && autoLabelJob.error" class="error-message">
          {{ autoLabelJob.error }}
        </p>

        <div class="augment-actions">
          <button
            v-if="!autoLabelRunning"
            class="btn btn-primary"
            :disabled="!unannotatedCount"
            @click="startAutoLabel"
          >
            <Icon name="command" size="sm" />
            <span>Auto-label {{ unannotatedCount }} image{{ unannotatedCount === 1 ? '' : 's' }}</span>
          </button>
          <button v-else class="btn btn-danger" @click="cancelAutoLabel">
            <Icon name="x" size="sm" />
            <span>Cancel</span>
          </button>
          <span v-if="!unannotatedCount" class="augment-ok">
            Every image is annotated.
          </span>
        </div>
      </div>

      <!-- Color Augmentation -->
      <div class="augment-section">
        <h3 class="section-title">
          <Icon name="zap" size="sm" />
          Generate Color Variants
        </h3>
        <p class="augment-desc">
          สร้างภาพหลายโทนสีจากรูปเดิมเพื่อเพิ่มข้อมูลเทรน โดยใช้ตำแหน่ง ROI เดิมอัตโนมัติ (ไม่ต้องตีกรอบใหม่)
        </p>

        <div class="augment-controls">
          <label class="augment-field">
            <span>Variants / Tone</span>
            <input v-model.number="augmentVariants" type="number" min="1" max="20" class="form-input" />
          </label>

          <label class="augment-field">
            <span>Strength</span>
            <input v-model.number="augmentStrength" type="range" min="0.2" max="2" step="0.1" />
            <small>{{ augmentStrength.toFixed(1) }}</small>
          </label>

          <div class="augment-field augment-auto-note">
            <span>Filters (26 ฟิลเตอร์อัตโนมัติ)</span>
            <div class="filter-groups">
              <div v-for="group in filterGroups" :key="group.label" class="filter-group">
                <span class="filter-group-head">
                  <Icon :name="group.icon" size="xs" />
                  {{ group.label }}
                </span>
                <span class="filter-group-items">{{ group.items }}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="augment-actions">
          <button
            class="btn btn-primary"
            :disabled="!canGenerateVariants || augmenting"
            @click="generateColorVariants"
          >
            <Icon name="zap" size="sm" />
            <span>{{ augmenting ? 'Generating...' : 'Generate Color Variants' }}</span>
          </button>
          <span v-if="!isFullyAnnotated" class="augment-warning">
            ต้อง Annotate ให้ครบก่อน ({{ annotatedCount }}/{{ totalCount }})
          </span>
          <span v-else class="augment-ok">
            พร้อมสร้างภาพเพิ่ม (ROI เดิมจะถูกคัดลอกไปภาพใหม่)
          </span>
        </div>

        <div v-if="augmentReport" class="augment-report">
          <p class="augment-report-line">
            <Icon name="check-circle" size="sm" />
            สร้างแล้ว {{ augmentReport.created }} จาก {{ augmentReport.planned }} ภาพ
          </p>
          <div v-if="augmentReport.dropped.length" class="augment-report-dropped">
            <p>
              <Icon name="alert-triangle" size="sm" />
              ฟิลเตอร์เหล่านี้ทำให้วัตถุในกรอบ ROI หายไป จึงไม่ถูกบันทึก
              (ภาพที่ label ชี้ไปยังพื้นที่ว่างจะทำให้โมเดลเรียนผิด)
            </p>
            <ul>
              <li v-for="[tone, count] in augmentReport.dropped" :key="tone">
                <code>{{ tone }}</code> &mdash; {{ count }} ภาพ
              </li>
            </ul>
          </div>
        </div>
      </div>

      <!-- Tags -->
      <div v-if="Object.keys(store.current.tags || {}).length > 0" class="tags-section">
        <h3 class="section-title">
          <Icon name="tag" size="sm" />
          Classes ({{ Object.keys(store.current.tags || {}).length }})
        </h3>
        <div class="tags-list">
          <span
            v-for="(stats, tag) in store.current.tags"
            :key="tag"
            class="tag-item"
            :title="`${stats.boxes} boxes across ${stats.images} images`"
          >
            <Icon name="tag" size="xs" />
            {{ tag }}: <strong>{{ stats.boxes }}</strong>
            <!--
              How many boxes were drawn says how much work went in; it says
              nothing about whether any of it worked. The figure from the last
              run is what tells someone which class to go and photograph more
              of.
            -->
            <span
              v-if="accuracyFor(tag) !== null"
              class="tag-accuracy"
              :class="accuracyClass(accuracyFor(tag))"
            >{{ Math.round(accuracyFor(tag) * 100) }}%</span>
          </span>
        </div>

        <div v-if="accuracy && accuracy.run" class="accuracy-note">
          <Icon name="chart-bar" size="sm" />
          <span>
            ความแม่นยำจากรอบ <strong>{{ accuracy.run }}</strong>
            <template v-if="accuracy.overall">
              (รวม {{ Math.round(accuracy.overall * 100) }}%)
            </template>
            <template v-if="accuracy.measured_at">
              เมื่อ {{ formatDateTime(accuracy.measured_at) }}
            </template>
            <template v-if="accuracy.note"> — {{ accuracy.note }}</template>
          </span>
        </div>
      </div>

      <!-- Images Grid -->
      <div class="images-section">
        <h3 class="section-title">
          <Icon name="image" size="sm" />
          Images ({{ store.images.length }})
          <span v-if="imagesMissing" class="images-expected">
            of {{ store.current.total_images }} expected
          </span>
        </h3>

        <!-- A project can hold thousands of images; rendering them all at once
             locks up the browser, so the grid is filtered and paged. -->
        <div v-if="store.images.length" class="images-toolbar">
          <div class="filter-chips">
            <button
              v-for="option in imageFilters"
              :key="option.value"
              class="filter-chip"
              :class="{ active: imageFilter === option.value }"
              @click="imageFilter = option.value"
            >
              {{ option.label }} ({{ option.count }})
            </button>
          </div>
          <label class="roi-toggle" title="Draw annotation boxes on each thumbnail">
            <input v-model="showRoi" type="checkbox" />
            <span>Show ROI</span>
          </label>
          <!--
            Filtered copies are cheap to make and were tedious to unmake: a
            preset that suited the data badly left a hundred images to delete
            one at a time.
          -->
          <button
            v-if="generatedCount"
            class="link-btn danger-link"
            :disabled="deletingImages"
            @click="deleteGenerated"
          >
            <Icon name="trash" size="sm" />
            {{ deletingImages ? 'กำลังลบ...' : `ลบภาพที่ฟิลเตอร์สร้าง (${generatedCount})` }}
          </button>

          <div v-if="pageCount > 1" class="pager">
            <button class="btn btn-secondary btn-sm" :disabled="page === 1" @click="page--">
              Prev
            </button>
            <span class="pager-label">Page {{ page }} / {{ pageCount }}</span>
            <button class="btn btn-secondary btn-sm" :disabled="page === pageCount" @click="page++">
              Next
            </button>
          </div>
        </div>

        <p v-if="actionError" class="error-message">{{ actionError }}</p>
        <p v-if="notice" class="success-message">{{ notice }}</p>
        
        <!-- Two different empty states. "This project never had images" and
             "this project's images are not on this machine" need completely
             different next steps, and showing the first when it is really the
             second is what makes the page look broken. -->
        <div v-if="imagesMissing" class="empty-state gallery-missing">
          <div class="empty-icon">
            <Icon name="alert-triangle" size="4xl" />
          </div>
          <h3>The image files are not on this machine</h3>
          <p>
            This project's records say it has
            <strong>{{ store.current.total_images }} images</strong> with
            <strong>{{ store.current.total_annotations }} boxes</strong> across
            {{ Object.keys(store.current.tags || {}).length }} classes, but the
            folder that should hold them is empty:
          </p>
          <p class="text-mono missing-path">{{ store.summary?.images_path }}</p>
          <p>
            Only <code>project.json</code> survived — the image and annotation
            files were never copied into this checkout. Put them back in
            <code>images\</code> and <code>annotations\</code> under the path
            above, then press <strong>Check again</strong>. Nothing is lost in
            the meantime; the counts and classes are still recorded.
          </p>
          <div class="gallery-missing-actions">
            <button class="btn btn-primary" :disabled="rescanning" @click="rescanProject()">
              <Icon name="refresh" size="sm" />
              {{ rescanning ? 'Checking…' : 'Check again' }}
            </button>
            <button class="btn btn-secondary" @click="showImportImages = true">
              <Icon name="upload" size="sm" />
              Import images instead
            </button>
          </div>
          <p v-if="rescanMessage" class="gallery-missing-result">{{ rescanMessage }}</p>
        </div>

        <div v-else-if="store.images.length === 0" class="empty-state">
          <div class="empty-icon">
            <Icon name="image" size="4xl" />
          </div>
          <p>No images yet. Import images to get started.</p>
          <button @click="showImportImages = true" class="btn btn-primary">
            <Icon name="upload" size="sm" />
            Import Images
          </button>
        </div>
        
        <div v-else-if="!pagedImages.length" class="empty-state">
          <p>No images match this filter.</p>
        </div>

        <div v-else class="grid grid-4">
          <div
            v-for="image in pagedImages"
            :key="image.filename"
            class="image-card card"
            @click="goToAnnotate(image.filename)"
          >
            <div class="image-preview">
              <RoiThumbnail
                :src="projectService.imageUrl(projectName, image.filename)"
                :alt="image.filename"
                :boxes="showRoi ? image.boxes : []"
                :width="image.width"
                :height="image.height"
                :classes="projectClasses"
              >
                <template #failed>
                  Image file missing
                </template>
                <button
                  class="image-delete"
                  title="Delete image"
                  @click.stop="deleteImage(image.filename)"
                >
                  <Icon name="trash" size="xs" />
                </button>
              </RoiThumbnail>
            </div>
            
            <div class="image-info">
              <div class="image-name" :title="image.filename">
                {{ truncateFilename(image.filename) }}
              </div>
              
              <div class="image-stats">
                <span v-if="image.annotated" class="badge badge-success">
                  <Icon name="check" size="xs" />
                  Annotated ({{ image.regions_count }})
                </span>
                <span v-else class="badge badge-warning">
                  <Icon name="clock" size="xs" />
                  Pending
                </span>
              </div>
              
              <div v-if="image.tags && image.tags.length > 0" class="image-tags">
                <span
                  v-for="tag in image.tags"
                  :key="tag"
                  class="tag-badge"
                >
                  {{ tag }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
      </div><!-- End content-wrapper -->
    </div>

    <!-- Import Images Modal -->
    <div v-if="showImportImages" class="modal-overlay" @click="showImportImages = false">
      <div class="modal" @click.stop>
        <div class="modal-header">
          <h3 class="modal-title">Import Images</h3>
          <button @click="showImportImages = false" class="modal-close">&times;</button>
        </div>
        
        <div class="modal-body">
          <div class="upload-area" @click="fileInput.click()" @drop.prevent="handleDrop" @dragover.prevent>
            <input
              type="file"
              ref="fileInput"
              multiple
              accept=".jpg,.jpeg,.png,.bmp,.webp"
              @change="handleFileSelect"
              style="display: none"
            />
            
            <div class="upload-content">
              <div class="upload-icon">
                <Icon name="folder" size="4xl" />
              </div>
              <p>Click or drag images here</p>
              <p class="upload-hint">Support: JPG, PNG, BMP, WEBP</p>
            </div>
          </div>
          
          <div v-if="uploading" class="upload-progress">
            <div class="upload-progress-bar" :style="{ width: uploadProgress + '%' }"></div>
            <span>{{ uploadProgress }}%</span>
          </div>

          <p v-if="actionError" class="error-message">{{ actionError }}</p>

          <div v-if="selectedFiles.length > 0" class="selected-files">
            <p>Selected: {{ selectedFiles.length }} files</p>
            <ul class="file-list">
              <li v-for="(file, idx) in selectedFiles.slice(0, 5)" :key="idx">
                {{ file.name }}
              </li>
              <li v-if="selectedFiles.length > 5">
                ... and {{ selectedFiles.length - 5 }} more
              </li>
            </ul>
          </div>
        </div>
        
        <div class="modal-footer">
          <button @click="showImportImages = false" class="btn btn-secondary">
            Cancel
          </button>
          <button
            @click="uploadImages"
            :disabled="selectedFiles.length === 0 || uploading"
            class="btn btn-primary"
          >
            {{ uploading ? 'Uploading…' : `Upload ${selectedFiles.length} images` }}
          </button>
        </div>
      </div>
    </div>

    <!-- Import Dataset Modal -->
    <div v-if="showImportDataset" class="modal-overlay" @click="showImportDataset = false">
      <div class="modal" @click.stop>
        <div class="modal-header">
          <h3 class="modal-title">Import Dataset</h3>
          <button @click="showImportDataset = false" class="modal-close">&times;</button>
        </div>
        
        <div class="modal-body">
          <div class="upload-area" @click="datasetInput.click()">
            <input
              type="file"
              ref="datasetInput"
              accept=".zip"
              @change="handleDatasetSelect"
              style="display: none"
            />
            
            <div class="upload-content">
              <div class="upload-icon">
                <Icon name="package" size="4xl" />
              </div>
              <p>Click to select dataset ZIP file</p>
              <p class="upload-hint">Previously exported dataset</p>
            </div>
          </div>
          
          <div v-if="selectedDataset" class="selected-files">
            <p>Selected: {{ selectedDataset.name }}</p>
          </div>

          <p v-if="actionError" class="error-message">{{ actionError }}</p>
        </div>
        
        <div class="modal-footer">
          <button @click="showImportDataset = false" class="btn btn-secondary">
            Cancel
          </button>
          <button
            @click="uploadDataset"
            :disabled="!selectedDataset || uploading"
            class="btn btn-primary"
          >
            {{ uploading ? 'Importing...' : 'Import Dataset' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import Icon from '@/components/Icon.vue'
import RoiThumbnail from '@/components/RoiThumbnail.vue'
import { errorMessage, projectService, trainingService } from '@/services'
import { useProjectStore } from '@/stores/projectStore'
import { formatDateTime } from '@/utils/format'

const route = useRoute()
const router = useRouter()
const store = useProjectStore()

const projectName = computed(() => route.params.name)

const showImportImages = ref(false)
const showImportDataset = ref(false)
const selectedFiles = ref([])
const selectedDataset = ref(null)
const uploading = ref(false)
const uploadProgress = ref(0)
const fileInput = ref(null)
const datasetInput = ref(null)
const augmenting = ref(false)
const augmentVariants = ref(3)
const augmentStrength = ref(1.0)
// Kept on screen rather than flashed: when a preset does not suit this
// project's images the run drops those variants, and knowing which ones is
// what tells you to deselect them next time.
const augmentReport = ref(null)

const notice = ref(null)
const actionError = ref(null)

// Grouped rather than one long line, so the three families of filter are
// distinguishable without a coloured emoji doing the work.
const filterGroups = [
  {
    icon: 'image',
    label: 'สี',
    items: 'warm, cool, bright, dark, vivid, sepia, high_contrast, invert'
  },
  {
    icon: 'layers',
    label: 'B&W / contrast',
    items: 'gray, equalize, clahe, clahe_strong, sharpen, unsharp, '
         + 'adaptive_thresh, tophat, blackhat, gamma_low, gamma_high, bilateral'
  },
  {
    icon: 'square',
    label: 'Contour / Edge (เน้นเลขชัด)',
    items: 'canny_overlay, contour_inv, laplacian, sobel, emboss, clahe_sharp'
  }
]

const rescanning = ref(false)
const rescanMessage = ref('')

const autoLabelThreshold = ref(0.4)
// Empty means "this project's own best run"; otherwise the path of a model
// trained elsewhere. A new project has none of its own, which is exactly when
// pre-labelling saves the most work.
// Per-class accuracy from the newest completed run. Kept separate from the
// tag counts because they answer different questions: one is how much was
// drawn, the other is whether it worked.
const accuracy = ref(null)

const loadAccuracy = async () => {
  try {
    accuracy.value = await projectService.classAccuracy(projectName.value)
  } catch {
    accuracy.value = null
  }
}

const accuracyFor = (tag) => {
  const entry = (accuracy.value?.classes || []).find((c) => c.name === tag)
  // null rather than 0 for a class the validation split never contained: not
  // measured is not the same as scored zero, and showing 0% would send someone
  // to fix a model that was never tested on it.
  return entry && entry.measured ? entry.ap50 : null
}

const accuracyClass = (value) =>
  value >= 0.7 ? 'tag-accuracy--good'
    : value >= 0.4 ? 'tag-accuracy--fair'
      : 'tag-accuracy--poor'

const deletingImages = ref(false)
const generatedCount = computed(
  () => store.images.filter((image) => image.augmented).length)

const deleteGenerated = async () => {
  if (!generatedCount.value || deletingImages.value) return
  if (!confirm(
    `Delete all ${generatedCount.value} images generated by filters? ` +
    'The photographs you annotated are not touched.'
  )) return

  deletingImages.value = true
  actionError.value = null
  try {
    const result = await projectService.deleteImages(projectName.value,
                                                     { only_generated: true })
    flash(result.message)
    await refresh()
  } catch (error) {
    actionError.value = errorMessage(error)
  } finally {
    deletingImages.value = false
  }
}

const autoLabelModel = ref('')
const otherProjectModels = ref([])

const loadOtherProjectModels = async () => {
  try {
    const models = await trainingService.listTrainedModels()
    otherProjectModels.value = models.filter(
      (m) => m.project !== projectName.value && m.checkpoint === 'best')
  } catch {
    otherProjectModels.value = []
  }
}
const autoLabelJob = ref(null)
let autoLabelTimer = null

const totalCount = computed(() => Number(store.current?.total_images || 0))
const annotatedCount = computed(() => Number(store.current?.annotated_images || 0))
const isFullyAnnotated = computed(
  () => totalCount.value > 0 && annotatedCount.value >= totalCount.value
)
const canGenerateVariants = computed(
  () => isFullyAnnotated.value && augmentVariants.value >= 1 && !augmenting.value
)

const PAGE_SIZE = 60

const imageFilter = ref('all')
const page = ref(1)

// Remembered per browser: an annotator wants the outlines on, someone
// reviewing image quality wants them off, and neither should have to toggle
// it on every visit.
const ROI_KEY = 'vision-training.showRoi'
const showRoi = ref(true)
try {
  const stored = window.localStorage.getItem(ROI_KEY)
  if (stored !== null) showRoi.value = stored === '1'
} catch { /* private mode: keep the default */ }

watch(showRoi, (value) => {
  try {
    window.localStorage.setItem(ROI_KEY, value ? '1' : '0')
  } catch { /* not fatal */ }
})

const imageFilters = computed(() => {
  const images = store.images
  return [
    { value: 'all', label: 'All', count: images.length },
    { value: 'annotated', label: 'Annotated', count: images.filter((i) => i.annotated).length },
    { value: 'pending', label: 'Pending', count: images.filter((i) => !i.annotated).length },
    { value: 'augmented', label: 'Augmented', count: images.filter((i) => i.augmented).length }
  ]
})

const filteredImages = computed(() => {
  switch (imageFilter.value) {
    case 'annotated': return store.images.filter((i) => i.annotated)
    case 'pending': return store.images.filter((i) => !i.annotated)
    case 'augmented': return store.images.filter((i) => i.augmented)
    default: return store.images
  }
})

const pageCount = computed(() =>
  Math.max(1, Math.ceil(filteredImages.value.length / PAGE_SIZE))
)

const pagedImages = computed(() =>
  filteredImages.value.slice((page.value - 1) * PAGE_SIZE, page.value * PAGE_SIZE)
)

// Changing the filter (or deleting the last image on a page) can leave the
// page number past the end of the list.
watch([filteredImages, imageFilter], () => {
  if (page.value > pageCount.value) page.value = 1
})

const unannotatedCount = computed(
  () => store.images.filter((image) => !image.annotated).length
)

// Sorted server-side, which is what keeps each class's ROI colour stable.
const projectClasses = computed(() => Object.keys(store.current?.tags || {}))

// True when the cached counters claim images that are not on disk.
const imagesMissing = computed(() =>
  Boolean(store.current)
  && store.current.images_available === false
  && (store.current.total_images || 0) > 0
)

const autoLabelRunning = computed(
  () => autoLabelJob.value?.status === 'running'
       || autoLabelJob.value?.status === 'cancelling'
)

const autoLabelPercent = computed(() => {
  const job = autoLabelJob.value
  if (!job?.total) return 0
  return Math.round((job.processed / job.total) * 100)
})

const refresh = () => store.refresh(projectName.value)

const pollAutoLabel = () => {
  if (autoLabelTimer) return
  autoLabelTimer = window.setInterval(async () => {
    try {
      const { job } = await projectService.autoLabelStatus(projectName.value)
      autoLabelJob.value = job
      if (!job || job.status !== 'running') {
        stopAutoLabelPolling()
        // The gallery and counters change as boxes are written, so the page
        // is reloaded once the pass ends rather than left stale.
        await refresh()
      }
    } catch {
      stopAutoLabelPolling()
    }
  }, 1500)
}

const stopAutoLabelPolling = () => {
  if (autoLabelTimer) {
    window.clearInterval(autoLabelTimer)
    autoLabelTimer = null
  }
}

/** Re-read the project from disk, so restored files appear without a restart. */
const rescanProject = async (clearIfMissing = false) => {
  rescanning.value = true
  rescanMessage.value = ''
  try {
    const result = await projectService.rescan(projectName.value, {
      clear_if_missing: clearIfMissing
    })
    rescanMessage.value = result.message
    await refresh()
  } catch (error) {
    rescanMessage.value = errorMessage(error)
  } finally {
    rescanning.value = false
  }
}

const clearStaleCounts = async () => {
  if (!confirm(
    'Reset this project\'s image and annotation counts to zero?\n\n' +
    'Only the cached numbers are cleared. If the image files come back later, '
    + 'use "Check again" instead and nothing will have been lost.'
  )) return
  await rescanProject(true)
}

const startAutoLabel = async () => {
  actionError.value = null
  try {
    const { job } = await projectService.startAutoLabel(projectName.value, {
      score_threshold: autoLabelThreshold.value,
      only_unannotated: true,
      // Empty means the project's own best run; the server decides then.
      model_path: autoLabelModel.value || undefined
    })
    autoLabelJob.value = job
    pollAutoLabel()
  } catch (error) {
    actionError.value = errorMessage(error)
  }
}

const cancelAutoLabel = async () => {
  try {
    await projectService.cancelAutoLabel(projectName.value)
  } catch (error) {
    actionError.value = errorMessage(error)
  }
}

onMounted(async () => {
  await refresh()
  await loadOtherProjectModels()
  await loadAccuracy()
  try {
    const { job } = await projectService.autoLabelStatus(projectName.value)
    autoLabelJob.value = job
    // A pass started before this page was opened keeps reporting progress.
    if (job?.status === 'running') pollAutoLabel()
  } catch { /* the panel simply starts idle */ }
})

onBeforeUnmount(stopAutoLabelPolling)

const flash = (message) => {
  notice.value = message
  actionError.value = null
  window.setTimeout(() => { notice.value = null }, 6000)
}

const ACCEPTED_TYPES = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']

const isImageFile = (file) =>
  file.type.startsWith('image/') ||
  ACCEPTED_TYPES.some((ext) => file.name.toLowerCase().endsWith(ext))

const handleFileSelect = (event) => {
  selectedFiles.value = Array.from(event.target.files).filter(isImageFile)
}

const handleDrop = (event) => {
  selectedFiles.value = Array.from(event.dataTransfer.files).filter(isImageFile)
}

const handleDatasetSelect = (event) => {
  selectedDataset.value = event.target.files[0] || null
}

const uploadImages = async () => {
  if (!selectedFiles.value.length || uploading.value) return
  uploading.value = true
  uploadProgress.value = 0
  actionError.value = null
  try {
    const result = await projectService.uploadImages(
      projectName.value,
      selectedFiles.value,
      (percent) => { uploadProgress.value = percent }
    )
    showImportImages.value = false
    selectedFiles.value = []
    if (fileInput.value) fileInput.value.value = ''

    let message = `Imported ${result.imported_count} images.`
    if (result.rejected?.length) {
      message += ` ${result.rejected.length} file(s) were rejected.`
    }
    flash(message)
    await refresh()
  } catch (error) {
    actionError.value = errorMessage(error)
  } finally {
    uploading.value = false
    uploadProgress.value = 0
  }
}

const uploadDataset = async () => {
  if (!selectedDataset.value || uploading.value) return
  uploading.value = true
  actionError.value = null
  try {
    const result = await projectService.importDataset(projectName.value, selectedDataset.value)
    showImportDataset.value = false
    selectedDataset.value = null
    if (datasetInput.value) datasetInput.value.value = ''
    flash(result.message)
    await refresh()
  } catch (error) {
    actionError.value = errorMessage(error)
  } finally {
    uploading.value = false
  }
}

const exportDataset = async () => {
  actionError.value = null
  try {
    const blob = await projectService.exportDataset(projectName.value)
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${projectName.value}_dataset.zip`
    document.body.appendChild(link)
    link.click()
    link.remove()
    // Revoking immediately can cancel the download in some browsers.
    window.setTimeout(() => window.URL.revokeObjectURL(url), 10000)
  } catch (error) {
    actionError.value = errorMessage(error)
  }
}

const generateColorVariants = async () => {
  if (!canGenerateVariants.value) return

  const tonesPerImage = augmentVariants.value
  if (!confirm(
    `Generate ${tonesPerImage} variant(s) per colour preset from every annotated ` +
    'image? This can add a large number of files.'
  )) return

  augmenting.value = true
  actionError.value = null
  augmentReport.value = null
  try {
    const result = await projectService.augment(projectName.value, {
      variants_per_tone: tonesPerImage,
      strength: augmentStrength.value,
      require_all_annotated: true
    })
    augmentReport.value = {
      created: result.created_count,
      planned: result.planned,
      dropped: Object.entries(result.dropped_by_tone || {})
        .sort((a, b) => b[1] - a[1])
    }
    flash(`${result.message} (${result.created_count} created)`)
    await refresh()
  } catch (error) {
    actionError.value = errorMessage(error)
  } finally {
    augmenting.value = false
  }
}

const deleteImage = async (filename) => {
  if (!confirm(`Delete ${filename} and its annotations?`)) return
  actionError.value = null
  try {
    await projectService.deleteImage(projectName.value, filename)
    await refresh()
  } catch (error) {
    actionError.value = errorMessage(error)
  }
}

const goToAnnotate = (filename) => {
  router.push({ name: 'Annotate', params: { name: projectName.value, filename } })
}

const goToTrain = () => {
  router.push({ name: 'Train', params: { name: projectName.value } })
}

const truncateFilename = (filename, maxLength = 20) => {
  if (filename.length <= maxLength) return filename
  const ext = filename.split('.').pop()
  const stem = filename.substring(0, Math.max(1, maxLength - ext.length - 4))
  return `${stem}...${ext}`
}

</script>

<style scoped>
.project-detail-view {
  min-height:100vh;
  background:var(--grad-surface) 100%);
}

/* Project Top Bar */
.project-topbar {
  display:flex;
  align-items:center;
  gap:1rem;
  padding:0.75rem 1.5rem;
  background:var(--surface);
  border-bottom:1px solid var(--border-color);
  position:sticky;
  top:0;
  z-index:50;
}

.project-title-block {
  flex:1;
  overflow:hidden;
}

.project-name-title {
  font-size:1.125rem;
  font-weight:600;
  margin:0 0 0.125rem 0;
  color:var(--text-primary);
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}

.project-desc-small {
  font-size:0.8125rem;
  color:var(--text-secondary);
  display:block;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}

.btn-back {
  display:inline-flex;
  align-items:center;
  gap:0.375rem;
  padding:0.375rem 0.875rem;
  background:transparent;
  border:1px solid var(--border-color);
  color:var(--text-secondary);
  border-radius:8px;
  font-weight:500;
  font-size:0.8125rem;
  cursor:pointer;
  transition:all 0.2s;
  white-space:nowrap;
  flex-shrink:0;
}

.btn-back:hover {
  background:var(--gray-50);
  color:var(--text-primary);
  border-color:var(--gray-300);
}

.header-actions {
  display:flex;
  gap:0.5rem;
  align-items:center;
  flex-wrap:wrap;
  flex-shrink:0;
}

/* Content Wrapper */
.content-wrapper {
  padding:0 2rem 2rem;
  max-width:1400px;
  margin:0 auto;
}

/* Stats Grid */
.stats-grid {
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));
  gap:1.25rem;
  margin-bottom:2rem;
}

.stat-card {
  background:var(--surface);
  padding:1.75rem;
  border-radius:16px;
  box-shadow: var(--shadow-sm);
  display:flex;
  align-items:center;
  gap:1.25rem;
  transition:all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border:1px solid var(--border);
  position:relative;
  overflow:hidden;
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
  width:60px;
  height:60px;
  border-radius:14px;
  display:flex;
  align-items:center;
  justify-content:center;
  color:var(--text);
  box-shadow: var(--shadow);
  flex-shrink:0;
}

.stat-icon.primary {
  background:linear-gradient(135deg, var(--primary-500), var(--primary-600));
}

.stat-icon.success {
  background:linear-gradient(135deg, var(--success-500), var(--success-600));
}

.stat-icon.warning {
  background:linear-gradient(135deg, var(--warning-500), var(--warning-600));
}

.stat-icon.danger {
  background:linear-gradient(135deg, var(--danger-500), var(--danger-600));
}

.stat-value {
  font-size:2.25rem;
  font-weight:700;
  color:var(--text-primary);
  line-height:1;
  margin-bottom:0.25rem;
}

.stat-label {
  color:var(--text-secondary);
  font-size:0.9375rem;
  font-weight:500;
}

/* Tags Section */
.tags-section {
  background:var(--surface);
  padding:1.75rem;
  border-radius:16px;
  margin-bottom:1.5rem;
  box-shadow: var(--shadow-sm);
  border:1px solid var(--border);
}

.augment-section {
  background:var(--surface);
  padding:1.75rem;
  border-radius:16px;
  margin-bottom:1.5rem;
  box-shadow: var(--shadow-sm);
  border:1px solid var(--border);
}

.augment-desc {
  margin:-0.4rem 0 1rem 0;
  color:var(--text-secondary);
  font-size:0.92rem;
}

.augment-controls {
  display:grid;
  grid-template-columns:1fr 1fr 2fr;
  gap:1rem;
  margin-bottom:1rem;
}

.augment-field {
  display:flex;
  flex-direction:column;
  gap:0.45rem;
}

.augment-field > span {
  font-size:0.85rem;
  font-weight:600;
  color:var(--text-primary);
}

.augment-auto-note {
  justify-content:center;
  background:var(--gray-50);
  border:1px dashed var(--border-color);
  border-radius:10px;
  padding:0.6rem 0.8rem;
}

.augment-auto-note small {
  color:var(--text-secondary);
  line-height:1.4;
}

.augment-actions {
  display:flex;
  align-items:center;
  gap:0.8rem;
  flex-wrap:wrap;
}

.augment-warning {
  color:var(--warning-700);
  background:var(--warning-100);
  border:1px solid var(--warning-300);
  border-radius:999px;
  padding:0.3rem 0.75rem;
  font-size:0.82rem;
}

.augment-report {
  margin-top:1rem;
  padding:0.9rem 1rem;
  border:1px solid var(--border-color);
  border-radius:var(--radius-md);
  background:var(--surface-2);
  font-size:0.88rem;
}

.augment-report-line {
  display:flex;
  align-items:center;
  gap:0.5rem;
  margin:0;
  color:var(--text-secondary);
}

.augment-report-dropped {
  margin-top:0.75rem;
  padding-top:0.75rem;
  border-top:1px solid var(--border-color);
  color:var(--warning-700);
}

.augment-report-dropped p {
  display:flex;
  align-items:flex-start;
  gap:0.5rem;
  margin:0 0 0.5rem 0;
  line-height:1.6;
}

.augment-report-dropped ul {
  margin:0;
  padding-left:1.6rem;
  color:var(--text-secondary);
}

.augment-report-dropped code {
  color:var(--text-primary);
  font-family:var(--font-mono);
}

.augment-ok {
  color:var(--success-700);
  background:var(--success-100);
  border:1px solid var(--success-300);
  border-radius:999px;
  padding:0.3rem 0.75rem;
  font-size:0.82rem;
}

.section-title {
  font-size:1.25rem;
  font-weight:600;
  margin:0 0 1.25rem 0;
  color:var(--text-primary);
  display:flex;
  align-items:center;
  gap:0.5rem;
}

.tag-accuracy {
  margin-left:0.35rem;
  padding:0.05rem 0.35rem;
  border-radius:999px;
  font-size:0.74rem;
  font-weight:700;
  font-family:var(--font-mono);
}

.tag-accuracy--good { background:var(--success-100); color:var(--success-700); }
.tag-accuracy--fair { background:var(--warning-100); color:var(--warning-700); }
.tag-accuracy--poor { background:var(--danger-100);  color:var(--danger-700); }

.accuracy-note {
  display:flex;
  align-items:flex-start;
  gap:0.5rem;
  margin-top:0.6rem;
  color:var(--text-secondary);
  font-size:0.82rem;
  line-height:1.6;
}

.tags-list {
  display:flex;
  flex-wrap:wrap;
  gap:0.75rem;
}

.tag-item {
  display:flex;
  align-items:center;
  gap:0.375rem;
  padding:0.5rem 1rem;
  background:var(--grad-surface));
  color:var(--primary-700);
  border-radius:9999px;
  font-size:0.9375rem;
  border:1px solid var(--primary-200);
  transition:all 0.2s;
}

.tag-item:hover {
  background:linear-gradient(135deg, var(--primary-100), var(--primary-200));
  transform:scale(1.05);
}

.tag-item strong {
  font-weight:600;
}

/* Images Section */
.images-section {
  margin-bottom:2rem;
}

.images-section .section-title {
  margin-bottom:1.5rem;
}

.empty-state {
  text-align:center;
  padding:4rem 2rem;
  background:var(--surface);
  border-radius:16px;
  box-shadow: var(--shadow-sm);
  border:2px dashed var(--border-strong);
}

.empty-icon {
  color:var(--primary-300);
  margin-bottom:1.5rem;
  display:flex;
  justify-content:center;
}

.empty-state p {
  color:var(--text-secondary);
  margin:0 0 2rem 0;
  font-size:1.0625rem;
}

.grid-4 {
  display:grid;
  grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));
  gap:1.25rem;
}

.image-card {
  background:var(--surface);
  border-radius:16px;
  overflow:hidden;
  box-shadow: var(--shadow-sm);
  cursor:pointer;
  transition:all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border:1px solid var(--border);
}

.image-card:hover {
  transform:translateY(-4px);
  box-shadow: var(--shadow-lg);
  border-color:var(--primary-300);
}

.image-preview {
  width:100%;
  aspect-ratio:4/3;
  overflow:hidden;
  background:linear-gradient(135deg, var(--gray-100), var(--gray-200));
  position:relative;
}

.image-preview::after {
  content:'';
  position:absolute;
  inset:0;
  background:rgba(0, 0, 0, 0);
  transition:background 0.3s ease;
}

.image-card:hover .image-preview::after {
  background:var(--accent-soft);
}

.image-preview img {
  width:100%;
  height:100%;
  object-fit:cover;
  transition:transform 0.3s ease;
}

.image-card:hover .image-preview img {
  transform:scale(1.05);
}

.image-info {
  padding:1.25rem;
  display:flex;
  flex-direction:column;
  gap:0.75rem;
}

.image-name {
  font-weight:600;
  color:var(--text-primary);
  font-size:1rem;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}

.image-stats {
  display:flex;
  gap:0.5rem;
  align-items:center;
}

.badge {
  padding:0.375rem 0.875rem;
  border-radius:9999px;
  font-size:0.8125rem;
  font-weight:500;
  display:inline-flex;
  align-items:center;
  gap:0.375rem;
}

.badge-success {
  background:linear-gradient(135deg, var(--success-100), var(--success-200));
  color:var(--success-700);
  border:1px solid var(--success-300);
}

.badge-warning {
  background:linear-gradient(135deg, var(--warning-100), var(--warning-200));
  color:var(--warning-700);
  border:1px solid var(--warning-300);
}

.image-tags {
  display:flex;
  flex-wrap:wrap;
  gap:0.375rem;
}

.tag-badge {
  background:var(--primary-100);
  color:var(--primary-700);
  padding:0.25rem 0.625rem;
  border-radius:9999px;
  font-size:0.8125rem;
  font-weight:500;
  border:1px solid var(--primary-200);
}

/* Modal Styling */
.modal-overlay {
  position:fixed;
  top:0;
  left:0;
  right:0;
  bottom:0;
  background-color:rgba(0, 0, 0, 0.6);
  display:flex;
  align-items:center;
  justify-content:center;
  z-index:1000;
  backdrop-filter:blur(4px);
  animation:fadeIn 0.2s ease-out;
}

.modal {
  background:var(--surface);
  border-radius:20px;
  box-shadow: var(--shadow-lg);
  width:90%;
  max-width:600px;
  max-height:90vh;
  overflow:hidden;
  animation:slideUp 0.3s ease-out;
}

.modal-header {
  padding:1.75rem 2rem;
  border-bottom:1px solid var(--border-color);
  display:flex;
  justify-content:space-between;
  align-items:center;
  background:var(--grad-surface));
}

.modal-title {
  font-size:1.375rem;
  font-weight:600;
  margin:0;
  color:var(--text-primary);
}

.modal-close {
  background:var(--gray-100);
  border:none;
  cursor:pointer;
  padding:0.5rem;
  color:var(--text-secondary);
  transition:all 0.2s;
  border-radius:10px;
  width:36px;
  height:36px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:1.5rem;
  line-height:1;
}

.modal-close:hover {
  background-color:var(--danger-100);
  color:var(--danger-600);
  transform:rotate(90deg);
}

.modal-body {
  padding:2rem;
}

.modal-footer {
  padding:1.5rem 2rem;
  border-top:1px solid var(--border-color);
  display:flex;
  justify-content:flex-end;
  gap:1rem;
  background-color:var(--gray-50);
}

.upload-area {
  border:3px dashed var(--border-color);
  border-radius:16px;
  padding:3rem 2rem;
  text-align:center;
  cursor:pointer;
  transition:all 0.3s;
  background:var(--gray-50);
}

.upload-area:hover {
  border-color:var(--primary-500);
  background:var(--primary-50);
}

.upload-content {
  pointer-events:none;
}

.upload-icon {
  margin-bottom:1rem;
  color:var(--primary-400);
  display:flex;
  justify-content:center;
}

.upload-content p {
  margin:0.5rem 0;
  color:var(--text-primary);
  font-weight:500;
  font-size:1.0625rem;
}

.upload-hint {
  color:var(--text-secondary);
  font-size:0.9375rem;
  font-weight:400;
}

.selected-files {
  margin-top:1.5rem;
  padding:1.25rem;
  background:var(--primary-50);
  border-radius:12px;
  border:1px solid var(--primary-200);
}

.selected-files > p {
  font-weight:600;
  color:var(--primary-700);
  margin:0 0 0.75rem 0;
}

.file-list {
  list-style:none;
  padding:0;
  margin:0;
}

.file-list li {
  padding:0.375rem 0;
  color:var(--text-secondary);
  font-size:0.9375rem;
  border-bottom:1px solid var(--primary-100);
}

.file-list li:last-child {
  border-bottom:none;
}

.loading {
  display:flex;
  align-items:center;
  justify-content:center;
  min-height:50vh;
  font-size:1.125rem;
  color:var(--text-secondary);
}

/* Animations */
@keyframes fadeIn {
  from { opacity:0; }
  to { opacity:1; }
}

@keyframes slideUp {
  from {
    opacity:0;
    transform:translateY(20px) scale(0.95);
  }
  to {
    opacity:1;
    transform:translateY(0) scale(1);
  }
}

/* Responsive Design */
@media (max-width:768px) {
  .project-topbar {
    flex-wrap:wrap;
    padding:0.625rem 1rem;
  }
  
  .header-actions {
    width:100%;
    flex-wrap:wrap;
  }
  
  .header-actions .btn {
    flex:1;
    justify-content:center;
  }
  
  .content-wrapper {
    padding:0 1rem 1rem;
  }
  
  .stats-grid {
    grid-template-columns:1fr;
    gap:1rem;
  }
  
  .stat-card {
    padding:1.25rem;
  }
  
  .stat-icon {
    width:48px;
    height:48px;
  }
  
  .stat-value {
    font-size:1.75rem;
  }
  
  .grid-4 {
    grid-template-columns:1fr;
  }
  
  .modal {
    width:95%;
    margin:1rem;
  }
  
  .modal-body {
    padding:1.5rem;
  }
  
  .upload-area {
    padding:2rem 1rem;
  }
}

@media (min-width:769px) and (max-width: 1024px) {
  .stats-grid {
    grid-template-columns:repeat(2, 1fr);
  }
  
  .grid-4 {
    grid-template-columns:repeat(2, 1fr);
  }
}
</style>

<style scoped>
.images-toolbar {
  display:flex;
  flex-wrap:wrap;
  align-items:center;
  justify-content:space-between;
  gap:0.75rem;
  margin-bottom:1rem;
}

.filter-chips {
  display:flex;
  flex-wrap:wrap;
  gap:0.375rem;
}

.filter-chip {
  padding:0.3125rem 0.75rem;
  border:1px solid var(--border-color);
  border-radius:9999px;
  background:var(--surface);
  color:var(--text-secondary);
  font-size:0.75rem;
  cursor:pointer;
  transition:all 0.15s ease;
}

.filter-chip:hover {
  border-color:var(--primary-400);
}

.filter-chip.active {
  background:var(--primary-600);
  border-color:var(--primary-600);
  color:var(--text);
}

.pager {
  display:flex;
  align-items:center;
  gap:0.5rem;
}

.pager-label {
  font-size:0.8125rem;
  color:var(--text-secondary);
  white-space:nowrap;
}

.upload-progress {
  position:relative;
  height:22px;
  margin-top:1rem;
  border-radius:9999px;
  background:var(--bg-subtle);
  overflow:hidden;
}

.upload-progress-bar {
  height:100%;
  background:var(--primary-500);
  transition:width 0.2s ease;
}

.upload-progress span {
  position:absolute;
  inset:0;
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:0.6875rem;
  font-weight:600;
  color:var(--text-primary);
}

.image-preview {
  position:relative;
}

.image-delete {
  position:absolute;
  top:0.375rem;
  right:0.375rem;
  display:flex;
  align-items:center;
  justify-content:center;
  width:24px;
  height:24px;
  border:none;
  border-radius:6px;
  background:rgba(5, 6, 10, 0.72);
  color:var(--text);
  cursor:pointer;
  opacity:0;
  transition:opacity 0.15s ease, background 0.15s ease;
}

.image-card:hover .image-delete {
  opacity:1;
}

.image-delete:hover {
  background:var(--danger);
}
</style>

<style scoped>
.autolabel-section {
  margin-bottom: 1.25rem;
  padding: 1.25rem;
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  background: var(--grad-surface);
  position: relative;
  overflow: hidden;
}

/* A thin accent edge marks this as the assisted path rather than a plain
   panel, without adding another coloured block to the page. */
.autolabel-section::before {
  content: '';
  position: absolute;
  inset: 0 auto 0 0;
  width: 3px;
  background: var(--grad-accent);
}

.section-tip {
  margin-left: auto;
  font-size: var(--fs-xs);
  font-weight: 400;
  color: var(--text-3);
}

.autolabel-count {
  font-size: var(--fs-lg);
  font-weight: 650;
  color: var(--accent-hover);
  font-variant-numeric: tabular-nums;
}

.autolabel-progress { margin: 0.875rem 0; }

.autolabel-live {
  margin-top: 0.5rem;
  font-size: var(--fs-sm);
  color: var(--text-2);
}

.autolabel-done {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  margin: 0.875rem 0;
  padding: 0.6875rem 0.875rem;
  border: 1px solid rgba(52, 211, 153, 0.3);
  border-radius: var(--r);
  background: var(--success-soft);
  color: #6ee7b7;
  font-size: var(--fs-sm);
  line-height: 1.5;
}
</style>

<style scoped>
.images-missing {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  margin-bottom: 1.25rem;
  padding: 1rem 1.125rem;
  border: 1px solid rgba(251, 191, 36, 0.35);
  border-radius: var(--r-md);
  background: var(--warning-soft);
  color: var(--warning);
}

.images-missing strong {
  display: block;
  margin-bottom: 0.25rem;
  color: var(--text);
}

.images-missing p {
  font-size: var(--fs-sm);
  line-height: 1.6;
  color: var(--text-2);
}

.images-missing code {
  padding: 0.0625rem 0.3125rem;
  border-radius: var(--r-sm);
  background: var(--bg);
  color: var(--text);
  font-size: 0.9em;
}
</style>

<style scoped>
.filter-groups {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 0.375rem;
}

.filter-group {
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
}

.filter-group-head {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  font-size: var(--fs-xs);
  font-weight: 600;
  color: var(--text-2);
}

.filter-group-items {
  font-size: var(--fs-xs);
  line-height: 1.6;
  color: var(--text-3);
  overflow-wrap: anywhere;
}
</style>

<style scoped>
.danger-link {
  display:inline-flex;
  align-items:center;
  gap:0.35rem;
  color:var(--danger-600);
}

.roi-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.4375rem;
  padding: 0.3125rem 0.6875rem;
  border: 1px solid var(--border-strong);
  border-radius: var(--r-full);
  background: var(--surface-2);
  font-size: var(--fs-xs);
  color: var(--text-2);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}

.roi-toggle:hover {
  border-color: var(--accent-muted);
  color: var(--text);
}

.roi-toggle input {
  width: 14px;
  height: 14px;
}

/* The preview is a fixed-height frame so a mixed-aspect-ratio gallery still
   lines up; the thumbnail letterboxes inside it. */
.image-preview {
  position: relative;
  height: 168px;
  border-radius: var(--r) var(--r) 0 0;
  overflow: hidden;
  background: var(--bg);
}
</style>

<style scoped>
.images-missing-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.75rem;
}

.images-missing-result {
  font-size: var(--fs-sm);
  color: var(--text-2);
}
</style>

<style scoped>
.images-expected {
  margin-left: 0.5rem;
  font-size: var(--fs-sm);
  font-weight: 500;
  color: var(--warning);
}

.gallery-missing {
  gap: 0.75rem;
  border: 1px dashed rgba(251, 191, 36, 0.35);
  border-radius: var(--r-md);
  background: var(--warning-soft);
}

.gallery-missing .empty-icon { color: var(--warning); }
.gallery-missing h3 { color: var(--text); }

.gallery-missing p {
  max-width: 64ch;
  color: var(--text-2);
  line-height: 1.6;
}

.missing-path {
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--border-strong);
  border-radius: var(--r-sm);
  background: var(--bg);
  color: var(--text);
  font-size: var(--fs-sm);
  overflow-wrap: anywhere;
}

.gallery-missing code {
  padding: 0.0625rem 0.3125rem;
  border-radius: var(--r-sm);
  background: var(--bg);
  color: var(--text);
  font-size: 0.9em;
}

.gallery-missing-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.625rem;
  margin-top: 0.25rem;
}

.gallery-missing-result {
  font-size: var(--fs-sm);
  color: var(--text-2);
}
</style>

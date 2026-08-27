<template>
  <div class="train-view">

    <div v-if="loading" class="loading">Loading project information...</div>

    <div v-else class="train-container">
      <!-- ── Dataset Summary ───────────────────────────────────────── -->
      <div class="summary-card">
        <h3 class="card-title">
          <Icon name="bar-chart" size="sm" />
          Dataset Summary
        </h3>
        
        <div class="summary-grid">
          <div class="summary-item item-primary">
            <div class="summary-icon">
              <Icon name="image" size="lg" />
            </div>
            <div class="summary-info">
              <div class="summary-value">{{ projectInfo?.total_images || 0 }}</div>
              <div class="summary-label">Total Images</div>
            </div>
          </div>
          
          <div class="summary-item item-secondary">
            <div class="summary-icon">
              <Icon name="check-circle" size="lg" />
            </div>
            <div class="summary-info">
              <div class="summary-value">{{ projectInfo?.annotated_images || 0 }}</div>
              <div class="summary-label">Annotated</div>
            </div>
          </div>
          
          <div class="summary-item item-tertiary">
            <div class="summary-icon">
              <Icon name="tag" size="lg" />
            </div>
            <div class="summary-info">
              <div class="summary-value">{{ projectInfo?.total_annotations || 0 }}</div>
              <div class="summary-label">Total Annotations</div>
            </div>
          </div>
          
          <div class="summary-item item-quaternary">
            <div class="summary-icon">
              <Icon name="layers" size="lg" />
            </div>
            <div class="summary-info">
              <div class="summary-value">{{ Object.keys(projectInfo?.tags || {}).length }}</div>
              <div class="summary-label">Classes</div>
            </div>
          </div>
        </div>

        <div v-if="(projectInfo?.annotated_images || 0) === 0" class="warning-message">
          <Icon name="alert-triangle" size="sm" />
          <span>No annotated images found. Please annotate images before training.</span>
        </div>
      </div>

      <!-- ── Dataset Readiness Validation ───────────────────────────── -->
      <div v-if="datasetSummary" class="validation-card">
        <h3 class="card-title">
          <Icon name="check-circle" size="sm" />
          Dataset Readiness
        </h3>

        <!-- Readiness Score Meter -->
        <div class="readiness-meter">
          <div class="score-circle" :class="getReadinessClass()">
            <div class="score-value">{{ Math.round(datasetSummary.readiness_score) }}%</div>
            <div class="score-label">Ready</div>
          </div>
          <div class="readiness-info">
            <div class="readiness-status" :class="getReadinessClass()">
              {{ getReadinessLabel() }}
            </div>
            <div class="readiness-desc">
              {{ getReadinessDescription() }}
            </div>
          </div>
        </div>

        <!-- Tags Breakdown -->
        <div v-if="datasetSummary.tags && Object.keys(datasetSummary.tags).length > 0" class="tags-breakdown">
          <h4>Per-Class Statistics</h4>
          <div class="tags-table">
            <div class="tags-header">
              <span class="col-tag">Tag</span>
              <span class="col-images">Images</span>
              <span class="col-boxes">Boxes</span>
              <span class="col-status">Status</span>
            </div>
            <div
              v-for="(stats, tag) in datasetSummary.tags"
              :key="tag"
              class="tag-row"
            >
              <span class="col-tag">
                <span class="tag-badge">{{ tag }}</span>
              </span>
              <span class="col-images">{{ stats.images }}</span>
              <span class="col-boxes">{{ stats.boxes }}</span>
              <span class="col-status">
                <span class="status-badge" :class="getTagStatusClass(stats.images)">
                  {{ getTagStatus(stats.images) }}
                </span>
              </span>
            </div>
          </div>
        </div>

        <!-- Warnings -->
        <div v-if="datasetSummary.warnings && datasetSummary.warnings.length > 0" class="validation-warnings">
          <div class="warning-header">
            <Icon name="alert-triangle" size="sm" />
            <span>Issues Found</span>
          </div>
          <ul class="warnings-list">
            <li v-for="(warn, i) in datasetSummary.warnings" :key="i">{{ warn }}</li>
          </ul>
        </div>

        <!-- Recommendations -->
        <div v-if="datasetSummary.recommendations && datasetSummary.recommendations.length > 0" class="validation-recommendations">
          <div class="recommendations-list">
            <div v-for="(rec, i) in datasetSummary.recommendations" :key="i" class="recommendation-item">
              {{ rec }}
            </div>
          </div>
        </div>
      </div>

      <!-- Training Configuration -->
      <div class="config-card">
        <h3 class="card-title">
          <Icon name="settings" size="sm" />
          Training Configuration
        </h3>
        
        <form @submit.prevent="startTraining">
          <div class="form-group">
            <label class="form-label">Model Name (Optional)</label>
            <input
              v-model="trainingConfig.model_name"
              type="text"
              class="form-input"
              placeholder="e.g., my-model-v1 (leave empty for auto-generated)"
            />
            <p class="form-hint">
              Custom name for your model. If not specified, will use: {{ projectName }}_{{ new Date().toISOString().split('T')[0] }}
            </p>
          </div>

          <div class="form-group">
            <label class="form-label">Model Architecture</label>
            <select v-model="trainingConfig.model_type" class="form-select" @change="onModelTypeChange">
              <optgroup label="YOLO — แนะนำ (เร็วกว่าราว 20 เท่า)">
                <option value="yolo11n">YOLO11n — Nano (เร็วที่สุด, RAM น้อย)</option>
                <option value="yolo11s">YOLO11s — Small (เร็ว, แนะนำสำหรับตัวเลข)</option>
                <option value="yolo11m">YOLO11m — Medium (สมดุล)</option>
                <option value="yolo11l">YOLO11l — Large (accuracy สูง)</option>
                <option value="yolov8n">YOLOv8n — Nano</option>
                <option value="yolov8s">YOLOv8s — Small</option>
                <option value="yolov8m">YOLOv8m — Medium</option>
              </optgroup>
              <optgroup label="PyTorch Faster R-CNN — ช้ากว่ามาก">
                <option value="faster_rcnn">Faster R-CNN ResNet50-FPN</option>
              </optgroup>
            </select>
            <p class="form-hint model-note" v-if="isYoloModel">
              <Icon name="zap" size="xs" />
              <span>
                ราว 5-15 นาทีต่อ epoch บน GPU — export เป็น
                <strong>.pt</strong> แล้วแปลงเป็น blob สำหรับกล้อง Luxonis ได้
              </span>
            </p>
            <p class="form-hint model-note is-warning" v-else>
              <Icon name="alert-triangle" size="xs" />
              <span>
                ช้ากว่ามาก ราว 1.7 ชั่วโมงต่อ epoch — 100 epoch ใช้เวลาราว 170 ชั่วโมง
              </span>
            </p>
          </div>

          <div class="form-row">
            <div class="form-group">
              <label class="form-label">Epochs</label>
              <input
                v-model.number="trainingConfig.epochs"
                type="number"
                min="1"
                max="500"
                class="form-input"
              />
              <p class="form-hint">Number of training iterations (default: 100)</p>
            </div>

            <div class="form-group">
              <label class="form-label">Batch Size</label>
              <input
                v-model.number="trainingConfig.batch_size"
                type="number"
                min="1"
                max="64"
                class="form-input"
              />
              <p class="form-hint">Images per batch (default: 16)</p>
            </div>
          </div>

          <div class="form-group">
            <label class="form-label">Image Size</label>
            <select v-model.number="trainingConfig.img_size" class="form-select">
              <option :value="320">320x320 (Fastest)</option>
              <option :value="416">416x416</option>
              <option :value="640">640x640 (Recommended)</option>
              <option :value="1280">1280x1280 (Best Quality)</option>
            </select>
          </div>

          <div class="form-group">
            <label class="form-label">Export Formats</label>
            <div class="export-formats-grid">
              <label v-for="fmt in availableExportFormats" :key="fmt.value" class="fmt-check">
                <input
                  type="checkbox"
                  :value="fmt.value"
                  v-model="trainingConfig.export_formats"
                />
                <span class="fmt-label">
                  <strong>{{ fmt.label }}</strong>
                  <span class="fmt-desc">{{ fmt.desc }}</span>
                </span>
              </label>
            </div>
            <p class="form-hint">
              Select at least one. Every selected format is exported once
              training finishes; a failed export does not lose the weights.
            </p>
          </div>

          <p v-if="startError" class="error-message">{{ startError }}</p>

          <div v-if="datasetReport" class="dataset-report">
            Prepared
            <strong>{{ datasetReport.train_images }}</strong> training and
            <strong>{{ datasetReport.val_images }}</strong> validation images
            ({{ datasetReport.train_boxes }} / {{ datasetReport.val_boxes }} boxes).
            <span v-if="datasetReport.empty_classes?.length" class="report-warn">
              No boxes for: {{ datasetReport.empty_classes.join(', ') }}.
            </span>
            <span v-if="datasetReport.skipped_count" class="report-warn">
              {{ datasetReport.skipped_count }} image(s) skipped.
            </span>
          </div>

          <div class="form-actions">
            <button
              type="submit"
              class="btn btn-success btn-lg"
              :disabled="training || isActive || !canStartTraining"
            >
              <Icon name="zap" size="sm" />
              {{ training || isActive ? 'Training in progress…' : 'Start training' }}
            </button>
          </div>
        </form>
      </div>

      <p v-if="statusError" class="error-message">{{ statusError }}</p>

      <!-- ── Training Status ───────────────────────────────────────── -->
      <div v-if="trainingStatus" class="status-card">
        <div class="status-card-header">
          <h3 class="card-title" style="margin-bottom:0;border-bottom:none;padding-bottom:0">
            <Icon name="activity" size="sm" />
            Training Status
          </h3>
          <button
            v-if="isActive"
            @click="stopTraining"
            class="btn btn-danger btn-sm"
          >
            <Icon name="x" size="sm" />
            Stop
          </button>
          <button
            v-if="['failed', 'stopped', 'running', 'preparing'].includes(trainingStatus.status)"
            @click="resetTraining"
            class="btn btn-secondary btn-sm"
            style="margin-left:0.5rem"
          >
            <Icon name="refresh-cw" size="sm" />
            Reset
          </button>
        </div>

        <!-- Progress Bar -->
        <div v-if="isActive" class="progress-section">
          <div class="progress-header">
            <span class="progress-label">Epoch {{ trainingStatus.current_epoch }} / {{ trainingStatus.total_epochs }}</span>
            <span class="progress-pct">{{ progressPercent }}%</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" :style="{ width: progressPercent + '%' }"></div>
          </div>
          <!-- Live metrics -->
          <div v-if="displayMetrics.length" class="metrics-row">
            <div v-for="metric in displayMetrics" :key="metric.key" class="metric-chip">
              <span class="metric-key">{{ metric.label }}</span>
              <span class="metric-val">{{ metric.value }}</span>
            </div>
          </div>
        </div>

        <!-- Completed: show exported formats only -->
        <div v-if="trainingStatus.status === 'completed'" class="complete-banner">
          <Icon name="check-circle" size="sm" />
          Training completed at {{ formatDate(trainingStatus.completed_at) }}
          <div class="export-download-row">
            <template v-if="trainingStatus.exported_models && Object.keys(trainingStatus.exported_models).length">
              <a
                v-for="(path, fmt) in trainingStatus.exported_models"
                :key="fmt"
                v-show="path"
                :href="trainingService.downloadUrl(projectName, path)"
                class="btn btn-primary btn-sm"
              >
                <Icon name="download" size="sm" /> {{ fmtLabel(fmt) }}
              </a>
            </template>
            <span v-else class="text-muted" style="font-size:0.85rem">ไม่มีไฟล์ export (ดู log)</span>
          </div>
        </div>

        <!-- Stopped banner -->
        <div v-if="trainingStatus.status === 'stopped'" class="stopped-banner">
          <Icon name="alert-triangle" size="sm" />
          Training was stopped after
          {{ trainingStatus.current_epoch }} of {{ trainingStatus.total_epochs }} epochs.
          Weights written so far are still available below.
        </div>

        <!-- Error banner -->
        <div v-if="trainingStatus.status === 'failed'" class="error-banner">
          <Icon name="alert-triangle" size="sm" />
          {{ trainingStatus.error || 'Training failed. See the log below.' }}
        </div>

        <div class="status-info" style="margin-top:1rem">
          <div class="status-item">
            <span class="status-label">Status:</span>
            <span :class="['status-badge', `status-${trainingStatus.status}`]">
              {{ trainingStatus.status }}
            </span>
          </div>
          <div class="status-item">
            <span class="status-label">Model:</span>
            <span>{{ trainingStatus.model_type }}</span>
          </div>
          <div class="status-item">
            <span class="status-label">Epochs:</span>
            <span>{{ trainingStatus.epochs }}</span>
          </div>
          <div class="status-item">
            <span class="status-label">Train / Val images:</span>
            <span>{{ trainingStatus.train_images }} / {{ trainingStatus.val_images }}</span>
          </div>
          <div class="status-item">
            <span class="status-label">Started:</span>
            <span>{{ formatDate(trainingStatus.started_at) }}</span>
          </div>
        </div>

        <div v-if="!isActive && displayMetrics.length" class="metrics-row final-metrics">
          <div v-for="metric in displayMetrics" :key="metric.key" class="metric-chip">
            <span class="metric-key">{{ metric.label }}</span>
            <span class="metric-val">{{ metric.value }}</span>
          </div>
        </div>

        <!-- Per-class accuracy: one overall mAP hides which class is weak -->
        <div v-if="perClassRows.length" class="per-class">
          <div class="per-class-header">
            <h4>Accuracy per class</h4>
            <span class="per-class-hint">
              sorted worst first — these are the classes to add images for
            </span>
          </div>
          <div class="per-class-list">
            <div v-for="row in perClassRows" :key="row.name" class="per-class-row">
              <span class="per-class-name">{{ row.name }}</span>
              <div class="per-class-track">
                <div
                  class="per-class-fill"
                  :class="row.band"
                  :style="{ width: row.percent + '%' }"
                ></div>
              </div>
              <span class="per-class-value" :class="row.band">{{ row.display }}</span>
            </div>
          </div>
          <p v-if="weakestClasses.length" class="per-class-advice">
            Weakest: <strong>{{ weakestClasses.join(', ') }}</strong>.
            More annotated examples of these will help more than more epochs.
          </p>
        </div>

        <div class="status-classes">
          <h4>Classes</h4>
          <div class="classes-tags">
            <span v-for="cls in trainingStatus.classes" :key="cls" class="badge badge-primary">
              {{ cls }}
            </span>
          </div>
        </div>

        <!-- Log viewer -->
        <div class="log-section">
          <div class="log-header">
            <h4>Training Log</h4>
            <button @click="loadLogs" class="btn btn-secondary btn-sm">Refresh</button>
          </div>
          <div ref="logBox" class="log-box">
            <div v-if="!trainingLogs.length" class="log-empty">No logs yet...</div>
            <div v-for="(line, i) in trainingLogs" :key="i" class="log-line">{{ line }}</div>
          </div>
        </div>
      </div>

      <!-- ── Trained Models ─────────────────────────────────────────── -->
      <div v-if="trainedModels.length" class="status-card">
        <h3 class="card-title">
          <Icon name="box" size="sm" />
          Trained Models
        </h3>
        <div class="models-list">
          <div v-for="m in trainedModels" :key="m.path" class="model-item">
            <div class="model-info">
              <span class="model-name">{{ m.name }}</span>
              <span class="model-meta">{{ m.size_mb }} MB &bull; {{ formatDate(m.modified) }}</span>
            </div>
            <a :href="trainingService.downloadUrl(projectName, m.path)"
               class="btn btn-secondary btn-sm">
              <Icon name="download" size="sm" /> Download
            </a>
          </div>
        </div>
      </div>

      <!-- ── Tips ──────────────────────────────────────────────────── -->
      <div class="tips-card">
        <h3 class="card-title">
          <Icon name="lightbulb" size="sm" />
          Training Tips
        </h3>
        <ul class="tips-list">
          <li><strong>More data is better:</strong> At least 100+ images per class for good results</li>
          <li><strong>Balanced dataset:</strong> Each class should have a similar number of examples</li>
          <li><strong>Quality over quantity:</strong> Accurate annotations matter more than volume</li>
          <li><strong>Start small:</strong> Use YOLOv8n or YOLOv8s for faster iterations</li>
          <li><strong>Save dataset:</strong> Export your dataset as ZIP to reuse or continue training later</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { errorMessage, projectService, trainingService } from '@/services'
import Icon from '@/components/Icon.vue'
import { formatDateTime as formatDate, formatMetric } from '@/utils/format'

const route = useRoute()
const projectName = computed(() => route.params.name)

const loading = ref(true)
const training = ref(false)
const projectInfo = ref(null)
const datasetSummary = ref(null)
const trainingStatus = ref(null)
const trainingLogs = ref([])
const trainedModels = ref([])
const logBox = ref(null)

const statusError = ref(null)
const startError = ref(null)
const modelOptions = ref(null)
const datasetReport = ref(null)

// Statuses that mean a worker process is still alive.
const ACTIVE_STATUSES = ['preparing', 'running', 'stopping']
const POLL_INTERVAL_MS = 3000

let pollTimer = null
const pollFailures = ref(0)

const trainingConfig = ref({
  model_name: '',
  model_type: 'yolo11s',
  epochs: 100,
  batch_size: 16,
  img_size: 640,
  export_formats: ['onnx']
})

const isYoloModel = computed(() => trainingConfig.value.model_type !== 'faster_rcnn')

function onModelTypeChange() {
  if (isYoloModel.value) {
    trainingConfig.value.batch_size = 16
    trainingConfig.value.epochs = 100
  } else {
    // Faster R-CNN is far heavier per image.
    trainingConfig.value.batch_size = 4
    trainingConfig.value.epochs = 30
  }

  // Drop any selection the new family cannot produce, otherwise the request
  // is rejected with a format error the user did not knowingly choose.
  const allowed = new Set(availableExportFormats.value.map((f) => f.value))
  trainingConfig.value.export_formats =
    trainingConfig.value.export_formats.filter((f) => allowed.has(f))
  if (!trainingConfig.value.export_formats.length) {
    trainingConfig.value.export_formats = ['onnx']
  }
}

const EXPORT_FORMATS = {
  pt:          { label: 'PyTorch (.pt)',       desc: 'The trained weights' },
  onnx:        { label: 'ONNX',                desc: 'Cross-platform inference' },
  torchscript: { label: 'TorchScript',         desc: 'Optimised PyTorch' },
  blob:        { label: 'DepthAI (.blob)',     desc: 'Luxonis OAK device' }
}

// Faster R-CNN always writes its weights as a .pth state_dict; a .pt would be
// a full pickle that needs this source tree to load and that the model tester
// cannot open, so it is not offered.
const FORMATS_BY_FAMILY = {
  yolo: ['pt', 'onnx', 'torchscript', 'blob'],
  faster_rcnn: ['onnx', 'torchscript', 'blob']
}

const availableExportFormats = computed(() =>
  FORMATS_BY_FAMILY[isYoloModel.value ? 'yolo' : 'faster_rcnn']
    .map((value) => ({ value, ...EXPORT_FORMATS[value] }))
)

const canStartTraining = computed(() => {
  const hasAnnotations = projectInfo.value?.annotated_images > 0 && 
                         projectInfo.value?.total_annotations > 0
  const hasGoodReadiness = datasetSummary.value?.readiness_score >= 40
  const hasExportFormat = trainingConfig.value.export_formats.length > 0
  return hasAnnotations && hasGoodReadiness && hasExportFormat
})

const isActive = computed(() =>
  ACTIVE_STATUSES.includes(trainingStatus.value?.status)
)

// Metrics arrive as a flat dict of raw trainer keys. Only the ones that mean
// something to a user are shown, in a fixed order, with 0-1 scores rendered
// as percentages.
const METRIC_LABELS = [
  ['mAP50', 'mAP@50', 'percent'],
  ['mAP50_95', 'mAP@50-95', 'percent'],
  ['precision', 'Precision', 'percent'],
  ['recall', 'Recall', 'percent'],
  ['train_loss', 'Train loss', 'number'],
  ['val_loss', 'Val loss', 'number'],
  ['train_box_loss', 'Box loss', 'number'],
  ['train_cls_loss', 'Class loss', 'number'],
  ['lr', 'Learning rate', 'number']
]

const displayMetrics = computed(() => {
  const metrics = trainingStatus.value?.metrics || {}
  return METRIC_LABELS
    .filter(([key]) => metrics[key] !== undefined && metrics[key] !== null)
    .map(([key, label, kind]) => ({
      key,
      label,
      value: kind === 'percent' ? formatMetric(metrics[key]) : String(metrics[key])
    }))
})

/**
 * Per-class AP rows, worst first.
 *
 * A single mAP number cannot tell a ten-digit detector apart from one that
 * reads nine digits perfectly and never sees an 8. This is the view that
 * turns "accuracy is 72%" into a concrete next action.
 */
const perClassRows = computed(() => {
  const perClass = trainingStatus.value?.per_class || {}
  return Object.entries(perClass)
    .map(([name, stats]) => {
      const ap = stats?.ap50
      const value = typeof ap === 'number' ? ap : null
      return {
        name,
        value,
        percent: value === null ? 0 : Math.max(0, Math.min(100, value * 100)),
        display: value === null ? '—' : formatMetric(value),
        band: value === null ? 'unknown'
          : value >= 0.75 ? 'good'
          : value >= 0.5 ? 'fair'
          : 'poor'
      }
    })
    .sort((a, b) => (a.value ?? 1) - (b.value ?? 1))
})

const weakestClasses = computed(() =>
  perClassRows.value.filter((row) => row.band === 'poor').map((row) => row.name)
)

const progressPercent = computed(() => {
  if (!trainingStatus.value) return 0
  const { current_epoch, total_epochs } = trainingStatus.value
  if (!total_epochs) return 0
  return Math.round((current_epoch / total_epochs) * 100)
})

onMounted(async () => {
  await loadProjectInfo()
  await loadDatasetSummary()
  await loadTrainingStatus()
  await loadModels()
})

onUnmounted(() => { stopPolling() })

// ── data loaders ────────────────────────────────────────────────────
const loadProjectInfo = async () => {
  loading.value = true
  try {
    const res = await projectService.get(projectName.value)
    if (res.success) projectInfo.value = res.project
  } catch (e) {
    console.error(e)
  } finally {
    loading.value = false
  }
}

const loadDatasetSummary = async () => {
  try {
    const res = await projectService.datasetSummary(projectName.value)
    if (res.success) {
      datasetSummary.value = res
    }
  } catch (e) {
    console.error('Error loading dataset summary:', e)
  }
}

const loadTrainingStatus = async () => {
  try {
    const res = await trainingService.status(projectName.value)
    trainingStatus.value = res.status
    if (ACTIVE_STATUSES.includes(res.status?.status)) {
      training.value = true
      startPolling()
    }
  } catch (e) {
    statusError.value = errorMessage(e)
    trainingStatus.value = null
  }
}

const loadLogs = async () => {
  try {
    const res = await trainingService.logs(projectName.value)
    if (res.success) {
      trainingLogs.value = res.logs
      await nextTick()
      if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight
    }
  } catch { /* ignore */ }
}

const loadModels = async () => {
  try {
    const res = await trainingService.models(projectName.value)
    if (res.success) trainedModels.value = res.models
  } catch { /* ignore */ }
}

// ── polling ──────────────────────────────────────────────────────────
const startPolling = () => {
  if (pollTimer) return
  pollFailures.value = 0
  pollTimer = setInterval(async () => {
    try {
      const res = await trainingService.status(projectName.value)
      trainingStatus.value = res.status
      await loadLogs()
      if (!ACTIVE_STATUSES.includes(res.status?.status)) {
        // The run reached a terminal state: pick up the produced weights and
        // stop polling.
        stopPolling()
        await Promise.all([loadModels(), loadDatasetSummary()])
        training.value = false
      }
    } catch (e) {
      // Keep polling through a transient network blip; only a persistent
      // failure is worth surfacing.
      pollFailures.value += 1
      if (pollFailures.value >= 5) {
        statusError.value = errorMessage(e)
        stopPolling()
      }
    }
  }, POLL_INTERVAL_MS)
}

const stopPolling = () => {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

// ── actions ──────────────────────────────────────────────────────────
const startTraining = async () => {
  if (!canStartTraining.value) {
    if (!projectInfo.value?.annotated_images) {
      alert('Please annotate images before training.')
    } else if (datasetSummary.value?.readiness_score < 40) {
      alert(`Dataset readiness is ${Math.round(datasetSummary.value.readiness_score)}%. You need at least 40% readiness to start training. Please add more annotations or balance your classes.`)
    } else {
      alert('Please complete dataset preparation before training.')
    }
    return
  }
  const label = trainingConfig.value.model_type
  if (!confirm(`Start training ${label} for ${trainingConfig.value.epochs} epochs? ` +
               'This runs in the background and can take a long time.')) return

  startError.value = null
  training.value = true
  try {
    const res = await trainingService.start(projectName.value, trainingConfig.value)
    trainingStatus.value = res.config
    datasetReport.value = res.dataset
    startPolling()
    await loadLogs()
  } catch (e) {
    startError.value = errorMessage(e)
    training.value = false
  }
}

const stopTraining = async () => {
  if (!confirm('Stop training? Weights saved so far are kept.')) return
  try {
    await trainingService.stop(projectName.value)
    stopPolling()
    training.value = false
    await Promise.all([loadTrainingStatus(), loadLogs(), loadModels()])
  } catch (e) {
    statusError.value = errorMessage(e)
  }
}

const resetTraining = async () => {
  if (!confirm('Reset the run status to idle? Use this only if a run is stuck.')) return
  statusError.value = null
  try {
    await trainingService.reset(projectName.value)
    stopPolling()
    training.value = false
    await loadTrainingStatus()
  } catch (e) {
    statusError.value = errorMessage(e)
  }
}


const fmtLabel = (fmt) => ({
  pth: 'PyTorch (.pth)',
  pt: 'PyTorch (.pt)',
  onnx: 'ONNX (.onnx)',
  torchscript: 'TorchScript (.torchscript)',
  openvino: 'OpenVINO',
  engine: 'TensorRT (.engine)',
  tflite: 'TensorFlow Lite',
  blob: 'DepthAI Blob (.blob)'
}[fmt] || String(fmt).toUpperCase())

// ── readiness helpers ────────────────────────────────────────────────
const getReadinessClass = () => {
  if (!datasetSummary.value) return 'low'
  const score = datasetSummary.value.readiness_score
  if (score >= 70) return 'high'
  if (score >= 40) return 'medium'
  return 'low'
}

const getReadinessLabel = () => {
  if (!datasetSummary.value) return 'Unknown'
  const score = datasetSummary.value.readiness_score
  if (score >= 70) return 'Ready for Training'
  if (score >= 40) return 'Partially Ready'
  return 'Not Ready'
}

const getReadinessDescription = () => {
  if (!datasetSummary.value) return ''
  const score = datasetSummary.value.readiness_score
  if (score >= 70) return 'Your dataset has sufficient quality and quantity for training.'
  if (score >= 40) return 'Training is possible, but accuracy may be limited.'
  return 'Your dataset needs more annotations before training.'
}

const getTagStatus = (imageCount) => {
  if (imageCount >= 30) return 'Good'
  if (imageCount >= 10) return 'OK'
  return 'Low'
}

const getTagStatusClass = (imageCount) => {
  if (imageCount >= 30) return 'good'
  if (imageCount >= 10) return 'ok'
  return 'low'
}
</script>

<style scoped>
.train-view {
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
  max-width:1200px;
  margin:0 auto;
  padding:0 2rem;
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
}

.header-left {
  flex:1;
  display:flex;
  flex-direction:column;
  gap:1rem;
}

.btn-back {
  display:inline-flex;
  align-items:center;
  gap:0.5rem;
  padding:0.5rem 1rem;
  background:transparent;
  border:1px solid var(--border-color);
  color:var(--text-secondary);
  border-radius:8px;
  font-weight:500;
  font-size:0.875rem;
  cursor:pointer;
  transition:all 0.2s ease;
  align-self:flex-start;
}

.btn-back:hover {
  background:var(--gray-50);
  color:var(--text-primary);
  border-color:var(--gray-300);
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

/* Train Container */
.train-container {
  padding:0 2rem 2rem;
  max-width:1200px;
  margin:0 auto;
  display:flex;
  flex-direction:column;
  gap:1.5rem;
}

/* Cards */
.summary-card,
.validation-card,
.config-card,
.status-card,
.tips-card {
  background:var(--surface);
  padding:2rem;
  border-radius:16px;
  box-shadow: var(--shadow-sm);
  border:1px solid var(--border);
}

.card-title {
  font-size:1.375rem;
  font-weight:600;
  color:var(--text-primary);
  margin:0 0 1.5rem 0;
  display:flex;
  align-items:center;
  gap:0.625rem;
  padding-bottom:1rem;
  border-bottom:2px solid var(--border-color);
}

/* Summary Grid */
.summary-grid {
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(248px, 1fr));
  gap:1rem;
  margin-bottom:1.5rem;
  align-items: stretch;
}

.summary-item {
  display:flex;
  align-items:center;
  gap:1.25rem;
  padding:1.5rem;
  background:var(--gray-50);
  border-radius:12px;
  transition:all 0.3s;
  border:2px solid transparent;
}

.summary-item:hover {
  transform:translateY(-4px);
  box-shadow: var(--shadow);
}

.summary-item.item-primary {
  background:var(--grad-surface);
  border-color:var(--primary-200);
}

.summary-item.item-secondary {
  background:var(--grad-surface);
  border-color:var(--purple-200);
}

.summary-item.item-tertiary {
  background:var(--grad-surface);
  border-color:var(--gray-200);
}

.summary-item.item-quaternary {
  background:var(--grad-surface);
  border-color:var(--border);
}

.summary-icon {
  width:56px;
  height:56px;
  display:flex;
  align-items:center;
  justify-content:center;
  border-radius:var(--radius-lg);
  flex-shrink:0;
  box-shadow: var(--shadow);
}

.summary-item.item-primary .summary-icon {
  background:linear-gradient(135deg, var(--primary-500), var(--primary-600));
  color:var(--text);
}

.summary-item.item-secondary .summary-icon {
  background:linear-gradient(135deg, var(--purple-500), var(--purple-600));
  color:var(--text);
}

.summary-item.item-tertiary .summary-icon {
  background:linear-gradient(135deg, var(--gray-500), var(--gray-600));
  color:var(--text);
}

.summary-item.item-quaternary .summary-icon {
  background:var(--grad-accent-2);
  color:var(--text);
}

.summary-value {
  font-size:2rem;
  font-weight:700;
  line-height:1;
  margin-bottom:0.375rem;
  color:var(--text-primary);
}

.summary-label {
  color: var(--text-secondary);
  font-size: var(--fs-sm);
  font-weight: 500;
  /* Kept on one line: at the 16px base "Total Annotations" wrapped and left
     the four cards at different heights. */
  white-space: nowrap;
}

/* Classes Section */
.classes-section {
  margin-top:1.75rem;
  padding-top:1.75rem;
  border-top:2px solid var(--border-color);
}

.classes-section h4 {
  font-size:1.125rem;
  font-weight:600;
  color:var(--text-primary);
  margin-bottom:1.25rem;
}

.classes-list {
  display:flex;
  flex-direction:column;
  gap:1rem;
}

.class-item {
  display:grid;
  grid-template-columns:1fr auto;
  gap:0.75rem;
  align-items:center;
}

.class-name {
  font-weight:500;
  color:var(--text-primary);
  font-size:1rem;
}

.class-count {
  font-weight:700;
  color:var(--primary-600);
  min-width:50px;
  text-align:right;
  font-size:1.125rem;
}

.class-bar {
  grid-column:1 / -1;
  height:10px;
  background:var(--bg-subtle);
  border-radius:9999px;
  overflow:hidden;
}

.class-bar-fill {
  height:100%;
  background:linear-gradient(90deg, var(--primary-500), var(--primary-600));
  border-radius:9999px;
  transition:width 0.5s ease;
  box-shadow: var(--shadow);
}

.warning-message {
  background:linear-gradient(135deg, var(--warning-50), var(--warning-100));
  color:var(--warning-800);
  padding:1.25rem;
  border-radius:var(--radius-lg);
  border:2px solid var(--warning-300);
  margin-top:1.5rem;
  display:flex;
  align-items:center;
  gap:0.75rem;
  font-weight:500;
}

/* Form */
.form-group {
  margin-bottom:1.5rem;
}

.form-label {
  display:block;
  margin-bottom:0.625rem;
  font-weight:600;
  color:var(--text-primary);
  font-size:0.9375rem;
}

.form-select,
.form-input {
  width:100%;
  padding:0.75rem 1rem;
  border:2px solid var(--border-color);
  border-radius:var(--radius-md);
  font-size:1rem;
  transition:all 0.2s;
  font-family:inherit;
}

.form-select:focus,
.form-input:focus {
  outline:none;
  border-color:var(--primary-500);
  box-shadow: var(--shadow);
}

.form-row {
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:1.25rem;
}

.form-hint {
  margin-top:0.5rem;
  font-size:0.875rem;
  color:var(--text-secondary);
  line-height:1.4;
}

.form-actions {
  margin-top:2rem;
  padding-top:2rem;
  border-top:2px solid var(--border-color);
}

.btn-lg {
  padding:1rem 2rem;
  font-size:1.125rem;
  width:100%;
  display:flex;
  align-items:center;
  justify-content:center;
  gap:0.625rem;
  font-weight:600;
}

.btn-lg:disabled {
  opacity:0.6;
  cursor:not-allowed;
}

/* Status */
.status-info {
  display:flex;
  flex-direction:column;
  gap:1rem;
  margin-bottom:1.75rem;
}

.status-item {
  display:flex;
  justify-content:space-between;
  align-items:center;
  padding:1rem 1.25rem;
  background:var(--gray-50);
  border-radius:var(--radius-lg);
  border:1px solid var(--border-color);
}

.status-label {
  font-weight:600;
  color:var(--text-secondary);
  font-size:0.9375rem;
}

.status-item > span:last-child {
  font-weight:600;
  color:var(--text-primary);
}

.status-badge {
  padding:0.5rem 1rem;
  border-radius:9999px;
  font-weight:600;
  font-size:0.9375rem;
  text-transform:uppercase;
  letter-spacing:0.5px;
}

.status-pending {
  background:linear-gradient(135deg, var(--warning-100), var(--warning-200));
  color:var(--warning-700);
  border:1px solid var(--warning-300);
}

.status-training {
  background:linear-gradient(135deg, var(--primary-100), var(--primary-200));
  color:var(--primary-700);
  border:1px solid var(--primary-300);
}

.status-completed {
  background:linear-gradient(135deg, var(--success-100), var(--success-200));
  color:var(--success-700);
  border:1px solid var(--success-300);
}

.status-classes {
  margin-bottom:1.75rem;
  padding-bottom:1.75rem;
  border-bottom:2px solid var(--border-color);
}

.status-classes h4 {
  font-size:1.0625rem;
  font-weight:600;
  color:var(--text-primary);
  margin-bottom:1rem;
}

.classes-tags {
  display:flex;
  flex-wrap:wrap;
  gap:0.625rem;
}

.badge {
  padding:0.5rem 1rem;
  border-radius:9999px;
  font-size:0.875rem;
  font-weight:500;
  border:1px solid;
}

.badge-primary {
  background:linear-gradient(135deg, var(--primary-100), var(--primary-200));
  color:var(--primary-700);
  border-color:var(--primary-300);
}

.status-note {
  background:var(--grad-surface));
  padding:1.5rem;
  border-radius:var(--radius-lg);
  border:2px solid var(--primary-200);
  display:flex;
  gap:1rem;
  align-items:flex-start;
}

.status-note > div {
  flex:1;
}

.status-note p {
  margin:0 0 1rem 0;
  color:var(--text-primary);
  line-height:1.6;
}

.status-note strong {
  color:var(--primary-700);
}

.status-note code {
  display:block;
  background:var(--surface);
  padding:0.75rem 1rem;
  border-radius:var(--radius-md);
  font-family:monospace;
  font-size:0.875rem;
  color:var(--primary-700);
  border:1px solid var(--primary-200);
  word-break:break-all;
}

/* Tips */
.tips-list {
  list-style:none;
  padding:0;
  margin:0;
}

.tips-list li {
  padding:1rem 0;
  border-bottom:1px solid var(--border-color);
  color:var(--text-secondary);
  line-height:1.7;
  font-size:0.9375rem;
}

.tips-list li:last-child {
  border-bottom:none;
  padding-bottom:0;
}

.tips-list strong {
  color:var(--text-primary);
  font-weight:600;
}

/* ── Validation Card Styles ────────────────────────────────────────── */
.readiness-meter {
  display:flex;
  gap:2rem;
  align-items:center;
  padding:1.5rem;
  background:var(--grad-surface) 100%);
  border-radius:12px;
  margin-bottom:2rem;
}

.score-circle {
  width:100px;
  height:100px;
  border-radius:50%;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  border:4px solid;
  flex-shrink:0;
}

.score-circle.high {
  background:linear-gradient(135deg, var(--success-50) 0%, var(--success-100) 100%);
  border-color:var(--success-500);
  color:var(--success-700);
}

.score-circle.medium {
  background:linear-gradient(135deg, var(--warning-50) 0%, var(--warning-100) 100%);
  border-color:var(--warning-500);
  color:var(--warning-700);
}

.score-circle.low {
  background:linear-gradient(135deg, var(--danger-50) 0%, var(--danger-100) 100%);
  border-color:var(--danger-500);
  color:var(--danger-700);
}

.score-value {
  font-size:2rem;
  font-weight:700;
  line-height:1;
}

.score-label {
  font-size:0.75rem;
  text-transform:uppercase;
  letter-spacing:0.05em;
  margin-top:0.25rem;
}

.readiness-info {
  flex:1;
}

.readiness-status {
  font-size:1.25rem;
  font-weight:600;
  margin-bottom:0.5rem;
}

.readiness-status.high { color:var(--success-600); }
.readiness-status.medium { color:var(--warning-600); }
.readiness-status.low { color:var(--danger-600); }

.readiness-desc {
  font-size:0.9375rem;
  color:var(--text-secondary);
  line-height:1.6;
}

.tags-breakdown {
  margin-top:1.5rem;
}

.tags-breakdown h4 {
  font-size:1rem;
  font-weight:600;
  margin:0 0 1rem 0;
  color:var(--text-primary);
}

.tags-table {
  border:1px solid var(--border-color);
  border-radius:8px;
  overflow:hidden;
}

.tags-header,
.tag-row {
  display:grid;
  grid-template-columns:2fr 1fr 1fr 1fr;
  gap:1rem;
  padding:0.75rem 1rem;
}

.tags-header {
  background:var(--gray-50);
  font-weight:600;
  font-size:0.875rem;
  color:var(--text-secondary);
  text-transform:uppercase;
  letter-spacing:0.03em;
}

.tag-row {
  border-top:1px solid var(--border-color);
  align-items:center;
}

.tag-row:hover {
  background:var(--gray-50);
}

.tag-badge {
  display:inline-block;
  padding:0.25rem 0.75rem;
  background:var(--primary-100);
  color:var(--primary-700);
  border-radius:6px;
  font-weight:500;
  font-size:0.875rem;
}

.status-badge {
  display:inline-block;
  padding:0.25rem 0.75rem;
  border-radius:6px;
  font-weight:500;
  font-size:0.8125rem;
  text-align:center;
}

.status-badge.good {
  background:var(--success-100);
  color:var(--success-700);
}

.status-badge.ok {
  background:var(--warning-100);
  color:var(--warning-700);
}

.status-badge.low {
  background:var(--danger-100);
  color:var(--danger-700);
}

.validation-warnings {
  margin-top:1.5rem;
  padding:1rem;
  background:var(--warning-50);
  border-left:4px solid var(--warning-500);
  border-radius:8px;
}

.warning-header {
  display:flex;
  align-items:center;
  gap:0.5rem;
  font-weight:600;
  color:var(--warning-700);
  margin-bottom:0.75rem;
}

.warnings-list {
  list-style:none;
  padding:0;
  margin:0;
}

.warnings-list li {
  padding:0.5rem 0;
  color:var(--warning-700);
  font-size:0.9375rem;
  line-height:1.5;
}

.validation-recommendations {
  margin-top:1.5rem;
}

.recommendations-list {
  display:flex;
  flex-direction:column;
  gap:0.75rem;
}

.recommendation-item {
  padding:1rem;
  background:var(--gray-50);
  border-radius:8px;
  font-size:0.9375rem;
  color:var(--text-primary);
  line-height:1.6;
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
  from {
    opacity:0;
  }
  to {
    opacity:1;
  }
}

@keyframes fadeInUp {
  from {
    opacity:0;
    transform:translateY(20px);
  }
  to {
    opacity:1;
    transform:translateY(0);
  }
}

@keyframes float {
  0%, 100% {
    transform:translateY(0) rotate(0deg);
  }
  50% {
    transform:translateY(-20px) rotate(180deg);
  }
}

/* Responsive Design */
@media (max-width:768px) {
  .gradient-header {
    padding:1.5rem 1rem 2rem;
  }
  
  .header-text h1 {
    font-size:1.75rem;
  }
  
  .header-text p {
    font-size:1rem;
  }
  
  .train-container {
    padding:1rem;
    gap:1.25rem;
  }
  
  .summary-card,
  .config-card,
  .status-card,
  .tips-card {
    padding:1.5rem;
  }
  
  .summary-grid {
    grid-template-columns:1fr;
    gap:1rem;
  }
  
  .summary-item {
    padding:1.25rem;
  }
  
  .summary-icon {
    width:48px;
    height:48px;
  }
  
  .summary-value {
    font-size:1.75rem;
  }
  
  .form-row {
    grid-template-columns:1fr;
    gap:1rem;
  }
  
  .card-title {
    font-size:1.125rem;
  }
}

@media (min-width:769px) and (max-width: 1024px) {
  .summary-grid {
    grid-template-columns:repeat(2, 1fr);
  }
}

/* ── Status card header with stop button ── */
.status-card-header {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:1.5rem;
  padding-bottom:1rem;
  border-bottom:2px solid var(--border-color);
}

/* ── Progress bar ── */
.progress-section {
  margin-bottom:1.5rem;
  padding:1.25rem;
  background:var(--gray-50);
  border-radius:12px;
  border:1px solid var(--border-color);
}

.progress-header {
  display:flex;
  justify-content:space-between;
  margin-bottom:0.75rem;
}

.progress-label { font-weight:600; color:var(--text-primary); }
.progress-pct   { font-weight:700; color:var(--primary-600); }

.progress-bar-bg {
  height:12px;
  background:var(--bg-subtle);
  border-radius:9999px;
  overflow:hidden;
  margin-bottom:1rem;
}

.progress-bar-fill {
  height:100%;
  background:linear-gradient(90deg, var(--accent), var(--accent-muted));
  border-radius:9999px;
  transition:width 0.5s ease;
  box-shadow: var(--shadow);
}

.metrics-row {
  display:flex;
  flex-wrap:wrap;
  gap:0.5rem;
}

.metric-chip {
  display:flex;
  gap:0.375rem;
  padding:0.25rem 0.75rem;
  background:var(--surface);
  border:1px solid var(--border-color);
  border-radius:9999px;
  font-size:0.8125rem;
}

.metric-key { color:var(--text-secondary); }
.metric-val { font-weight:700; color:var(--primary-700); }

/* ── Banners ── */
.complete-banner {
  display:flex;
  flex-direction:column;
  gap:0.75rem;
  padding:1rem 1.25rem;
  background:linear-gradient(135deg, var(--success-50), var(--success-100));
  border:1px solid var(--success-300);
  border-radius:10px;
  color:var(--success-800);
  font-weight:500;
  margin-bottom:1.25rem;
}

.export-download-row {
  display:flex;
  gap:0.5rem;
  flex-wrap:wrap;
}

/* ── Export formats checkboxes ── */
.export-formats-grid {
  display:grid;
  grid-template-columns:repeat(auto-fill, minmax(220px, 1fr));
  gap:0.5rem;
  margin-top:0.5rem;
}

.fmt-check {
  display:flex;
  align-items:flex-start;
  gap:0.5rem;
  padding:0.625rem 0.875rem;
  background:var(--gray-50);
  border:1px solid var(--border-color);
  border-radius:8px;
  cursor:pointer;
  transition:border-color 0.2s, background 0.2s;
}

.fmt-check:has(input:checked) {
  border-color:var(--primary-400);
  background:var(--primary-50);
}

.fmt-check input {
  margin-top:2px;
  accent-color:var(--primary-600);
}

.fmt-label {
  display:flex;
  flex-direction:column;
  gap:0.125rem;
}

.fmt-label strong {
  font-size:0.875rem;
  color:var(--text-primary);
}

.fmt-desc {
  font-size:0.75rem;
  color:var(--text-secondary);
}

.error-banner {
  display:flex;
  align-items:flex-start;
  gap:0.75rem;
  padding:1rem 1.25rem;
  background:linear-gradient(135deg, var(--danger-50), var(--danger-100));
  border:1px solid var(--danger-300);
  border-radius:10px;
  color:var(--danger-800);
  font-weight:500;
  margin-bottom:1.25rem;
}

/* ── Status stopped badge ── */
.status-stopped {
  background:linear-gradient(135deg, var(--gray-100), var(--gray-200));
  color:var(--gray-700);
  border:1px solid var(--gray-300);
}

.status-running {
  background:linear-gradient(135deg, var(--accent-soft), var(--accent-soft));
  color:var(--accent-muted);
  border:1px solid var(--accent-hover);
  animation:pulse-badge 1.5s ease-in-out infinite;
}

@keyframes pulse-badge {
  0%, 100% { opacity:1; }
  50%       { opacity:0.65; }
}

/* ── Log viewer ── */
.log-section { margin-top:1.5rem; }

.log-header {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:0.75rem;
}

.log-header h4 {
  font-size:1rem;
  font-weight:600;
  color:var(--text-primary);
  margin:0;
}

.log-box {
  background:var(--bg);
  border-radius:10px;
  padding:1rem;
  max-height:260px;
  overflow-y:auto;
  font-family:'Courier New', monospace;
  font-size:0.8125rem;
}

.log-line {
  color:var(--text);
  line-height:1.7;
  white-space:pre-wrap;
  word-break:break-all;
}

.log-empty {
  color:var(--text-3);
  font-style:italic;
}

/* ── Models list ── */
.models-list {
  display:flex;
  flex-direction:column;
  gap:0.75rem;
}

.model-item {
  display:flex;
  justify-content:space-between;
  align-items:center;
  padding:1rem 1.25rem;
  background:var(--gray-50);
  border-radius:10px;
  border:1px solid var(--border-color);
}

.model-info {
  display:flex;
  flex-direction:column;
  gap:0.25rem;
}

.model-name {
  font-weight:600;
  color:var(--text-primary);
}

.model-meta {
  font-size:0.8125rem;
  color:var(--text-secondary);
}

.btn-sm {
  padding:0.375rem 0.875rem;
  font-size:0.8125rem;
}

.btn-danger {
  background:var(--danger-500);
  color:var(--text);
  border:none;
}

.btn-danger:hover {
  background:var(--danger-600);
}
</style>

<style scoped>
.dataset-report {
  margin:0 0 1rem;
  padding:0.75rem 1rem;
  border-radius:8px;
  background:var(--primary-50);
  color:var(--text-secondary);
  font-size:0.8125rem;
  line-height:1.6;
}

.report-warn {
  display:block;
  color:var(--warning);
}

.stopped-banner {
  display:flex;
  align-items:center;
  gap:0.5rem;
  padding:0.75rem 1rem;
  border-radius:8px;
  background:var(--warning-soft);
  color:var(--warning);
  font-size:0.875rem;
}

.final-metrics {
  margin-top:1rem;
}
</style>

<style scoped>
.per-class {
  margin-top: 1.25rem;
  padding: 1rem;
  border: 1px solid var(--border);
  border-radius: var(--r);
  background: var(--bg-subtle);
}

.per-class-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 0.875rem;
}

.per-class-header h4 {
  font-size: var(--fs-base);
  font-weight: 600;
}

.per-class-hint {
  font-size: var(--fs-xs);
  color: var(--text-3);
}

.per-class-list {
  display: flex;
  flex-direction: column;
  gap: 0.4375rem;
}

.per-class-row {
  display: grid;
  grid-template-columns: minmax(56px, 88px) 1fr 56px;
  align-items: center;
  gap: 0.75rem;
}

.per-class-name {
  font-size: var(--fs-sm);
  font-weight: 600;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.per-class-track {
  height: 8px;
  border-radius: var(--r-full);
  background: var(--surface-3);
  overflow: hidden;
}

.per-class-fill {
  height: 100%;
  border-radius: var(--r-full);
  transition: width 0.45s var(--ease-out);
}

/* Colour carries the same meaning as the number, so the row is readable at a
   glance without reading every figure. */
.per-class-fill.good { background: var(--grad-success); }
.per-class-fill.fair { background: linear-gradient(90deg, var(--warning), var(--amber)); }
.per-class-fill.poor { background: var(--grad-danger); }
.per-class-fill.unknown { background: var(--border-strong); }

.per-class-value {
  font-size: var(--fs-sm);
  font-weight: 600;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.per-class-value.good { color: var(--success); }
.per-class-value.fair { color: var(--warning); }
.per-class-value.poor { color: var(--danger); }
.per-class-value.unknown { color: var(--text-3); }

.per-class-advice {
  margin-top: 0.875rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border);
  font-size: var(--fs-sm);
  color: var(--text-2);
}
</style>

<style scoped>
.model-note {
  display: flex;
  align-items: flex-start;
  gap: 0.375rem;
}

.model-note :deep(svg) {
  margin-top: 0.2em;
  flex-shrink: 0;
  color: var(--accent-hover);
}

.model-note.is-warning :deep(svg) {
  color: var(--warning);
}
</style>

<template>
  <div class="model-test-view">

    <!-- ── Page header ─────────────────────────────────────── -->
    <header class="page-header">
      <div class="header-left">
        <div class="header-icon-wrap">
          <Icon name="zap" size="md" />
        </div>
        <div>
          <h1 class="page-title">Model Test Tool</h1>
          <p class="page-sub">ทดสอบโมเดลที่เทรนแล้วโดยการ import รูปและดูผล ROI</p>
        </div>
      </div>
      <div v-if="inferenceSummary" class="summary-chips">
        <span class="chip chip-purple">
          <Icon name="box" size="sm" /> {{ inferenceSummary.model_name }}
        </span>
        <span class="chip chip-blue">
          <Icon name="server" size="sm" /> {{ inferenceSummary.device }}
        </span>
        <span class="chip chip-green">
          <Icon name="image" size="sm" /> {{ inferenceSummary.total }} รูป
        </span>
      </div>
    </header>

    <!-- ── Main grid ───────────────────────────────────────── -->
    <div class="main-grid">

      <!-- ── Left: Config panel ── -->
      <aside class="config-card">

        <!-- Step 1: model -->
        <div class="step-block">
          <div class="step-head">
            <div class="step-num">1</div>
            <span>เลือกโมเดล</span>
          </div>
          <label
            class="drop-zone"
            :class="{ 'drop-zone--active': modelFile }"
            @dragover.prevent
            @drop.prevent="onModelDrop"
          >
            <input ref="modelInput" type="file" accept=".pth,.pt,.onnx,.torchscript" class="hidden-input" @change="onModelChange" />
            <div v-if="!modelFile" class="dz-inner">
              <Icon name="upload" size="lg" />
              <span class="dz-title">คลิก หรือ ลากไฟล์โมเดลมาวาง</span>
              <span class="dz-hint">.pth &bull; .pt &bull; .onnx &bull; .torchscript</span>
            </div>
            <div v-else class="dz-chosen">
              <div class="chosen-icon"><Icon name="box" size="md" /></div>
              <div class="chosen-info">
                <div class="chosen-name">{{ modelFile.name }}</div>
                <div class="chosen-size">{{ (modelFile.size / 1048576).toFixed(1) }} MB</div>
              </div>
              <button class="chosen-remove" @click.prevent="modelFile = null">
                <Icon name="x" size="sm" />
              </button>
            </div>
          </label>
          <div class="format-tags">
            <span v-for="fmt in MODEL_FORMATS" :key="fmt.ext"
              class="fmt-tag" :class="isActiveFormat(fmt.ext) ? 'fmt-tag--active' : ''">
              {{ fmt.label }}
            </span>
          </div>

          <div v-if="trainedModels.length" class="trained-picker">
            <div class="trained-head">
              <Icon name="box" size="sm" />
              <span>หรือเลือกโมเดลที่เทรนไว้แล้วในระบบ</span>
            </div>
            <ul class="trained-list">
              <li v-for="model in trainedModels" :key="model.path">
                <button
                  type="button"
                  class="trained-item"
                  :class="{ 'trained-item--on': pickedModel?.path === model.path }"
                  @click="pickTrained(model)"
                >
                  <span class="trained-name">{{ model.name }}</span>
                  <span class="trained-meta">
                    {{ model.project }} / {{ model.run }} &bull;
                    {{ model.format }} &bull; {{ model.size_mb }} MB
                  </span>
                </button>
              </li>
            </ul>
          </div>
        </div>

        <!-- Step 2: label names -->
        <div class="step-block">
          <div class="step-head">
            <div class="step-num">2</div>
            <span>Label Names <span class="optional">(optional)</span></span>
          </div>
          <div v-if="resolvedLabelSource" class="label-hint">
            ใช้ชื่อคลาสจากโมเดลอัตโนมัติ: {{ resolvedLabelSource }}
          </div>
          <textarea
            v-model="labelNames"
            class="field-textarea"
            rows="3"
            placeholder="ปล่อยว่างได้ ระบบจะอ้างอิงจากโมเดลถ้าพบข้อมูล&#10;หรือกรอกเอง เช่น: bottle, cap, label"
          />
        </div>

        <!-- Step 3: threshold -->
        <div class="step-block">
          <div class="step-head">
            <div class="step-num">3</div>
            <span>Score Threshold</span>
            <span class="threshold-val">{{ scoreThreshold.toFixed(2) }}</span>
          </div>
          <div class="slider-wrap">
            <span class="sl-lo">0.10</span>
            <input v-model.number="scoreThreshold" type="range" min="0.1" max="0.95" step="0.05" class="styled-slider" />
            <span class="sl-hi">0.95</span>
          </div>
          <div class="threshold-bar">
            <div class="threshold-fill" :style="{ width: thresholdPct + '%' }"></div>
          </div>
        </div>

        <!-- Step 4: images -->
        <div class="step-block">
          <div class="step-head">
            <div class="step-num">4</div>
            <span>เลือกรูปที่จะทดสอบ</span>
          </div>
          <label
            class="drop-zone"
            :class="{ 'drop-zone--active': imageFiles.length }"
            @dragover.prevent
            @drop.prevent="onImagesDrop"
          >
            <input ref="imagesInput" type="file" accept="image/*" multiple class="hidden-input" @change="onImagesChange" />
            <div v-if="!imageFiles.length" class="dz-inner">
              <Icon name="image" size="lg" />
              <span class="dz-title">คลิก หรือ ลากรูปมาวาง</span>
              <span class="dz-hint">jpg / png / bmp &bull; หลายรูปได้</span>
            </div>
            <div v-else class="image-thumbs">
              <div v-for="(f, i) in imageFiles.slice(0, 8)" :key="i" class="thumb-wrap">
                <img :src="previewUrls[i]" class="thumb" />
                <span class="thumb-name">{{ f.name }}</span>
              </div>
              <div v-if="imageFiles.length > 8" class="thumb-more">
                +{{ imageFiles.length - 8 }} รูป
              </div>
            </div>
          </label>
          <div v-if="imageFiles.length" class="img-count-row">
            <Icon name="image" size="sm" />
            <span>เลือกแล้ว <strong>{{ imageFiles.length }}</strong> ไฟล์</span>
            <button class="link-btn" @click="clearImages">ล้าง</button>
          </div>
        </div>

        <!-- Run button -->
        <button class="run-btn" :class="{ 'run-btn--loading': loading }" :disabled="!canTest || loading" @click="runTest">
          <span v-if="!loading" class="run-inner">
            <Icon name="zap" size="sm" />
            ทดสอบโมเดล
          </span>
          <span v-else class="run-inner">
            <span class="spinner"></span>
            กำลังประมวลผล…
          </span>
        </button>

        <div v-if="error" class="error-box">
          <Icon name="x" size="sm" />
          {{ error }}
        </div>
      </aside>

      <!-- ── Right: Results ── -->
      <main class="result-area">

        <!-- Empty state -->
        <div v-if="!results.length && !loading" class="empty-state">
          <div class="empty-icon">
            <Icon name="search" size="xl" />
          </div>
          <h3>ยังไม่มีผลลัพธ์</h3>
          <p>เลือกโมเดล, เลือกรูป แล้วกด <strong>ทดสอบโมเดล</strong></p>
          <div class="empty-steps">
            <div v-for="(s, i) in EMPTY_STEPS" :key="i" class="empty-step">
              <span class="e-num">{{ i + 1 }}</span>
              <span>{{ s }}</span>
            </div>
          </div>
        </div>

        <!-- Loading skeleton -->
        <div v-else-if="loading" class="skeleton-grid">
          <div v-for="i in imageFiles.length || 3" :key="i" class="skeleton-card">
            <div class="sk sk-img"></div>
            <div class="sk sk-line"></div>
            <div class="sk sk-line sk-line--short"></div>
          </div>
        </div>

        <!-- Results grid -->
        <div v-else>
          <div class="results-summary-bar">
            <span class="rs-label">ผลลัพธ์ {{ results.length }} รูป</span>
            <span class="rs-det">
              พบวัตถุรวม <strong>{{ totalDetections }}</strong> รายการ
              (threshold ≥ {{ scoreThreshold.toFixed(2) }})
            </span>
          </div>
          <div class="results-grid">
            <article v-for="item in results" :key="item.filename" class="result-card">

              <div class="rc-header">
                <div class="rc-filename" :title="item.filename">
                  <Icon name="image" size="sm" />
                  {{ item.filename }}
                </div>
                <span class="rc-badge" :class="item.detection_count ? 'rc-badge--found' : 'rc-badge--none'">
                  {{ item.detection_count }} ROI
                </span>
              </div>

              <div class="rc-img-wrap">
                <img :src="item.annotated_image" :alt="item.filename" class="rc-img" />
              </div>

              <div v-if="item.detections.length" class="rc-table">
                <div class="rct-head">
                  <span>Label</span><span>Score</span><span>Bounding Box</span>
                </div>
                <div v-for="(d, idx) in item.detections" :key="idx" class="rct-row">
                  <span class="rct-label">{{ d.label_name }}</span>
                  <span class="rct-score">{{ d.score }}</span>
                  <span class="rct-box">{{ d.box.join(', ') }}</span>
                </div>
              </div>
              <div v-else class="rc-none">
                ไม่พบวัตถุที่ผ่าน threshold
              </div>

            </article>
          </div>
        </div>

      </main>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import Icon from '@/components/Icon.vue'
import { errorMessage, trainingService } from '@/services'

// ── Constants ───────────────────────────────────────────────────────
const MODEL_FORMATS = [
  { ext: '.pth',         label: '.pth — PyTorch weights' },
  { ext: '.pt',          label: '.pt — YOLO weights' },
  { ext: '.onnx',        label: '.onnx — ONNX Runtime' },
  { ext: '.torchscript', label: '.torchscript — TorchScript' },
]

const EMPTY_STEPS = [
  'Import โมเดล (.pth / .pt / .onnx / .torchscript)',
  'ระบุ Label Names (ถ้าทราบ) แยกด้วย comma',
  'ปรับ Score Threshold ตามต้องการ',
  'เลือกรูปที่จะทดสอบ (หลายรูปได้)',
  'กด ทดสอบโมเดล',
]

// ── State ────────────────────────────────────────────────────────────
const modelFile     = ref(null)
const modelInput    = ref(null)
// Models this server has already produced. Uploading best.pt back to the
// machine that just wrote it was the only way to try a freshly trained model.
const trainedModels = ref([])
const pickedModel   = ref(null)
const imageFiles    = ref([])
const imagesInput   = ref(null)
const previewUrls   = ref([])
const labelNames    = ref('')
const scoreThreshold = ref(0.5)
const loading       = ref(false)
const error         = ref('')
const results       = ref([])
const inferenceSummary = ref(null)
const resolvedLabelSource = ref('')

// ── Computed ─────────────────────────────────────────────────────────
const chosenModel   = computed(() => pickedModel.value || modelFile.value)
const canTest       = computed(() => !!chosenModel.value && imageFiles.value.length > 0)
const thresholdPct  = computed(() => ((scoreThreshold.value - 0.1) / 0.85) * 100)
const totalDetections = computed(() => results.value.reduce((s, r) => s + r.detection_count, 0))

const isActiveFormat = (ext) => {
  const name = pickedModel.value?.name || modelFile.value?.name
  return name ? name.toLowerCase().endsWith(ext) : false
}

// ── File helpers ──────────────────────────────────────────────────────
const resetModelDerivedState = () => {
  resolvedLabelSource.value = ''
  error.value = ''
  results.value = []
}

const onModelChange = (e) => {
  modelFile.value = e.target.files?.[0] || null
  pickedModel.value = null
  resetModelDerivedState()
}
const onModelDrop   = (e) => {
  modelFile.value = e.dataTransfer.files?.[0] || null
  pickedModel.value = null
  resetModelDerivedState()
}
onMounted(async () => {
  // A failure here is not worth an error banner: the upload path still works,
  // the list is only a shortcut.
  try {
    trainedModels.value = await trainingService.listTrainedModels()
  } catch {
    trainedModels.value = []
  }
})

const pickTrained = (model) => {
  pickedModel.value = pickedModel.value?.path === model.path ? null : model
  modelFile.value = null
  resetModelDerivedState()
}

const buildPreviews = (files) => {
  previewUrls.value.forEach(u => URL.revokeObjectURL(u))
  previewUrls.value = Array.from(files).slice(0, 8).map(f => URL.createObjectURL(f))
}

const onImagesChange = (e) => {
  imageFiles.value = Array.from(e.target.files || [])
  buildPreviews(imageFiles.value)
}
const onImagesDrop = (e) => {
  imageFiles.value = Array.from(e.dataTransfer.files || []).filter(f => f.type.startsWith('image/'))
  buildPreviews(imageFiles.value)
}
const clearImages = () => {
  imageFiles.value = []
  previewUrls.value.forEach((url) => URL.revokeObjectURL(url))
  previewUrls.value = []
  if (imagesInput.value) imagesInput.value.value = ''
}

// Object URLs are held by the browser until revoked, so they are released
// when the page goes away.
onBeforeUnmount(() => {
  previewUrls.value.forEach((url) => URL.revokeObjectURL(url))
})

// ── Run inference ────────────────────────────────────────────────────
const runTest = async () => {
  if (!canTest.value || loading.value) return
  loading.value = true
  error.value   = ''
  results.value = []

  try {
    const res = await trainingService.testModel(
      chosenModel.value,
      imageFiles.value,
      { scoreThreshold: scoreThreshold.value, labelNames: labelNames.value }
    )
    inferenceSummary.value = {
      model_name: res.model_name,
      model_format: res.model_format,
      device: res.device,
      total: res.total_images,
    }
    if (!labelNames.value && Array.isArray(res.resolved_label_names) && res.resolved_label_names.length) {
      labelNames.value = res.resolved_label_names.join(', ')
    }
    resolvedLabelSource.value = res.resolved_label_source || ''
    results.value = res.results || []
  } catch (err) {
    error.value = errorMessage(err, 'เกิดข้อผิดพลาดในการทดสอบโมเดล')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
/* ── Shell ──────────────────────────────────────────────────────────── */
.model-test-view {
  padding:1.25rem 1.5rem;
  min-height:calc(100vh - 60px);
  background:var(--surface-2);
}

/* ── Page header ─────────────────────────────────────────────────────── */
.page-header {
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom:1.25rem;
  gap:1rem;
  flex-wrap:wrap;
}
.header-left {
  display:flex;
  align-items:center;
  gap:1rem;
}
.header-icon-wrap {
  width:48px;
  height:48px;
  border-radius:14px;
  background:linear-gradient(135deg, var(--primary-500, var(--accent-hover)), var(--primary-700, var(--accent)));
  display:flex;
  align-items:center;
  justify-content:center;
  color:var(--text);
  flex-shrink:0;
}
.page-title { font-size:1.3rem; margin:0; }
.page-sub   { font-size:0.82rem; color:var(--text-secondary, var(--text-3)); margin:0; }

.summary-chips { display:flex; gap:0.5rem; flex-wrap:wrap; }
.chip {
  font-size:0.76rem;
  font-weight:600;
  padding:0.28rem 0.65rem;
  border-radius:999px;
  display:flex;
  align-items:center;
  gap:0.3rem;
}
.chip-purple { background:var(--primary-100, var(--accent-soft)); color:var(--primary-700, var(--accent)); }
.chip-blue   { background:var(--info-soft); color:var(--info); }
.chip-green  { background:var(--success-soft); color:var(--success); }

/* ── Main grid ───────────────────────────────────────────────────────── */
.main-grid {
  display:grid;
  grid-template-columns:380px 1fr;
  gap:1.1rem;
  align-items:start;
}

/* ── Config card ─────────────────────────────────────────────────────── */
.config-card {
  background:var(--surface);
  border-radius:18px;
  border:1px solid var(--border-color, var(--border));
  box-shadow: var(--shadow);
  padding:1.25rem;
  display:flex;
  flex-direction:column;
  gap:1.1rem;
}

/* Step block */
.step-block { display:flex; flex-direction:column; gap:0.55rem; }
.step-head {
  display:flex;
  align-items:center;
  gap:0.55rem;
  font-weight:600;
  font-size:0.9rem;
  color:var(--text-primary, var(--text));
}
.step-num {
  width:24px; height:24px;
  border-radius:50%;
  background:var(--primary-600, var(--accent));
  color:var(--text);
  font-size:0.72rem;
  font-weight:700;
  display:flex;
  align-items:center;
  justify-content:center;
  flex-shrink:0;
}
.optional { font-weight:400; color:var(--text-tertiary, var(--text-3)); font-size:0.78rem; }
.threshold-val {
  margin-left:auto;
  background:var(--primary-50, var(--accent-softer));
  color:var(--primary-700, var(--accent));
  padding:0.15rem 0.55rem;
  border-radius:999px;
  font-size:0.8rem;
  font-weight:700;
}

/* Drop zone */
.drop-zone {
  display:block;
  border:2px dashed var(--border-color, var(--border));
  border-radius:14px;
  cursor:pointer;
  transition:border-color 0.2s, background 0.2s;
  overflow:hidden;
}
.drop-zone:hover,
.drop-zone--active { border-color:var(--primary-400, var(--accent-hover)); background:var(--primary-50, var(--accent-softer)); }
.dz-inner {
  padding:1.2rem;
  display:flex;
  flex-direction:column;
  align-items:center;
  gap:0.35rem;
  color:var(--text-secondary, var(--text-3));
}
.dz-title { font-weight:600; font-size:0.85rem; }
.dz-hint  { font-size:0.75rem; }
.hidden-input { display:none; }

/* Chosen model state */
.dz-chosen {
  display:flex;
  align-items:center;
  gap:0.75rem;
  padding:0.8rem;
}
.chosen-icon {
  width:40px; height:40px;
  border-radius:10px;
  background:var(--primary-100, var(--accent-soft));
  color:var(--primary-600, var(--accent));
  display:flex;
  align-items:center;
  justify-content:center;
  flex-shrink:0;
}
.chosen-info  { flex:1; min-width:0; }
.chosen-name  { font-weight:700; font-size:0.86rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.chosen-size  { font-size:0.75rem; color:var(--text-secondary, var(--text-3)); }
.chosen-remove {
  background:var(--danger-soft);
  border:0;
  border-radius:8px;
  color:var(--danger);
  width:30px; height:30px;
  display:flex;
  align-items:center;
  justify-content:center;
  cursor:pointer;
  flex-shrink:0;
}

/* Format tags */
.trained-picker {
  margin-top:1rem;
  padding-top:1rem;
  border-top:1px solid var(--border-color);
}

.trained-head {
  display:flex;
  align-items:center;
  gap:0.5rem;
  margin-bottom:0.6rem;
  color:var(--text-secondary);
  font-size:0.86rem;
}

.trained-list {
  display:flex;
  flex-direction:column;
  gap:0.4rem;
  margin:0;
  padding:0;
  max-height:15rem;
  overflow-y:auto;
  list-style:none;
}

.trained-item {
  display:flex;
  flex-direction:column;
  gap:0.15rem;
  width:100%;
  padding:0.55rem 0.75rem;
  border:1px solid var(--border-color);
  border-radius:var(--radius-md);
  background:var(--surface-2);
  color:var(--text-primary);
  font:inherit;
  text-align:left;
  cursor:pointer;
  transition:border-color var(--t) var(--ease), background var(--t) var(--ease);
}

.trained-item:hover {
  border-color:var(--accent);
}

.trained-item--on {
  border-color:var(--accent);
  background:var(--accent-soft);
}

.trained-name {
  font-size:0.9rem;
  font-weight:600;
}

.trained-meta {
  color:var(--text-secondary);
  font-size:0.78rem;
}

.format-tags { display:flex; flex-wrap:wrap; gap:0.35rem; }
.fmt-tag {
  font-size:0.7rem;
  padding:0.18rem 0.5rem;
  border-radius:6px;
  background:var(--gray-100, var(--surface-2));
  color:var(--text-secondary, var(--text-3));
  border:1px solid var(--border-color, var(--border));
}
.fmt-tag--active {
  background:var(--primary-100, var(--accent-soft));
  color:var(--primary-700, var(--accent));
  border-color:var(--primary-300, var(--accent-hover));
}

/* Textarea */
.field-textarea {
  width:100%;
  background: var(--bg-subtle);
  color: var(--text);
  font-family: inherit;
  border:1px solid var(--border-color, var(--border));
  border-radius:10px;
  padding:0.6rem 0.7rem;
  font-size:0.85rem;
  resize:vertical;
  font-family:inherit;
  color:var(--text-primary, var(--text));
}

.label-hint {
  font-size:0.76rem;
  color:var(--primary-700, var(--accent));
  background:var(--primary-50, var(--accent-softer));
  border:1px solid var(--primary-200, var(--accent-soft));
  border-radius:10px;
  padding:0.45rem 0.65rem;
}

/* Slider */
.slider-wrap {
  display:flex;
  align-items:center;
  gap:0.5rem;
}
.sl-lo, .sl-hi { font-size:0.72rem; color:var(--text-secondary, var(--text-3)); flex-shrink:0; }
.styled-slider { flex:1; accent-color:var(--primary-600, var(--accent)); }
.threshold-bar {
  height:4px;
  border-radius:99px;
  background:var(--border-color, var(--surface-3));
  overflow:hidden;
}
.threshold-fill {
  height:100%;
  background:var(--primary-500, var(--accent-hover));
  border-radius:99px;
  transition:width 0.15s;
}

/* Image thumbs */
.image-thumbs {
  display:flex;
  flex-wrap:wrap;
  gap:0.4rem;
  padding:0.6rem;
}
.thumb-wrap {
  width:68px;
  display:flex;
  flex-direction:column;
  align-items:center;
  gap:0.2rem;
}
.thumb {
  width:68px;
  height:52px;
  object-fit:cover;
  border-radius:6px;
  border:1px solid var(--border-color, var(--border));
}
.thumb-name {
  font-size:0.65rem;
  color:var(--text-secondary, var(--text-3));
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
  max-width:68px;
}
.thumb-more {
  width:68px; height:52px;
  border-radius:6px;
  background:var(--primary-100, var(--accent-soft));
  color:var(--primary-700, var(--accent));
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight:700;
  font-size:0.78rem;
}
.img-count-row {
  display:flex;
  align-items:center;
  gap:0.4rem;
  font-size:0.82rem;
  color:var(--text-secondary, var(--text-3));
}
.link-btn {
  background:none;
  border:0;
  color:var(--danger-600, var(--danger));
  font-size:0.8rem;
  cursor:pointer;
  text-decoration:underline;
  margin-left:auto;
}

/* Run button */
.run-btn {
  width:100%;
  height:46px;
  border:0;
  border-radius:12px;
  background:linear-gradient(120deg, var(--success-600, var(--success)), var(--success-700, var(--success)));
  color:var(--text);
  font-weight:700;
  font-size:0.95rem;
  cursor:pointer;
  box-shadow: var(--shadow);
  transition:opacity 0.2s, transform 0.1s;
}
.run-btn:hover:not(:disabled) { opacity:0.93; transform:translateY(-1px); }
.run-btn:disabled             { opacity:0.55; cursor:not-allowed; box-shadow: none; }
.run-inner {
  display:flex;
  align-items:center;
  justify-content:center;
  gap:0.45rem;
}
.spinner {
  width:16px; height:16px;
  border:2px solid var(--surface-2);
  border-top-color:var(--border-strong);
  border-radius:50%;
  animation:spin 0.7s linear infinite;
}
@keyframes spin { to { transform:rotate(360deg); } }

/* Error box */
.error-box {
  display:flex;
  align-items:flex-start;
  gap:0.5rem;
  background:var(--danger-50, var(--danger-soft));
  border:1px solid rgba(248, 113, 113, 0.35);
  color:var(--danger-700, var(--danger));
  border-radius:10px;
  padding:0.65rem 0.8rem;
  font-size:0.84rem;
}

/* ── Result area ─────────────────────────────────────────────────────── */
.result-area {
  background:var(--surface);
  border-radius:18px;
  border:1px solid var(--border-color, var(--border));
  box-shadow: var(--shadow);
  padding:1.25rem;
  min-height:400px;
}

/* Empty state */
.empty-state {
  min-height:380px;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  gap:0.7rem;
  color:var(--text-secondary, var(--text-3));
  text-align:center;
}
.empty-icon {
  width:72px; height:72px;
  border-radius:20px;
  background:var(--primary-50, var(--accent-softer));
  color:var(--primary-400, var(--accent-hover));
  display:flex;
  align-items:center;
  justify-content:center;
}
.empty-state h3 { color:var(--text-primary, var(--text)); margin:0; }
.empty-state p  { font-size:0.88rem; margin:0; }
.empty-steps {
  display:flex;
  flex-direction:column;
  gap:0.45rem;
  padding-top:0.5rem;
  text-align:left;
}
.empty-step {
  display:flex;
  align-items:center;
  gap:0.55rem;
  font-size:0.82rem;
}
.e-num {
  width:20px; height:20px;
  border-radius:50%;
  background:var(--primary-100, var(--accent-soft));
  color:var(--primary-700, var(--accent));
  font-size:0.7rem;
  font-weight:700;
  display:flex;
  align-items:center;
  justify-content:center;
  flex-shrink:0;
}

/* Skeleton */
.skeleton-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:1rem; }
.skeleton-card { border:1px solid var(--border-color, var(--border)); border-radius:14px; padding:0.8rem; }
.sk {
  border-radius:8px;
  background:linear-gradient(90deg, var(--surface-2) 25%, var(--surface-3) 50%, var(--surface-2) 75%);
  background-size:200% 100%;
  animation:shimmer 1.4s infinite;
}
.sk-img  { height:160px; margin-bottom:0.6rem; }
.sk-line { height:14px; margin-bottom:0.4rem; }
.sk-line--short { width:60%; }
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

/* Summary bar */
.results-summary-bar {
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding-bottom:0.85rem;
  margin-bottom:0.85rem;
  border-bottom:1px solid var(--border-color, var(--border));
}
.rs-label { font-weight:700; font-size:0.95rem; }
.rs-det   { font-size:0.84rem; color:var(--text-secondary, var(--text-3)); }

/* Results grid */
.results-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:1rem; }

.result-card {
  border:1px solid var(--border-color, var(--border));
  border-radius:14px;
  overflow:hidden;
  background:var(--bg-secondary, var(--bg-subtle));
  transition:box-shadow 0.2s;
}
.result-card:hover { box-shadow: var(--shadow-lg); }

.rc-header {
  display:flex;
  align-items:center;
  gap:0.5rem;
  padding:0.65rem 0.8rem;
  border-bottom:1px solid var(--border-color, var(--border));
  background:var(--surface);
}
.rc-filename {
  flex:1;
  font-size:0.82rem;
  font-weight:600;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
  display:flex;
  align-items:center;
  gap:0.35rem;
  color:var(--text-primary, var(--text));
}
.rc-badge {
  font-size:0.72rem;
  font-weight:700;
  padding:0.2rem 0.5rem;
  border-radius:999px;
}
.rc-badge--found { background:var(--success-soft); color:var(--success-700, var(--success)); }
.rc-badge--none  { background:var(--gray-100, var(--surface-2));   color:var(--text-secondary, var(--text-3)); }

.rc-img-wrap { position:relative; }
.rc-img { width:100%; display:block; aspect-ratio:4/3; object-fit:contain; background:var(--bg); }

.rc-table { padding:0.55rem 0.7rem; }
.rct-head {
  display:grid;
  grid-template-columns:1fr auto auto;
  gap:0.3rem;
  font-size:0.72rem;
  font-weight:700;
  color:var(--text-secondary, var(--text-3));
  text-transform:uppercase;
  letter-spacing:0.04em;
  padding-bottom:0.3rem;
  border-bottom:1px solid var(--border-color, var(--border));
  margin-bottom:0.3rem;
}
.rct-row {
  display:grid;
  grid-template-columns:1fr auto auto;
  gap:0.3rem;
  font-size:0.78rem;
  padding:0.22rem 0;
  border-bottom:1px dashed var(--border);
}
.rct-label { font-weight:700; color:var(--primary-700, var(--accent)); }
.rct-score { color:var(--success-700, var(--success)); font-weight:600; }
.rct-box   { font-family:monospace; font-size:0.7rem; color:var(--text-secondary, var(--text-3)); }

.rc-none {
  padding:0.55rem 0.7rem;
  font-size:0.8rem;
  color:var(--text-secondary, var(--text-3));
  font-style:italic;
}

/* ── Responsive ──────────────────────────────────────────────────────── */
@media (max-width:1100px) {
  .main-grid { grid-template-columns:1fr; }
}
</style>


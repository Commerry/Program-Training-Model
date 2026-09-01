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
          <p class="page-sub">Run a trained model over images, a video, or the camera</p>
        </div>
      </div>
      <div v-if="inferenceSummary" class="summary-chips">
        <span class="chip chip-purple">
          <Icon name="box" size="sm" /> {{ pickedModel ? pickedModel.run : inferenceSummary.model_name }}
        </span>
        <span class="chip chip-blue">
          <Icon name="server" size="sm" /> {{ inferenceSummary.device }}
        </span>
        <span class="chip chip-green">
          <Icon name="image" size="sm" /> {{ inferenceSummary.total }} images
        </span>
      </div>
    </header>

    <!-- ── Main grid ───────────────────────────────────────── -->
    <div class="main-grid">

      <!-- ── Left: Config panel ── -->
      <aside class="config-card">

        <!-- What to run the model over -->
        <div class="mode-tabs" role="tablist">
          <button
            v-for="option in MODES"
            :key="option.id"
            type="button"
            role="tab"
            class="mode-tab"
            :class="{ 'mode-tab--on': mode === option.id }"
            :aria-selected="mode === option.id"
            @click="setMode(option.id)"
          >
            <Icon :name="option.icon" size="sm" />
            <span>{{ option.label }}</span>
          </button>
        </div>

        <!-- Step 1: model -->
        <div class="step-block">
          <div class="step-head">
            <div class="step-num">1</div>
            <span>Choose a model</span>
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
              <span class="dz-title">Click, or drop a model file here</span>
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
              <span>Or pick one this server has already trained</span>
            </div>
            <ul class="trained-list">
              <li v-for="model in trainedModels" :key="model.path">
                <button
                  type="button"
                  class="trained-item"
                  :class="{ 'trained-item--on': pickedModel?.path === model.path }"
                  @click="pickTrained(model)"
                >
                  <!--
                    The run, not the file name: every run writes best.pt and
                    last.pt, so a list of file names is the same word repeated.
                  -->
                  <span class="trained-name">{{ model.label }}</span>
                  <span class="trained-meta">
                    {{ model.project }} &bull; {{ model.size_mb }} MB &bull;
                    {{ formatDateTime(model.modified) }}
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
            Class names taken from the model: {{ resolvedLabelSource }}
          </div>
          <textarea
            v-model="labelNames"
            class="field-textarea"
            rows="3"
            placeholder="Leave empty to use the names stored in the model&#10;or type them, for example: bottle, cap, label"
          />
          <div class="labels-file">
            <label class="labels-file-btn">
              <input type="file" accept=".txt,.csv,text/plain" @change="pickLabelsFile" />
              Load labels.txt
            </label>
            <span v-if="labelsFile" class="labels-file-name">
              {{ labelsFile.name }}
              <button type="button" class="labels-file-clear" @click="clearLabelsFile">Clear</button>
            </span>
            <span v-else class="labels-file-hint">
              An ONNX stores no class names; an export keeps them beside it
            </span>
          </div>
        </div>

        <!-- Optional: how a model from elsewhere wants to be fed -->
        <details class="conventions">
          <summary>Model built elsewhere? <span class="optional">(advanced)</span></summary>
          <p class="conventions-note">
            An ONNX does not record whether it wants padded or squashed frames,
            RGB or BGR, pixels as 0&ndash;1 or 0&ndash;255, or which order its
            box corners come in. Fed the wrong way it does not fail &mdash; it
            returns one confident box over the whole picture.
          </p>
          <p class="conventions-note">
            Run <code>python backend/tools/probe_onnx.py model.onnx folder-of-images</code>
            and paste what it reports, for example <code>stretch bgr raw xyxy</code>.
          </p>
          <input
            v-model="onnxConventions"
            class="field-input"
            type="text"
            placeholder="letterbox rgb unit xyxy"
          />
        </details>

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

        <!-- Step 4: what to detect in -->
        <div v-if="mode === 'images'" class="step-block">
          <div class="step-head">
            <div class="step-num">4</div>
            <span>Choose images</span>
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
              <span class="dz-title">Click, or drop images here</span>
              <span class="dz-hint">jpg / png / bmp &bull; several at once</span>
            </div>
            <div v-else class="image-thumbs">
              <div v-for="(f, i) in imageFiles.slice(0, 8)" :key="i" class="thumb-wrap">
                <img :src="previewUrls[i]" class="thumb" />
                <span class="thumb-name">{{ f.name }}</span>
              </div>
              <div v-if="imageFiles.length > 8" class="thumb-more">
                +{{ imageFiles.length - 8 }} more
              </div>
            </div>
          </label>
          <div v-if="imageFiles.length" class="img-count-row">
            <Icon name="image" size="sm" />
            <span><strong>{{ imageFiles.length }}</strong> file(s) chosen</span>
            <button class="link-btn" @click="clearImages">Clear</button>
          </div>
        </div>

        <!-- Step 4, video -->
        <div v-else-if="mode === 'video'" class="step-block">
          <div class="step-head">
            <div class="step-num">4</div>
            <span>Choose a video</span>
          </div>
          <label
            class="drop-zone"
            :class="{ 'drop-zone--active': videoFile }"
            @dragover.prevent
            @drop.prevent="onVideoDrop"
          >
            <input ref="videoInput" type="file" accept="video/*" class="hidden-input" @change="onVideoChange" />
            <div v-if="!videoFile" class="dz-inner">
              <Icon name="video" size="lg" />
              <span class="dz-title">Click, or drop a video here</span>
              <span class="dz-hint">mp4 / mov / avi / mkv / webm</span>
            </div>
            <div v-else class="dz-chosen">
              <div class="chosen-icon"><Icon name="video" size="md" /></div>
              <div class="chosen-info">
                <div class="chosen-name">{{ videoFile.name }}</div>
                <div class="chosen-size">{{ (videoFile.size / 1048576).toFixed(1) }} MB</div>
              </div>
              <button class="chosen-remove" @click.prevent="clearVideo">
                <Icon name="x" size="sm" />
              </button>
            </div>
          </label>

          <label class="sample-row">
            <span>Frames sampled per second</span>
            <input v-model.number="sampleFps" type="range" min="1" max="15" step="1" class="styled-slider" />
            <strong>{{ sampleFps }}</strong>
          </label>
          <p class="field-note">
            Higher is more detailed and slower. Only the box coordinates come back;
            the clip you chose is played here and drawn over, so no video is
            sent in return.
          </p>
        </div>

        <!-- Step 4, webcam -->
        <div v-else class="step-block">
          <div class="step-head">
            <div class="step-num">4</div>
            <span>Camera</span>
          </div>

          <div v-if="!cameraSupported" class="error-box">
            <Icon name="alert-triangle" size="sm" />
            <span>
              Browsers only allow the camera on https or localhost. This page is open
              at <code>{{ pageOrigin }}</code>. Open it on the machine running
              the server, at
              <code>http://localhost:{{ pagePort }}</code>, instead.
            </span>
          </div>

          <template v-else>
            <select v-if="cameras.length > 1" v-model="cameraId" class="field-select">
              <option v-for="cam in cameras" :key="cam.deviceId" :value="cam.deviceId">
                {{ cam.label || 'Camera' }}
              </option>
            </select>
            <label class="sample-row">
              <span>Detections per second</span>
              <input v-model.number="liveFps" type="range" min="1" max="15" step="1" class="styled-slider" />
              <strong>{{ liveFps }}</strong>
            </label>
            <p v-if="liveStats.fps" class="field-note">
              Actually managing {{ liveStats.fps.toFixed(1) }} fps
              ({{ liveStats.ms }} ms per frame)
            </p>
          </template>
        </div>

        <!-- Run button -->
        <button
          v-if="mode !== 'webcam'"
          class="run-btn"
          :class="{ 'run-btn--loading': loading }"
          :disabled="!canTest || loading"
          @click="runTest"
        >
          <span v-if="!loading" class="run-inner">
            <Icon name="zap" size="sm" />
            {{ mode === 'video' ? 'Analyse video' : 'Run the model' }}
          </span>
          <span v-else class="run-inner">
            <span class="spinner"></span>
            {{ videoJob && videoJob.status === 'running'
              ? `${videoJob.frames_done} frames…` : 'Working…' }}
          </span>
        </button>

        <button
          v-else
          class="run-btn"
          :class="{ 'run-btn--loading': cameraOn }"
          :disabled="!chosenModel || !cameraSupported"
          @click="cameraOn ? stopCamera() : startCamera()"
        >
          <span class="run-inner">
            <Icon :name="cameraOn ? 'x' : 'video'" size="sm" />
            {{ cameraOn ? 'Stop the camera' : 'Start the camera' }}
          </span>
        </button>

        <button
          v-if="mode === 'video' && videoJob && videoJob.status === 'running'"
          class="link-btn stop-link"
          @click="stopVideoJob"
        >Stop the analysis</button>

        <div v-if="error" class="error-box">
          <Icon name="x" size="sm" />
          {{ error }}
        </div>
      </aside>

      <!-- ── Right: Results ── -->
      <main class="result-area">

        <!-- Webcam feed -->
        <div v-if="mode === 'webcam'" class="live-panel">
          <div class="live-stage">
            <video ref="cameraVideo" class="live-video" autoplay playsinline muted></video>
            <DetectionOverlay
              :width="liveSize.width"
              :height="liveSize.height"
              :detections="liveDetections"
            />
            <div v-if="!cameraOn" class="live-idle">
              <Icon name="video" size="xl" />
              <p>Press <strong>Start the camera</strong> to detect live</p>
            </div>
          </div>
          <div v-if="cameraOn" class="live-bar">
            <span v-if="liveReading" class="reading-chip">{{ liveReading }}</span>
            <span><strong>{{ liveDetections.length }}</strong> object(s)</span>
            <span v-for="(count, label) in liveTally" :key="label" class="chip chip-purple">
              {{ label }} × {{ count }}
            </span>
          </div>
        </div>

        <!-- Video with the boxes drawn over it -->
        <div v-else-if="mode === 'video' && videoUrl" class="live-panel">
          <div class="live-stage">
            <video
              ref="playbackVideo"
              class="live-video"
              :src="videoUrl"
              controls
              playsinline
              @loadedmetadata="onPlaybackReady"
              @timeupdate="syncPlaybackBoxes"
              @seeked="syncPlaybackBoxes"
            ></video>
            <DetectionOverlay
              :width="videoSize.width"
              :height="videoSize.height"
              :detections="playbackDetections"
            />
          </div>
          <div v-if="videoJob" class="live-bar">
            <span v-if="videoJob.status === 'running'">
              Analysed {{ videoJob.frames_done }} frames…
            </span>
            <template v-else>
              <span>{{ videoJob.frames_total }} frames</span>
              <span><strong>{{ videoJob.detection_count }}</strong> objects in total</span>
              <span>took {{ videoJob.elapsed_s }}s</span>
              <a
                v-if="videoJob.frames_total"
                class="csv-link"
                :href="csvUrl"
                download
              >
                <Icon name="download" size="sm" />
                <span>Download CSV</span>
              </a>
              <span v-if="playbackReading" class="reading-chip">{{ playbackReading }}</span>
              <span v-if="playbackDetections.length" class="chip chip-green">
                {{ playbackDetections.length }} here
              </span>
            </template>
          </div>
          <p v-if="videoJob && videoJob.message" class="field-note">{{ videoJob.message }}</p>
        </div>

        <!-- Empty state -->
        <div v-else-if="!results.length && !loading" class="empty-state">
          <div class="empty-icon">
            <Icon name="search" size="xl" />
          </div>
          <h3>Nothing to show yet</h3>
          <p>Choose a model and some images, then press <strong>Run the model</strong></p>
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
          <!--
            Files the model never saw, and why. They used to vanish: a request
            with three images came back with two results and nothing to say
            where the third went, or failed the whole batch with a bare 500.
          -->
          <div v-if="rejected.length" class="rejected-bar">
            <Icon name="alert-triangle" size="sm" />
            <div>
              <strong>{{ rejected.length }} file(s) could not be processed</strong>
              <ul>
                <li v-for="item in rejected" :key="item.filename">
                  <code>{{ item.filename }}</code> — {{ item.reason }}
                </li>
              </ul>
            </div>
          </div>

          <div class="results-summary-bar">
            <span class="rs-label">{{ results.length }} image(s)</span>
            <span class="rs-det">
              <strong>{{ totalDetections }}</strong> object(s) found
              (threshold ≥ {{ scoreThreshold.toFixed(2) }})
            </span>
            <span v-if="uncertainCount" class="rs-uncertain">
              <Icon name="alert-triangle" size="sm" />
              {{ uncertainCount }} below {{ UNCERTAIN_BELOW }} — check by eye
            </span>
            <button class="link-btn" :disabled="exporting" @click="exportResults">
              <Icon name="download" size="sm" />
              {{ exporting ? 'Preparing...' : 'Export to Excel' }}
            </button>
          </div>
          <div class="results-grid">
            <article
              v-for="item in results"
              :key="item.filename"
              class="result-card result-card--clickable"
              @click="openDetail(item)"
            >

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

              <div v-if="item.reading" class="rc-reading">
                <span class="rc-reading-label">Reads</span>
                <span class="rc-reading-value">{{ item.reading }}</span>
              </div>

              <div v-if="item.detections.length" class="rc-table">
                <div class="rct-head">
                  <span>#</span><span>Label</span><span>Score</span><span>Bounding Box</span>
                </div>
                <div v-for="(d, idx) in item.detections" :key="idx" class="rct-row">
                  <span class="rct-order">{{ idx + 1 }}</span>
                  <span class="rct-label">{{ d.label_name }}</span>
                  <span class="rct-score">{{ d.score }}</span>
                  <span class="rct-box">{{ d.box.join(', ') }}</span>
                </div>
              </div>
              <div v-else class="rc-none">
                Nothing above the threshold
              </div>

            </article>
          </div>
        </div>

      </main>
    </div>
  </div>

    <!--
      One result, large. The grid answers "how did it do overall"; this answers
      "what exactly did it say about this picture", which is the question asked
      of anything that looks wrong.
    -->
    <div v-if="detail" class="detail-backdrop" @click.self="detail = null">
      <div class="detail-panel">
        <header class="detail-head">
          <div>
            <h3 class="detail-title">{{ detail.filename }}</h3>
            <p class="detail-sub">
              {{ detail.detections.length }} object(s)
              <template v-if="detail.reading"> &bull; reads
                <strong>{{ detail.reading }}</strong>
              </template>
            </p>
          </div>
          <button class="chosen-remove" @click="detail = null">
            <Icon name="x" size="sm" />
          </button>
        </header>

        <img :src="detail.annotated_image" :alt="detail.filename" class="detail-image" />

        <table v-if="detail.detections.length" class="detail-table">
          <thead>
            <tr>
              <th>#</th><th>Label</th><th>Confidence</th>
              <th>Box</th><th>Note</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(d, index) in detail.detections"
              :key="index"
              :class="{ 'row-uncertain': d.score < UNCERTAIN_BELOW }"
            >
              <td>{{ index + 1 }}</td>
              <td class="detail-label">{{ d.label_name }}</td>
              <td>{{ Math.round(d.score * 100) }}%</td>
              <td class="detail-box">{{ d.box.join(', ') }}</td>
              <td>
                <span v-if="d.score < UNCERTAIN_BELOW">
                  Below {{ UNCERTAIN_BELOW }} — check by eye
                </span>
              </td>
            </tr>
          </tbody>
        </table>
        <p v-else class="detail-empty">
          Nothing above the threshold. Either the object is absent, or the model
          missed it.
        </p>
      </div>
    </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import Icon from '@/components/Icon.vue'
import DetectionOverlay from '@/components/DetectionOverlay.vue'
import { formatDateTime } from '@/utils/format'
import { errorMessage, trainingService } from '@/services'

// ── Constants ───────────────────────────────────────────────────────
const MODEL_FORMATS = [
  { ext: '.pth',         label: '.pth — PyTorch weights' },
  { ext: '.pt',          label: '.pt — YOLO weights' },
  { ext: '.onnx',        label: '.onnx — ONNX Runtime' },
  { ext: '.torchscript', label: '.torchscript — TorchScript' },
]

const EMPTY_STEPS = [
  'Choose a model (.pth / .pt / .onnx / .torchscript)',
  'Type the class names if you know them, separated by commas',
  'Set the score threshold',
  'Choose images to test with',
  'Press Run the model',
]

// ── State ────────────────────────────────────────────────────────────
const MODES = [
  { id: 'images', label: 'Images', icon: 'image' },
  { id: 'video',  label: 'Video', icon: 'video' },
  { id: 'webcam', label: 'Camera', icon: 'camera' },
]
const mode = ref('images')

const modelFile     = ref(null)
const modelInput    = ref(null)

// ── Video ──────────────────────────────────────────────────────────────────
// The analysed video is never sent back. The server returns boxes per sampled
// frame and the browser plays the file the user already chose, drawing over
// it. Writing H.264 from OpenCV depends on a codec library that is not
// reliably present, so a server that returned video would work on one install
// and produce something unplayable on the next.
const videoFile          = ref(null)
const videoInput         = ref(null)
const videoUrl           = ref('')
const videoJob           = ref(null)
const videoSize          = ref({ width: 0, height: 0 })
const playbackVideo      = ref(null)
const playbackDetections = ref([])
const sampleFps          = ref(5)
let videoPoll = null

// ── Webcam ─────────────────────────────────────────────────────────────────
const cameraVideo    = ref(null)
const cameraOn       = ref(false)
const cameraId       = ref('')
const cameras        = ref([])
const liveDetections = ref([])
const liveSize       = ref({ width: 0, height: 0 })
const liveFps        = ref(6)
const liveStats      = ref({ fps: 0, ms: 0 })
let cameraStream = null
let liveTimer = null
let liveBusy = false

// getUserMedia only exists in a secure context: https, or localhost. Opening
// the app from another machine over plain http gives no camera at all, and the
// browser reports that as the API simply being absent, so it is worth saying
// what is actually wrong rather than letting the button do nothing.
const cameraSupported = computed(() =>
  typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia)
const pageOrigin = computed(() =>
  typeof window !== 'undefined' ? window.location.origin : '')
const pagePort = computed(() =>
  typeof window !== 'undefined' ? (window.location.port || '80') : '')
// Models this server has already produced. Uploading best.pt back to the
// machine that just wrote it was the only way to try a freshly trained model.
const trainedModels = ref([])
const pickedModel   = ref(null)
const imageFiles    = ref([])
const imagesInput   = ref(null)
const previewUrls   = ref([])
const labelNames    = ref('')
// How an ONNX built by other tooling wants to be fed. Nothing in the file
// records it, and feeding it the wrong way returns confident nonsense rather
// than an error, so this stays empty until backend/tools/probe_onnx.py has
// been run against the model.
const onnxConventions = ref('')
// The class names an ONNX does not carry. Custom Vision and most other
// exports write them to a labels.txt in the same folder, which does not come
// along when the model file alone is uploaded -- which is why such a model
// reports class_11 rather than the name it was trained with.
const labelsFile = ref(null)

const pickLabelsFile = (event) => {
  labelsFile.value = event.target.files?.[0] || null
}
const clearLabelsFile = () => { labelsFile.value = null }
const scoreThreshold = ref(0.5)
const loading       = ref(false)
const error         = ref('')
const results       = ref([])
const inferenceSummary = ref(null)
const resolvedLabelSource = ref('')

// ── Computed ─────────────────────────────────────────────────────────
const chosenModel   = computed(() => pickedModel.value || modelFile.value)
const canTest       = computed(() => {
  if (!chosenModel.value) return false
  if (mode.value === 'video') {
    // Only a model this server trained: the analysis outlives the request that
    // starts it, so an uploaded copy would be gone before the worker read it.
    return !!videoFile.value && !!pickedModel.value?.path
  }
  return imageFiles.value.length > 0
})

// The classes in reading order, as one string. For a project whose classes
// are the characters on a display this is the answer; a list of boxes is not.
const readingOf = (detections) => {
  const lines = {}
  for (const d of detections || []) {
    const line = d.line ?? 0
    ;(lines[line] ||= []).push(d.label_name)
  }
  return Object.keys(lines).sort((a, b) => a - b)
    .map((k) => lines[k].join('')).join('\n')
}

// Served as a file rather than fetched: the browser saves it directly, and a
// long analysis is a large string that has no reason to pass through here.
// The line under which a reading should be checked by a person before it is
// acted on. The score threshold already decided what to report at all; this is
// where a detector's own confidence stops being a majority.
const UNCERTAIN_BELOW = 0.5

const rejected = ref([])
const detail = ref(null)
const exporting = ref(false)

const openDetail = (item) => { detail.value = item }

const uncertainCount = computed(() => results.value.reduce(
  (total, item) => total + (item.detections || [])
    .filter((d) => d.score < UNCERTAIN_BELOW).length, 0))

const exportResults = async () => {
  if (exporting.value || !results.value.length) return
  exporting.value = true
  error.value = ''
  try {
    const blob = await trainingService.exportTestResults({
      results: results.value,
      model_name: inferenceSummary.value?.model_name,
      device: inferenceSummary.value?.device,
      score_threshold: scoreThreshold.value
    })
    // Handed to the browser rather than opened here: it is a file to keep.
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${(inferenceSummary.value?.model_name || 'model')
      .replace(/\.[^.]+$/, '')}_test.xlsx`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  } catch (err) {
    error.value = errorMessage(err, 'The spreadsheet could not be written')
  } finally {
    exporting.value = false
  }
}

const csvUrl = computed(() =>
  videoJob.value?.id ? trainingService.videoCsvUrl(videoJob.value.id) : '')

const liveReading = computed(() => readingOf(liveDetections.value))
const playbackReading = computed(() => readingOf(playbackDetections.value))

const liveTally = computed(() => {
  const counts = {}
  for (const d of liveDetections.value) {
    counts[d.label_name] = (counts[d.label_name] || 0) + 1
  }
  return counts
})
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

const setMode = (next) => {
  if (next === mode.value) return
  stopCamera()
  mode.value = next
  error.value = ''
}

// ── Video ──────────────────────────────────────────────────────────────────
const clearVideo = () => {
  if (videoUrl.value) URL.revokeObjectURL(videoUrl.value)
  videoUrl.value = ''
  videoFile.value = null
  videoJob.value = null
  playbackDetections.value = []
  videoSize.value = { width: 0, height: 0 }
  if (videoPoll) { clearInterval(videoPoll); videoPoll = null }
}

const chooseVideo = (file) => {
  clearVideo()
  if (!file) return
  videoFile.value = file
  videoUrl.value = URL.createObjectURL(file)
  error.value = ''
}
const onVideoChange = (e) => chooseVideo(e.target.files?.[0] || null)
const onVideoDrop   = (e) => chooseVideo(e.dataTransfer.files?.[0] || null)

const onPlaybackReady = () => {
  const el = playbackVideo.value
  if (el) videoSize.value = { width: el.videoWidth, height: el.videoHeight }
}

/**
 * Show the boxes from the sampled frame nearest the playhead.
 *
 * Sampling is coarser than playback, so between samples the last known boxes
 * stay on screen — which is what a viewer perceives anyway at these rates.
 * Binary search rather than a scan: a long clip holds thousands of samples and
 * this runs on every timeupdate.
 */
const syncPlaybackBoxes = () => {
  const frames = videoJob.value?.frames
  const el = playbackVideo.value
  if (!frames?.length || !el) { playbackDetections.value = []; return }

  const t = el.currentTime
  let lo = 0
  let hi = frames.length - 1
  let best = 0
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid].time_s <= t) { best = mid; lo = mid + 1 } else { hi = mid - 1 }
  }
  playbackDetections.value = frames[best].detections || []
}

const runVideo = async () => {
  loading.value = true
  error.value = ''
  playbackDetections.value = []
  try {
    videoJob.value = await trainingService.analyseVideo(
      pickedModel.value, videoFile.value,
      {
        scoreThreshold: scoreThreshold.value,
        labelNames: labelNames.value,
        sampleFps: sampleFps.value,
      })
    videoPoll = setInterval(async () => {
      try {
        const job = await trainingService.videoStatus(videoJob.value.id)
        videoJob.value = job
        if (job.status !== 'running') {
          clearInterval(videoPoll)
          videoPoll = null
          loading.value = false
          if (job.status === 'failed') error.value = job.message
          if (Array.isArray(job.label_names) && job.label_names.length && !labelNames.value) {
            labelNames.value = job.label_names.join(', ')
          }
          syncPlaybackBoxes()
        }
      } catch (err) {
        clearInterval(videoPoll)
        videoPoll = null
        loading.value = false
        error.value = errorMessage(err, 'Lost track of the video analysis')
      }
    }, 1000)
  } catch (err) {
    loading.value = false
    error.value = errorMessage(err, 'The video could not be analysed')
  }
}

const stopVideoJob = async () => {
  if (!videoJob.value?.id) return
  try {
    await trainingService.stopVideo(videoJob.value.id)
  } catch (err) {
    error.value = errorMessage(err)
  }
}

// ── Webcam ─────────────────────────────────────────────────────────────────
const startCamera = async () => {
  if (!cameraSupported.value || !chosenModel.value) return
  error.value = ''
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: cameraId.value
        ? { deviceId: { exact: cameraId.value } }
        : { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    })
    const el = cameraVideo.value
    el.srcObject = cameraStream
    await el.play()
    liveSize.value = { width: el.videoWidth, height: el.videoHeight }
    cameraOn.value = true

    // Labels only become readable once permission has been granted, so the
    // list is worth refreshing here rather than on mount.
    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      cameras.value = devices.filter((d) => d.kind === 'videoinput')
    } catch { /* listing cameras is a convenience, not a requirement */ }

    scheduleFrame()
  } catch (err) {
    error.value = err?.name === 'NotAllowedError'
      ? 'The browser refused the camera. Allow it from the address bar and try again.'
      : `The camera could not be opened: ${err?.message || err}`
  }
}

const stopCamera = () => {
  cameraOn.value = false
  if (liveTimer) { clearTimeout(liveTimer); liveTimer = null }
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop())
    cameraStream = null
  }
  if (cameraVideo.value) cameraVideo.value.srcObject = null
  liveDetections.value = []
  liveStats.value = { fps: 0, ms: 0 }
}

const scheduleFrame = () => {
  if (!cameraOn.value) return
  liveTimer = setTimeout(grabFrame, Math.round(1000 / liveFps.value))
}

/**
 * Send one frame and draw what comes back.
 *
 * Only ever one request in flight. Firing on a timer regardless would queue
 * requests behind each other the moment inference is slower than the interval,
 * and the feed would fall further behind the longer it ran.
 */
const grabFrame = async () => {
  if (!cameraOn.value || liveBusy) { scheduleFrame(); return }
  const el = cameraVideo.value
  if (!el || !el.videoWidth) { scheduleFrame(); return }

  liveBusy = true
  const started = performance.now()
  try {
    const canvas = document.createElement('canvas')
    canvas.width = el.videoWidth
    canvas.height = el.videoHeight
    canvas.getContext('2d').drawImage(el, 0, 0)
    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', 0.8))

    if (blob && cameraOn.value) {
      const res = await trainingService.detectFrame(chosenModel.value, blob, {
        scoreThreshold: scoreThreshold.value,
        labelNames: labelNames.value,
      })
      liveDetections.value = res.detections || []
      liveSize.value = { width: res.width, height: res.height }
      if (!labelNames.value && res.label_names?.length) {
        labelNames.value = res.label_names.join(', ')
      }
      const ms = performance.now() - started
      liveStats.value = { fps: 1000 / ms, ms: Math.round(ms) }
    }
  } catch (err) {
    error.value = errorMessage(err, 'That frame could not be processed')
    stopCamera()
    return
  } finally {
    liveBusy = false
  }
  scheduleFrame()
}

// Changing the model mid-feed would keep drawing the previous one's boxes.
watch(chosenModel, () => { if (cameraOn.value) stopCamera() })

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
  // Leaving the page must release the camera. Without this the browser keeps
  // showing the in-use indicator and the light stays on until the tab closes.
  stopCamera()
  clearVideo()
})

// ── Run inference ────────────────────────────────────────────────────
const runTest = async () => {
  if (!canTest.value || loading.value) return
  detail.value = null
  if (mode.value === 'video') { await runVideo(); return }
  loading.value = true
  error.value   = ''
  results.value = []
  rejected.value = []

  try {
    const res = await trainingService.testModel(
      chosenModel.value,
      imageFiles.value,
      {
        scoreThreshold: scoreThreshold.value,
        labelNames: labelNames.value,
        labelsFile: labelsFile.value,
        onnxConventions: onnxConventions.value.trim()
      }
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
    rejected.value = res.rejected || []
  } catch (err) {
    error.value = errorMessage(err, 'The model could not be run')
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

.mode-tabs {
  display:grid;
  grid-template-columns:repeat(3, 1fr);
  gap:0.3rem;
  margin-bottom:1.1rem;
  padding:0.3rem;
  border:1px solid var(--border-color);
  border-radius:var(--radius-md);
  background:var(--surface-2);
}

.mode-tab {
  display:flex;
  align-items:center;
  justify-content:center;
  gap:0.4rem;
  padding:0.5rem 0.4rem;
  border:0;
  border-radius:calc(var(--radius-md) - 2px);
  background:transparent;
  color:var(--text-secondary);
  font:inherit;
  font-size:0.86rem;
  cursor:pointer;
  transition:background var(--t) var(--ease), color var(--t) var(--ease);
}

.mode-tab:hover { color:var(--text-primary); }

.mode-tab--on {
  background:var(--accent);
  color:var(--text-inverse);
  font-weight:600;
}

.sample-row {
  display:flex;
  align-items:center;
  gap:0.6rem;
  margin-top:0.8rem;
  color:var(--text-secondary);
  font-size:0.86rem;
}

.sample-row .styled-slider { flex:1; }
.sample-row strong { color:var(--text-primary); min-width:1.6rem; text-align:right; }

.field-note {
  margin:0.5rem 0 0 0;
  color:var(--text-tertiary);
  font-size:0.8rem;
  line-height:1.55;
}

.field-note code,
.error-box code {
  padding:0.05rem 0.3rem;
  border-radius:4px;
  background:var(--surface-3);
  color:var(--text-primary);
  font-family:var(--font-mono);
  font-size:0.92em;
}

.field-select {
  width:100%;
  margin-top:0.6rem;
  padding:0.5rem 0.6rem;
  border:1px solid var(--border-color);
  border-radius:var(--radius-md);
  background:var(--surface-2);
  color:var(--text-primary);
  font:inherit;
}

.stop-link {
  display:block;
  margin:0.6rem auto 0;
}

/* ── Live stage, shared by the video and webcam views ────────────────── */
.live-panel {
  display:flex;
  flex-direction:column;
  gap:0.8rem;
}

.live-stage {
  position:relative;
  width:100%;
  aspect-ratio:16 / 9;
  border:1px solid var(--border-color);
  border-radius:var(--radius-lg);
  background:#05060a;
  overflow:hidden;
}

.live-video {
  width:100%;
  height:100%;
  object-fit:contain;
  display:block;
}

.live-idle {
  position:absolute;
  inset:0;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  gap:0.7rem;
  color:var(--text-tertiary);
  text-align:center;
}

.live-idle p { margin:0; }

.live-bar {
  display:flex;
  flex-wrap:wrap;
  align-items:center;
  gap:0.5rem 1rem;
  padding:0.6rem 0.9rem;
  border:1px solid var(--border-color);
  border-radius:var(--radius-md);
  background:var(--surface-2);
  color:var(--text-secondary);
  font-size:0.86rem;
}

.live-bar strong { color:var(--text-primary); }

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

.labels-file {
  display:flex;
  align-items:center;
  gap:0.5rem;
  flex-wrap:wrap;
  margin-top:0.5rem;
}
.labels-file-btn {
  cursor:pointer;
  font-size:0.76rem;
  font-weight:600;
  padding:0.35rem 0.65rem;
  border-radius:8px;
  border:1px solid var(--border-color, var(--border));
  background: var(--bg-subtle);
  color:var(--text-primary, var(--text));
}
.labels-file-btn input { display:none; }
.labels-file-name {
  font-size:0.76rem;
  color:var(--primary-700, var(--accent));
}
.labels-file-clear {
  margin-left:0.35rem;
  background:none;
  border:none;
  padding:0;
  cursor:pointer;
  font-size:0.72rem;
  color:var(--danger, #d9534f);
  text-decoration:underline;
}
.labels-file-hint {
  font-size:0.72rem;
  color:var(--text-tertiary, var(--text-3));
}

.conventions {
  border:1px solid var(--border-color, var(--border));
  border-radius:10px;
  padding:0.55rem 0.7rem;
  background: var(--bg-subtle);
}
.conventions > summary {
  cursor:pointer;
  font-size:0.82rem;
  font-weight:600;
  color:var(--text-primary, var(--text));
}
.conventions-note {
  margin:0.5rem 0 0;
  font-size:0.74rem;
  line-height:1.5;
  color:var(--text-tertiary, var(--text-3));
}
.conventions-note code {
  font-size:0.72rem;
  word-break:break-all;
  color:var(--text-primary, var(--text));
}
.field-input {
  width:100%;
  margin-top:0.55rem;
  background: var(--bg-subtle);
  border:1px solid var(--border-color, var(--border));
  border-radius:10px;
  padding:0.5rem 0.7rem;
  font-family:inherit;
  font-size:0.85rem;
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
.rc-reading {
  display:flex;
  align-items:baseline;
  gap:0.6rem;
  margin:0.6rem 0 0.2rem;
  padding:0.5rem 0.75rem;
  border:1px solid var(--accent);
  border-radius:var(--radius-md);
  background:var(--accent-soft);
}

.rc-reading-label {
  color:var(--text-secondary);
  font-size:0.76rem;
  text-transform:uppercase;
  letter-spacing:0.06em;
}

.rc-reading-value {
  color:var(--text-primary);
  font-family:var(--font-mono);
  font-size:1.35rem;
  font-weight:700;
  letter-spacing:0.08em;
  white-space:pre-line;
  user-select:text;
  cursor:text;
}

.result-card--clickable { cursor:pointer; }

.result-card--clickable:hover {
  border-color:var(--accent);
}

.rejected-bar {
  display:flex;
  align-items:flex-start;
  gap:0.6rem;
  margin-bottom:0.8rem;
  padding:0.7rem 0.9rem;
  border:1px solid var(--warning-300);
  border-radius:var(--radius-md);
  background:var(--warning-100);
  color:var(--warning-700);
  font-size:0.85rem;
}

.rejected-bar ul {
  margin:0.35rem 0 0;
  padding-left:1.1rem;
}

.rejected-bar code {
  font-family:var(--font-mono);
  font-size:0.92em;
}

.rs-uncertain {
  display:inline-flex;
  align-items:center;
  gap:0.35rem;
  color:var(--warning-700);
  font-size:0.82rem;
}

/* ── One result, large ─────────────────────────────────────────────── */
.detail-backdrop {
  position:fixed;
  inset:0;
  z-index:60;
  display:flex;
  align-items:center;
  justify-content:center;
  padding:2rem 1rem;
  background:rgba(4, 6, 12, 0.72);
  overflow-y:auto;
}

.detail-panel {
  width:min(52rem, 100%);
  max-height:100%;
  overflow-y:auto;
  padding:1.2rem 1.4rem 1.4rem;
  border:1px solid var(--border-color);
  border-radius:var(--radius-lg);
  background:var(--surface);
  box-shadow:var(--shadow-lg);
}

.detail-head {
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:1rem;
  margin-bottom:0.9rem;
}

.detail-title {
  margin:0;
  color:var(--text-primary);
  font-size:1rem;
  word-break:break-all;
}

.detail-sub {
  margin:0.2rem 0 0;
  color:var(--text-secondary);
  font-size:0.85rem;
}

.detail-image {
  width:100%;
  border-radius:var(--radius-md);
  background:#05060a;
}

.detail-table {
  width:100%;
  margin-top:1rem;
  border-collapse:collapse;
  font-size:0.84rem;
}

.detail-table th {
  padding:0.4rem 0.5rem;
  border-bottom:1px solid var(--border-color);
  color:var(--text-tertiary);
  font-weight:600;
  text-align:left;
}

.detail-table td {
  padding:0.4rem 0.5rem;
  border-bottom:1px solid var(--border-color);
  color:var(--text-secondary);
}

.detail-label { color:var(--text-primary); font-weight:600; }
.detail-box { font-family:var(--font-mono); font-size:0.78rem; }

.row-uncertain td {
  background:var(--warning-100);
  color:var(--warning-700);
}

.detail-empty {
  margin-top:1rem;
  color:var(--text-secondary);
  font-size:0.86rem;
}

.csv-link {
  display:inline-flex;
  align-items:center;
  gap:0.35rem;
  padding:0.2rem 0.6rem;
  border:1px solid var(--border-color);
  border-radius:var(--radius-md);
  background:var(--surface);
  color:var(--text-secondary);
  font-size:0.82rem;
  text-decoration:none;
}

.csv-link:hover {
  border-color:var(--accent);
  color:var(--text-primary);
}

.reading-chip {
  padding:0.2rem 0.6rem;
  border:1px solid var(--accent);
  border-radius:var(--radius-md);
  background:var(--accent-soft);
  color:var(--text-primary);
  font-family:var(--font-mono);
  font-size:1rem;
  font-weight:700;
  letter-spacing:0.06em;
  white-space:pre-line;
  user-select:text;
  cursor:text;
}

.rct-order {
  color:var(--text-tertiary);
  font-family:var(--font-mono);
}

.rct-head {
  display:grid;
  grid-template-columns:1.4rem 1fr auto auto;
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
  grid-template-columns:1.4rem 1fr auto auto;
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


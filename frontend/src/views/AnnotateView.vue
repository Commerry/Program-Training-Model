<template>
  <div class="annotate-view">
    <!-- Navigation bar -->
    <div class="annotate-header">
      <button @click="$router.back()" class="btn-back">
        <Icon name="arrow-left" size="sm" />
        Back
      </button>
      <span class="annotate-title">{{ filename }}</span>
      <span class="save-state" :class="saveStateClass">{{ saveStateLabel }}</span>
      <span v-if="copyNotice" class="copy-notice">{{ copyNotice }}</span>
      <div class="header-actions">
        <button @click="previousImage" :disabled="currentIndex <= 0 || saving" class="btn btn-secondary">
          <Icon name="arrow-left" size="sm" />
          <span>Previous</span>
        </button>
        <span class="image-counter">
          {{ currentIndex + 1 }} / {{ totalImages }}
        </span>
        <button @click="nextImage" :disabled="currentIndex >= totalImages - 1 || saving" class="btn btn-secondary">
          <Icon name="arrow-right" size="sm" />
          <span>Next</span>
        </button>
        <button
          class="btn btn-secondary"
          :disabled="currentIndex <= 0 || copying"
          :title="'Copy the boxes from the previous image  (C)'"
          @click="copyFromPrevious"
        >
          <Icon name="copy" size="sm" />
          <span>{{ copying ? 'Copying...' : 'Copy previous' }}</span>
        </button>
        <button @click="saveAnnotations" class="btn btn-primary" :disabled="saving">
          <Icon name="save" size="sm" />
          <span>{{ saving ? 'Saving...' : 'Save' }}</span>
        </button>
        <button
          class="btn-icon"
          title="Keyboard shortcuts"
          @click="showShortcuts = !showShortcuts"
        >
          <Icon name="command" size="sm" />
        </button>
        <button @click="goToTrain" class="btn btn-success">
          <Icon name="zap" size="sm" />
          <span>Train Model</span>
        </button>

        <div v-if="showShortcuts" class="shortcut-sheet" @click.stop>
          <h4>Keyboard</h4>
          <div v-for="row in shortcuts" :key="row.keys" class="shortcut-row">
            <kbd>{{ row.keys }}</kbd>
            <span>{{ row.action }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="annotation-container">
      <!-- Sidebar -->
      <div class="sidebar">
        <div class="sidebar-section">
          <h3 class="sidebar-title">
            <Icon name="tool" size="sm" />
            Tools
          </h3>
          <div class="tool-buttons">
            <button
              @click="activeTool = 'select'"
              :class="['tool-btn', { active: activeTool === 'select' }]"
              title="Select"
            >
              <Icon name="mouse-pointer" size="sm" />
              <span>Select</span>
            </button>
            <button
              @click="activeTool = 'draw'"
              :class="['tool-btn', { active: activeTool === 'draw' }]"
              title="Draw Box"
            >
              <Icon name="square" size="sm" />
              <span>Draw Box</span>
            </button>
          </div>
        </div>

        <div class="sidebar-section">
          <h3 class="sidebar-title">
            <Icon name="tag" size="sm" />
            Current Tag
          </h3>
          <select v-model="currentTag" class="form-select">
            <option value="">Select tag...</option>
            <option
              v-for="tag in existingTags"
              :key="tag"
              :value="tag"
            >
              {{ tag }}
            </option>
          </select>
          
          <div class="new-tag-form">
            <input
              v-model="newTag"
              type="text"
              class="form-input"
              placeholder="New tag name"
              @keyup.enter="addNewTag"
            />
            <button @click="addNewTag" class="btn btn-primary btn-sm">
              Add
            </button>
          </div>
        </div>

        <div class="sidebar-section">
          <h3 class="sidebar-title">
            <Icon name="list" size="sm" />
            Regions ({{ regions.length }})
          </h3>
          <div class="regions-list">
            <div
              v-for="(region, idx) in regions"
              :key="idx"
              :class="['region-item', { selected: selectedRegionIndex === idx }]"
              @click="selectRegion(idx)"
            >
              <div class="region-info">
                <span class="region-tag">{{ region.tag }}</span>
                <span class="region-coords">
                  {{ Math.round(region.x) }}, {{ Math.round(region.y) }}
                  ({{ Math.round(region.width) }} × {{ Math.round(region.height) }})
                </span>
              </div>
              <button @click.stop="deleteRegion(idx)" class="btn-delete">
                <Icon name="trash-2" size="sm" />
              </button>
            </div>
          </div>
        </div>

        <div class="sidebar-section">
          <h3 class="sidebar-title">
            <Icon name="command" size="sm" />
            Shortcuts
          </h3>
          <div class="shortcuts">
            <div class="shortcut">
              <kbd>Delete</kbd> Delete region
            </div>
            <div class="shortcut">
              <kbd>Ctrl+S</kbd> Save
            </div>
          </div>
        </div>
      </div>

      <!-- Canvas -->
      <div class="canvas-wrapper card">
        <div v-if="loading" class="loading">Loading image...</div>
        <canvas ref="canvas" id="canvas"></canvas>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { errorMessage, projectService } from '@/services'
import { fabric } from 'fabric'
import Icon from '@/components/Icon.vue'

const route = useRoute()
const router = useRouter()

const projectName = computed(() => route.params.name)
const filename = computed(() => route.params.filename)

const canvas = ref(null)
const fabricCanvas = ref(null)
const loading = ref(true)
const saving = ref(false)

const activeTool = ref('select')
const currentTag = ref('')
const newTag = ref('')
const existingTags = ref([])
const regions = ref([])
const selectedRegionIndex = ref(null)

const imageData = ref(null)
const allImages = ref([])
const currentIndex = ref(0)
const totalImages = ref(0)

const dirty = ref(false)
const saveError = ref(null)
const savedAt = ref(null)
const loadError = ref(null)

const saveStateLabel = computed(() => {
  if (saving.value) return 'Saving...'
  if (saveError.value) return 'Not saved'
  if (dirty.value) return 'Unsaved changes'
  if (savedAt.value) return 'Saved'
  return ''
})

const saveStateClass = computed(() => {
  if (saveError.value) return 'is-error'
  if (dirty.value || saving.value) return 'is-dirty'
  if (savedAt.value) return 'is-saved'
  return ''
})

/** Browsers show their own confirmation when a beforeunload handler cancels. */
const beforeUnload = (event) => {
  if (!dirty.value) return
  event.preventDefault()
  event.returnValue = ''
}

let isDrawing = false
let drawingRect = null
let startX = 0
let startY = 0
let imageScale = 1  // canvas scale factor, updated in loadImageToCanvas

// Reload image when route filename changes (prev/next navigation)
watch(filename, async (newFilename) => {
  if (newFilename && fabricCanvas.value) {
    currentIndex.value = allImages.value.findIndex(img => img.filename === newFilename)
    await loadImage()
  }
})

onMounted(async () => {
  try {
    // Project-wide tags come first so the tag picker is populated even for an
    // image that has no boxes yet.
    const [tagsResult, imagesResult] = await Promise.all([
      projectService.tags(projectName.value),
      projectService.images(projectName.value)
    ])
    existingTags.value = tagsResult.tags || []
    allImages.value = imagesResult.images || []
    totalImages.value = allImages.value.length
    currentIndex.value = allImages.value.findIndex((img) => img.filename === filename.value)
  } catch (error) {
    loadError.value = errorMessage(error)
  }

  await loadImage()
  initCanvas()
  window.addEventListener('keydown', handleKeydown)
  window.addEventListener('beforeunload', beforeUnload)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
  window.removeEventListener('beforeunload', beforeUnload)
  if (fabricCanvas.value) fabricCanvas.value.dispose()
})

const loadImage = async () => {
  loading.value = true
  copyNotice.value = ''
  regions.value = []
  selectedRegionIndex.value = null
  
  try {
    const result = await projectService.imageData(projectName.value, filename.value)
    if (result.success) {
      dirty.value = false
      saveError.value = null
      imageData.value = result.data
      regions.value = result.data.annotations?.regions || []
      // Don't override existingTags - it's loaded from global project tags
      // Add any new tags found in this image to the global list
      const imageTags = regions.value.map(r => r.tag).filter(Boolean)
      imageTags.forEach(tag => {
        if (!existingTags.value.includes(tag)) {
          existingTags.value.push(tag)
        }
      })
      if (fabricCanvas.value) loadImageToCanvas()
    }
  } catch (error) {
    loadError.value = errorMessage(error)
  } finally {
    loading.value = false
  }
}

const initCanvas = () => {
  fabricCanvas.value = new fabric.Canvas('canvas', {
    isDrawingMode: false,
    selection: true,
    preserveObjectStacking: true
  })

  fabricCanvas.value.on('mouse:down', handleMouseDown)
  fabricCanvas.value.on('mouse:move', handleMouseMove)
  fabricCanvas.value.on('mouse:up', handleMouseUp)
  fabricCanvas.value.on('selection:created', handleSelection)
  fabricCanvas.value.on('selection:updated', handleSelection)
  fabricCanvas.value.on('selection:cleared', () => { selectedRegionIndex.value = null })
  fabricCanvas.value.on('object:modified', handleObjectModified)
  fabricCanvas.value.on('object:moving', handleObjectMoving)

  if (imageData.value) loadImageToCanvas()
}

const loadImageToCanvas = () => {
  if (!fabricCanvas.value || !imageData.value) return
  fabricCanvas.value.clear()

  fabric.Image.fromURL(imageData.value.image, (img) => {
    const maxWidth = 1000
    const maxHeight = 650
    let scale = 1
    if (img.width > maxWidth || img.height > maxHeight) {
      scale = Math.min(maxWidth / img.width, maxHeight / img.height)
    }
    imageScale = scale

    fabricCanvas.value.setWidth(img.width * scale)
    fabricCanvas.value.setHeight(img.height * scale)
    fabricCanvas.value.setBackgroundImage(img, fabricCanvas.value.renderAll.bind(fabricCanvas.value), {
      scaleX: scale,
      scaleY: scale
    })

    regions.value.forEach((region, idx) => drawRegionOnCanvas(region, idx))
    fabricCanvas.value.renderAll()
  })
}

// Draw a single region rect + label on canvas
const drawRegionOnCanvas = (region, idx) => {
  const color = getColorForTag(region.tag)
  const s = imageScale

  const rect = new fabric.Rect({
    left: region.x * s,
    top: region.y * s,
    width: region.width * s,
    height: region.height * s,
    fill: 'rgba(0,0,0,0)',
    stroke: color,
    strokeWidth: 2,
    strokeUniform: true,
    cornerColor: color,
    cornerStrokeColor: 'white',
    cornerSize: 10,
    cornerStyle: 'circle',
    transparentCorners: false,
    lockRotation: true,
    hasRotatingPoint: false,
    regionIndex: idx
  })

  const label = new fabric.Text(region.tag || '', {
    left: region.x * s + 3,
    top: region.y * s + 3,
    fontSize: 12,
    fontFamily: 'Arial, sans-serif',
    fill: 'white',
    backgroundColor: color,
    padding: 3,
    selectable: false,
    evented: false,
    _linkedTo: idx
  })

  fabricCanvas.value.add(rect)
  fabricCanvas.value.add(label)
}

const handleMouseDown = (options) => {
  if (activeTool.value !== 'draw' || !currentTag.value) return
  if (options.target) return // let fabric handle clicks on existing boxes

  const pointer = fabricCanvas.value.getPointer(options.e)
  isDrawing = true
  startX = pointer.x
  startY = pointer.y

  drawingRect = new fabric.Rect({
    left: startX,
    top: startY,
    width: 0,
    height: 0,
    fill: 'rgba(255,255,255,0.05)',
    stroke: getColorForTag(currentTag.value),
    strokeWidth: 2,
    strokeDashArray: [5, 3],
    selectable: false,
    evented: false
  })

  fabricCanvas.value.add(drawingRect)
}

const handleMouseMove = (options) => {
  if (!isDrawing || !drawingRect) return

  const pointer = fabricCanvas.value.getPointer(options.e)
  const width = pointer.x - startX
  const height = pointer.y - startY

  drawingRect.set({
    width: Math.abs(width),
    height: Math.abs(height),
    left: width < 0 ? pointer.x : startX,
    top: height < 0 ? pointer.y : startY
  })
  fabricCanvas.value.renderAll()
}

const handleMouseUp = () => {
  if (!isDrawing || !drawingRect) return
  isDrawing = false

  if (drawingRect.width < 5 || drawingRect.height < 5) {
    fabricCanvas.value.remove(drawingRect)
    drawingRect = null
    return
  }

  const newIdx = regions.value.length
  regions.value.push({
    tag: currentTag.value,
    x: drawingRect.left / imageScale,
    y: drawingRect.top / imageScale,
    width: drawingRect.width / imageScale,
    height: drawingRect.height / imageScale
  })

  dirty.value = true
  fabricCanvas.value.remove(drawingRect)
  drawingRect = null
  drawRegionOnCanvas(regions.value[newIdx], newIdx)
  fabricCanvas.value.renderAll()

  if (!existingTags.value.includes(currentTag.value)) {
    existingTags.value.push(currentTag.value)
  }
  // Auto-switch to select mode after drawing a box
  activeTool.value = 'select'
}

// Move label in sync with rect while dragging
const handleObjectMoving = (options) => {
  const obj = options.target
  if (!obj || obj.regionIndex === undefined) return
  const label = fabricCanvas.value.getObjects().find(o => o._linkedTo === obj.regionIndex)
  if (label) label.set({ left: obj.left + 3, top: obj.top + 3 })
}

const handleObjectModified = (options) => {
  const obj = options.target
  if (!obj || obj.regionIndex === undefined) return

  const absWidth = obj.width * (obj.scaleX || 1)
  const absHeight = obj.height * (obj.scaleY || 1)

  regions.value[obj.regionIndex] = {
    ...regions.value[obj.regionIndex],
    x: obj.left / imageScale,
    y: obj.top / imageScale,
    width: absWidth / imageScale,
    height: absHeight / imageScale
  }

  dirty.value = true

  // Reset scale to 1 so future resizes are not compounded
  obj.set({ width: absWidth, height: absHeight, scaleX: 1, scaleY: 1 })

  // Sync label position after resize/move
  const label = fabricCanvas.value.getObjects().find(o => o._linkedTo === obj.regionIndex)
  if (label) label.set({ left: obj.left + 3, top: obj.top + 3 })
  fabricCanvas.value.renderAll()
}

const handleSelection = (options) => {
  const obj = options.selected?.[0]
  if (obj && obj.regionIndex !== undefined) {
    selectedRegionIndex.value = obj.regionIndex
  }
}

const selectRegion = (idx) => {
  selectedRegionIndex.value = idx
  activeTool.value = 'select'
  const rect = fabricCanvas.value.getObjects().find(obj => obj.regionIndex === idx)
  if (rect) {
    fabricCanvas.value.setActiveObject(rect)
    fabricCanvas.value.renderAll()
  }
}

const deleteRegion = (idx) => {
  regions.value.splice(idx, 1)
  dirty.value = true
  selectedRegionIndex.value = null
  if (imageData.value) loadImageToCanvas()
}

const addNewTag = () => {
  if (newTag.value && !existingTags.value.includes(newTag.value)) {
    existingTags.value.push(newTag.value)
    currentTag.value = newTag.value
    newTag.value = ''
  }
}

/**
 * Persist the current boxes.
 *
 * Returns true only when the server confirmed the write. Callers that are
 * about to navigate away must check the result: silently discarding a failed
 * save loses annotation work that cannot be recovered.
 */
const saveAnnotations = async () => {
  if (saving.value) return false
  saving.value = true
  saveError.value = null
  try {
    // Plain values: the reactive proxies do not survive JSON serialisation
    // of nested fabric-derived objects cleanly.
    const plainRegions = regions.value.map((region) => ({
      tag: region.tag,
      x: Number(region.x) || 0,
      y: Number(region.y) || 0,
      width: Number(region.width) || 0,
      height: Number(region.height) || 0
    }))
    await projectService.saveAnnotations(projectName.value, filename.value, plainRegions)
    dirty.value = false
    savedAt.value = Date.now()
    return true
  } catch (error) {
    saveError.value = errorMessage(error)
    return false
  } finally {
    saving.value = false
  }
}

/** Save before leaving the image; stay put if the save failed. */
const saveThenGo = async (navigate) => {
  if (dirty.value) {
    const saved = await saveAnnotations()
    if (!saved) {
      window.alert('Could not save your annotations: ' + saveError.value +
                   '\nStaying on this image so nothing is lost.')
      return
    }
  }
  navigate()
}

/**
 * Bring the previous image's boxes onto this one.
 *
 * A camera bolted above a line photographs the same object in nearly the same
 * place every time, so on those projects almost every box is a small nudge
 * away from the one before it. Redrawing each from scratch is the bulk of the
 * work and none of the value.
 *
 * The boxes are appended rather than replacing what is already here, and are
 * left unsaved, so nothing is lost if it turns out to be the wrong image to
 * copy from: undo is Ctrl+Z away, or move on without saving.
 */
const copying = ref(false)
const copyNotice = ref('')

const copyFromPrevious = async () => {
  if (currentIndex.value <= 0 || copying.value) return
  const source = allImages.value[currentIndex.value - 1]
  if (!source) return

  copying.value = true
  copyNotice.value = ''
  try {
    const result = await projectService.imageData(projectName.value, source.filename)
    const incoming = result.data?.annotations?.regions || []
    if (!incoming.length) {
      copyNotice.value = `"${source.filename}" has no boxes to copy.`
      return
    }

    // Sizes differ between images on some projects; a box copied onto a
    // smaller frame would sit outside it. Scaling by the ratio of the two
    // keeps it over the same part of the picture.
    const fromW = result.data?.width || 0
    const fromH = result.data?.height || 0
    const toW = imageData.value?.width || 0
    const toH = imageData.value?.height || 0
    const scaleX = fromW && toW ? toW / fromW : 1
    const scaleY = fromH && toH ? toH / fromH : 1

    for (const region of incoming) {
      regions.value.push({
        tag: region.tag,
        x: Math.round(region.x * scaleX),
        y: Math.round(region.y * scaleY),
        width: Math.round(region.width * scaleX),
        height: Math.round(region.height * scaleY),
      })
    }
    dirty.value = true
    copyNotice.value = `${incoming.length} box(es) copied from ` +
      `"${source.filename}". Adjust them, then save.`
  } catch (error) {
    copyNotice.value = errorMessage(error, 'Could not read the previous image.')
  } finally {
    copying.value = false
  }
}

// Navigate: fire-and-forget auto-save, navigate immediately
const previousImage = () => {
  if (currentIndex.value <= 0) return
  const target = allImages.value[currentIndex.value - 1].filename
  saveThenGo(() => router.push({
    name: 'Annotate', params: { name: projectName.value, filename: target }
  }))
}

const nextImage = () => {
  if (currentIndex.value >= totalImages.value - 1) return
  const target = allImages.value[currentIndex.value + 1].filename
  saveThenGo(() => router.push({
    name: 'Annotate', params: { name: projectName.value, filename: target }
  }))
}

const goToTrain = () => {
  saveThenGo(() => router.push({ name: 'Train', params: { name: projectName.value } }))
}

/**
 * Keyboard handling.
 *
 * On a 2000-image project the mouse round trip to the toolbar and the tag
 * list is most of the work. Digits pick a class directly, D and V switch
 * tools, and A/S step through images, so a whole image can be labelled
 * without leaving the canvas.
 *
 * Ignored while a text field has focus, otherwise typing a tag name would
 * fire shortcuts.
 */
const handleKeydown = (e) => {
  const target = e.target
  const typing = target && (
    target.tagName === 'INPUT' ||
    target.tagName === 'TEXTAREA' ||
    target.tagName === 'SELECT' ||
    target.isContentEditable
  )

  if (e.ctrlKey && (e.key === 's' || e.key === 'S')) {
    e.preventDefault()
    saveAnnotations()
    return
  }

  if (typing) return

  if (e.key === 'Delete' || e.key === 'Backspace') {
    e.preventDefault()
    const active = fabricCanvas.value?.getActiveObject()
    if (active && active.regionIndex !== undefined) {
      deleteRegion(active.regionIndex)
    } else if (selectedRegionIndex.value !== null) {
      deleteRegion(selectedRegionIndex.value)
    }
    return
  }

  if (e.key === 'Escape') {
    activeTool.value = 'select'
    isDrawing = false
    if (drawingRect) {
      fabricCanvas.value?.remove(drawingRect)
      drawingRect = null
      fabricCanvas.value?.renderAll()
    }
    return
  }

  // Tools
  if (e.key === 'c' || e.key === 'C') { copyFromPrevious(); return }
  if (e.key === 'd' || e.key === 'D') { activeTool.value = 'draw'; return }
  if (e.key === 'v' || e.key === 'V') { activeTool.value = 'select'; return }

  // Image navigation. Held modifiers are left alone so browser shortcuts work.
  if (e.ctrlKey || e.altKey || e.metaKey) return
  if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') {
    e.preventDefault()
    previousImage()
    return
  }
  if (e.key === 's' || e.key === 'S' || e.key === 'ArrowRight') {
    e.preventDefault()
    nextImage()
    return
  }

  // 1-9 then 0 select the first ten tags, which is exactly the layout of a
  // digit-recognition project.
  if (/^[0-9]$/.test(e.key)) {
    const index = e.key === '0' ? 9 : Number(e.key) - 1
    const tag = existingTags.value[index]
    if (tag) {
      currentTag.value = tag
      // Retagging the selected box is the common case; otherwise the choice
      // applies to the next box drawn.
      if (selectedRegionIndex.value !== null && regions.value[selectedRegionIndex.value]) {
        regions.value[selectedRegionIndex.value].tag = tag
        dirty.value = true
        if (imageData.value) loadImageToCanvas()
      }
    }
  }
}

const shortcuts = [
  { keys: 'D', action: 'Draw box' },
  { keys: 'V', action: 'Select' },
  { keys: '1-9, 0', action: 'Pick class' },
  { keys: 'A / S', action: 'Prev / next image' },
  { keys: 'Del', action: 'Delete box' },
  { keys: 'Esc', action: 'Cancel' },
  { keys: 'Ctrl+S', action: 'Save' }
]

const showShortcuts = ref(false)

const getColorForTag = (tag) => {
  if (!tag) return '#4ECDC4'
  const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
  const hash = tag.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
  return colors[hash % colors.length]
}
</script>

<style scoped>
.annotate-view {
  min-height:100vh;
  background:var(--grad-surface) 100%);
}

/* Annotation Header Bar */
.annotate-header {
  display:flex;
  align-items:center;
  gap:0.875rem;
  padding:0.625rem 1rem;
  background:var(--surface);
  border-bottom:1px solid var(--border-color);
  position:sticky;
  top:0;
  z-index:50;
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

.annotate-title {
  flex:1;
  font-size:0.9375rem;
  color:var(--text-secondary);
  font-weight:500;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}

.header-actions {
  display:flex;
  align-items:center;
  gap:0.5rem;
  flex-shrink:0;
}

.image-counter {
  padding:0.375rem 0.875rem;
  background:var(--gray-100);
  border-radius:8px;
  font-weight:500;
  font-size:0.8125rem;
  color:var(--text-primary);
  white-space:nowrap;
}

/* Annotation Container */
.annotation-container {
  display:grid;
  grid-template-columns:320px 1fr;
  gap:1.5rem;
  padding:2rem;
  max-width:1600px;
  margin:0 auto;
  animation:fadeIn 0.5s ease-out;
}

/* Sidebar */
.sidebar {
  background:var(--surface);
  border-radius:var(--radius-xl);
  padding:1.5rem;
  box-shadow: var(--shadow);
  height:fit-content;
  max-height:calc(100vh - 280px);
  overflow-y:auto;
  border:1px solid var(--border-color);
}

.sidebar-section {
  margin-bottom:1.75rem;
  padding-bottom:1.75rem;
  border-bottom:2px solid var(--border-color);
}

.sidebar-section:last-child {
  border-bottom:none;
  margin-bottom:0;
  padding-bottom:0;
}

.sidebar-title {
  font-size:1rem;
  font-weight:600;
  color:var(--text-primary);
  margin:0 0 1rem 0;
  display:flex;
  align-items:center;
  gap:0.5rem;
}

/* Tool Buttons */
.tool-buttons {
  display:flex;
  flex-direction:column;
  gap:0.75rem;
}

.tool-btn {
  padding:0.875rem 1rem;
  border:2px solid var(--border-color);
  background:var(--surface);
  border-radius:var(--radius-lg);
  cursor:pointer;
  font-size:0.9375rem;
  font-weight:500;
  transition:all 0.2s;
  display:flex;
  align-items:center;
  gap:0.625rem;
  color:var(--text-primary);
}

.tool-btn:hover {
  border-color:var(--primary-400);
  background:var(--primary-50);
  transform:translateX(4px);
}

.tool-btn.active {
  border-color:var(--primary-500);
  background:linear-gradient(135deg, var(--primary-500), var(--primary-600));
  color:var(--text);
  box-shadow: var(--shadow);
}

/* Form Elements */
.form-select,
.form-input {
  width:100%;
  padding:0.75rem;
  border:2px solid var(--border-color);
  border-radius:var(--radius-md);
  font-size:0.9375rem;
  transition:all 0.2s;
  font-family:inherit;
}

.form-select:focus,
.form-input:focus {
  outline:none;
  border-color:var(--primary-500);
  box-shadow: var(--shadow);
}

.new-tag-form {
  display:flex;
  gap:0.5rem;
  margin-top:0.75rem;
}

.btn-sm {
  padding:0.625rem 1rem;
  font-size:0.875rem;
}

/* Regions List */
.regions-list {
  max-height:350px;
  overflow-y:auto;
}

.region-item {
  display:flex;
  justify-content:space-between;
  align-items:center;
  padding:0.875rem;
  margin-bottom:0.625rem;
  background:var(--gray-50);
  border-radius:var(--radius-lg);
  cursor:pointer;
  transition:all 0.2s;
  border:2px solid transparent;
}

.region-item:hover {
  background:var(--primary-50);
  border-color:var(--primary-200);
}

.region-item.selected {
  background:var(--grad-surface));
  border-color:var(--primary-400);
  box-shadow: var(--shadow);
}

.region-info {
  flex:1;
  display:flex;
  flex-direction:column;
  gap:0.375rem;
}

.region-tag {
  font-weight:600;
  color:var(--text-primary);
  font-size:0.9375rem;
}

.region-coords {
  font-size:0.8125rem;
  color:var(--text-secondary);
  font-family:monospace;
}

.btn-delete {
  background:transparent;
  border:none;
  cursor:pointer;
  padding:0.5rem;
  color:var(--text-secondary);
  border-radius:var(--radius-md);
  transition:all 0.2s;
  display:flex;
  align-items:center;
  justify-content:center;
}

.btn-delete:hover {
  background:var(--danger-100);
  color:var(--danger-600);
  transform:scale(1.1);
}

/* Shortcuts */
.shortcuts {
  display:flex;
  flex-direction:column;
  gap:0.625rem;
}

.shortcut {
  display:flex;
  align-items:center;
  gap:0.625rem;
  font-size:0.9375rem;
  color:var(--text-secondary);
}

kbd {
  padding:0.375rem 0.625rem;
  background:var(--gray-100);
  border:1px solid var(--border-color);
  border-radius:var(--radius-md);
  font-family:monospace;
  font-size:0.8125rem;
  font-weight:600;
  color:var(--text-primary);
  box-shadow: var(--shadow-sm);
}

/* Canvas Wrapper */
.canvas-wrapper {
  background:var(--surface);
  border-radius:var(--radius-xl);
  padding:2rem;
  box-shadow: var(--shadow);
  display:flex;
  align-items:center;
  justify-content:center;
  overflow:auto;
  min-height:600px;
  border:1px solid var(--border-color);
}

#canvas {
  border:2px solid var(--border-color);
  border-radius:var(--radius-md);
  background:var(--surface);
  box-shadow: var(--shadow);
}

.loading {
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:1.125rem;
  color:var(--text-secondary);
  min-height:400px;
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
    transform:translateY(-15px) rotate(180deg);
  }
}

/* Responsive Design */
@media (max-width:1024px) {
  .annotation-container {
    grid-template-columns:280px 1fr;
    gap:1rem;
    padding:1.5rem;
  }
  
  .sidebar {
    max-height:calc(100vh - 300px);
  }
}

@media (max-width:768px) {
  .gradient-header {
    padding:1.5rem 1rem 2rem;
  }
  
  .header-top {
    flex-direction:column;
    align-items:stretch;
    gap:1rem;
  }
  
  .header-actions {
    flex-wrap:wrap;
  }
  
  .header-text h1 {
    font-size:1.5rem;
  }
  
  .header-text p {
    font-size:0.9375rem;
  }
  
  .annotation-container {
    grid-template-columns:1fr;
    padding:1rem;
  }
  
  .sidebar {
    max-height:none;
    margin-bottom:1rem;
  }
  
  .canvas-wrapper {
    padding:1rem;
    min-height:400px;
  }
  
  #canvas {
    max-width:100%;
  }
}
</style>

<style scoped>
.copy-notice {
  margin-left:0.6rem;
  padding:0.2rem 0.6rem;
  border-radius:var(--radius-md);
  background:var(--accent-soft);
  color:var(--text-secondary);
  font-size:0.78rem;
}

.save-state {
  font-size:0.75rem;
  font-weight:500;
  padding:0.1875rem 0.5rem;
  border-radius:9999px;
  white-space:nowrap;
}

.save-state.is-saved {
  background:var(--success-soft);
  color:var(--success);
}

.save-state.is-dirty {
  background:var(--warning-soft);
  color:var(--warning);
}

.save-state.is-error {
  background:var(--danger-soft);
  color:var(--danger);
}
</style>

<style scoped>
.shortcut-sheet {
  position: absolute;
  top: calc(100% + 8px);
  right: 1rem;
  z-index: 60;
  min-width: 232px;
  padding: 0.875rem;
  background: var(--surface-2);
  border: 1px solid var(--border-strong);
  border-radius: var(--r-md);
  box-shadow: var(--shadow-lg);
}

.shortcut-sheet h4 {
  margin-bottom: 0.625rem;
  font-size: var(--fs-sm);
  color: var(--text-2);
}

.shortcut-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.25rem 0;
  font-size: var(--fs-sm);
  color: var(--text-2);
}

kbd {
  padding: 0.125rem 0.4375rem;
  border: 1px solid var(--border-strong);
  border-bottom-width: 2px;
  border-radius: var(--r-sm);
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-mono);
  font-size: var(--fs-xs);
  white-space: nowrap;
}

.annotate-header { position: relative; }
</style>

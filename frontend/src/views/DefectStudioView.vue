<template>
  <div class="studio">
    <!-- ── hero ───────────────────────────────────────────────────────── -->
    <header class="hero">
      <div class="hero-mark"><Icon name="layers" size="md" /></div>
      <div class="hero-text">
        <h1>Defect Studio</h1>
        <p>
          A mono camera turns glove colour into one grey level, so a model
          trained on one colour is useless on another and every line needs its
          own defects — the rare half of any dataset. This moves defects you
          already labelled onto this line's good gloves, at the right depth for
          its grey, boxed from what is actually visible afterwards.
        </p>
      </div>
      <ol class="flow">
        <li :class="{ done: goodImages.length }">
          <b>{{ goodImages.length || '–' }}</b><span>good gloves</span>
        </li>
        <li :class="{ done: tags.length }">
          <b>{{ tags.length || '–' }}</b><span>defects</span>
        </li>
        <li :class="{ done: results.length }">
          <b>{{ results.length || '–' }}</b><span>made</span>
        </li>
      </ol>
    </header>

    <div v-if="error" class="banner">{{ error }}</div>

    <!-- ── bring the pictures in; first thing on the page ─────────────── -->
    <section class="card drop-card">
      <div
        :class="['drop', { over: dragging, busy: uploading }]"
        @dragover.prevent="dragging = true"
        @dragleave.prevent="dragging = false"
        @drop.prevent="onDrop"
        @click="uploading ? null : fileInput?.click()"
      >
        <input ref="fileInput" type="file" accept="image/*" multiple
               class="hidden-input" @change="onPick" />
        <div class="drop-ring"><Icon name="upload" size="md" /></div>
        <template v-if="!uploading">
          <h2>Drop this line's good gloves here</h2>
          <p>
            Straight off the camera, no boxes needed. Click to browse, or drag
            a folder's worth in.
            <template v-if="!targetName">
              Somewhere to keep them is made automatically — this is not tied
              to training and nothing has to be prepared first.
            </template>
          </p>
        </template>
        <template v-else>
          <h2>Adding {{ uploadCount }} image(s)…</h2>
          <div class="upload-bar"><div :style="{ width: uploadPercent + '%' }"></div></div>
        </template>
      </div>

      <div class="drop-foot">
        <label class="inline-field">
          <span>Kept in</span>
          <select v-model="targetName" class="control" @change="onTargetChange">
            <option :value="''">a new project</option>
            <option v-for="p in projects" :key="p.name" :value="p.name">
              {{ p.name }} — {{ p.total_images || 0 }} images
            </option>
            <option value="__new__">＋ name a new one…</option>
          </select>
        </label>
        <div v-if="makingNew" class="new-project">
          <input v-model="newName" class="control"
                 placeholder="e.g. black-line-2" @keyup.enter="createTarget" />
          <button class="btn btn-secondary" :disabled="!newName.trim() || creating"
                  @click="createTarget">
            {{ creating ? 'Creating…' : 'Create' }}
          </button>
        </div>
        <span v-if="goodImages.length" class="count-note">
          <b>{{ goodImages.length }}</b> unlabelled image(s) ready. Anything
          already boxed is left alone.
        </span>
      </div>

      <div v-if="goodImages.length" class="strip">
        <figure v-for="img in goodImages.slice(0, 14)" :key="img.filename">
          <img :src="imageUrl(targetName, img.filename)" :alt="img.filename"
               loading="lazy" />
        </figure>
        <div v-if="goodImages.length > 14" class="more">
          +{{ goodImages.length - 14 }}
        </div>
      </div>
    </section>

    <!-- ── the defects ────────────────────────────────────────────────── -->
    <section class="card">
      <div class="card-head">
        <h2>Defects to put on them</h2>
        <span v-if="tags.length" class="pill">{{ tags.length }} chosen</span>
      </div>

      <p class="hint top">
        This wants a <b>dataset</b> — photographs with the defects already
        boxed. A model file is the wrong thing here and cannot stand in for
        one; there is a note below on what a model <em>is</em> good for.
      </p>

      <div class="source-row">
        <label class="inline-field grow">
          <span>Collected from</span>
          <select v-model="sourceName" class="control" @change="loadLibrary">
            <option :value="''">Choose a labelled project…</option>
            <option v-for="p in projects" :key="p.name" :value="p.name">
              {{ p.name }} — {{ p.total_annotations || 0 }} annotated
            </option>
          </select>
        </label>
        <button class="btn btn-secondary" :disabled="!sourceName || collecting"
                @click="collect">
          <Icon name="download" size="sm" />
          <span>{{ collecting ? 'Reading signatures…' : 'Collect defects' }}</span>
        </button>
      </div>

      <p v-if="!library" class="hint">
        Pick the colour you have been running longest — the one whose defects
        are already drawn. Collecting reads the optical density each box added
        to the rubber it was on, which is the part that survives being moved to
        another colour.
      </p>

      <p v-else-if="!library.total" class="hint">
        Nothing collected from that project yet. Press <b>Collect defects</b>.
      </p>

      <!--
        The dropdown lists projects, and somebody arriving with an export from
        another tool has none -- so it offers a list of nothing and no way to
        fill it. Sending them to another page to make a project, import into
        it and come back is three steps for what is one action.
      -->
      <div :class="['mini-drop', { over: datasetOver, busy: importingSet }]"
           @dragover.prevent="datasetOver = true"
           @dragleave.prevent="datasetOver = false"
           @drop.prevent="onDatasetDrop"
           @click="importingSet ? null : datasetInput?.click()">
        <input ref="datasetInput" type="file" accept=".zip" class="hidden-input"
               @change="onDatasetPick" />
        <Icon name="upload" size="sm" />
        <span v-if="!importingSet">
          <b>Not in the list?</b> Drop a <b>.zip of a labelled dataset</b> here
          — YOLO (images + labels + label.txt), COCO (.json) or Pascal VOC
          (.xml). It becomes a project and appears above.
        </span>
        <span v-else>{{ datasetNote || 'Reading the dataset…' }}</span>
      </div>
      <p v-if="datasetNote && !importingSet" class="hint">{{ datasetNote }}</p>

      <!--
        Worth being plain about, because it is the obvious thing to expect and
        it is not true: a model cannot supply defects. Weights hold no
        pictures. What a model can do is find defects in photographs you
        already have, and those photographs then become the source.
      -->
      <details class="aside">
        <summary>No project with defects drawn yet?</summary>
        <p>
          A model file cannot stand in for one — an ONNX holds weights and no
          pictures at all, so there is nothing in it to copy a defect from.
          What a model <em>can</em> do is put the boxes on defect photographs
          you already have:
        </p>
        <ol>
          <li>Make a project and put the defect photographs in it.</li>
          <li>
            Bring the model in below, then
            <router-link to="/projects">open that project</router-link> and
            auto-label with it.
          </li>
          <li>Come back and collect from it.</li>
        </ol>
        <label class="btn btn-secondary">
          <input type="file" accept=".onnx,.pt,.pth,.torchscript,.zip"
                 class="hidden-input" @change="importModel" />
          <Icon name="upload" size="sm" />
          <span>{{ importingModel ? 'Importing…' : 'Import a model file (.onnx, .pt, or a zipped export folder)' }}</span>
        </label>
        <p v-if="modelNote" class="hint">{{ modelNote }}</p>
        <div v-if="modelLabels.length" class="chips quiet">
          <span v-for="name in modelLabels" :key="name" class="chip flat">
            {{ name }}
          </span>
        </div>
      </details>

      <template v-if="library && library.total">
        <div class="chips">
          <button v-for="row in usableTags" :key="row.tag"
                  :class="['chip', row.defect_class, { on: tags.includes(row.tag) }]"
                  @click="toggleTag(row.tag)">
            <i class="dot"></i>
            <span class="chip-name">{{ row.tag }}</span>
            <span class="chip-count">{{ row.count }}</span>
          </button>
        </div>

        <!--
          Shown rather than hidden. These are real classes in the dataset and
          somebody looking for one needs to find out why it is not on offer,
          not wonder where it went.
        -->
        <div v-if="wholeGloveTags.length" class="unusable">
          <p class="hint">
            These describe the whole glove rather than a mark on it — their
            boxes cover the glove. Nothing can be added to a good glove to make
            one true, so they cannot be synthesised:
          </p>
          <div class="chips">
            <span v-for="row in wholeGloveTags" :key="row.tag" class="chip off">
              <span class="chip-name">{{ row.tag }}</span>
              <span class="chip-count">{{ Math.round(row.share * 100) }}% of the glove</span>
            </span>
          </div>
        </div>
        <div class="legend">
          <span><i class="dot hole"></i> light through — the same brightness on
            every colour</span>
          <span><i class="dot stain"></i> more or less rubber — the same
            <em>density</em>, a different grey</span>
        </div>
      </template>
    </section>

    <!-- ── run ────────────────────────────────────────────────────────── -->
    <section class="card run-card">
      <!--
        Where the glove is, in the pictures being worked on. Taken from their
        own labels when they have them, which a dataset with Good on every
        frame gives for nothing. A model is only needed when they do not.
      -->
      <label class="inline-field wide">
        <span>Find the glove with</span>
        <select v-model="gloveModel" class="control">
          <option :value="''">the boxes already on the pictures</option>
          <option v-for="m in models" :key="m.path" :value="m.path">
            {{ m.project }} / {{ m.label }}
          </option>
        </select>
      </label>
      <p class="hint">
        A picture whose glove cannot be found is skipped and counted, never
        guessed at — guessing is what put defects on the machinery.
      </p>

      <div class="run-row">
        <button v-if="!running" class="btn btn-primary big" :disabled="!canRun"
                @click="run">
          <Icon name="zap" size="sm" />
          <span>Add defects to {{ goodImages.length }} image(s)</span>
        </button>
        <button v-else class="btn btn-danger big" @click="stop">
          <Icon name="x" size="sm" />
          <span>Stop</span>
        </button>

        <label class="inline-field">
          <span>per image</span>
          <input v-model.number="perImage" type="number" min="1" max="4"
                 class="control narrow" />
        </label>

        <span v-if="!canRun && !running" class="hint inline">{{ whyNotReady }}</span>

        <div class="run-spacer"></div>

        <button v-if="results.length" class="btn btn-secondary"
                :disabled="downloading" @click="download">
          <Icon name="download" size="sm" />
          <span>{{ downloading ? 'Packing…' : 'Download dataset' }}</span>
        </button>
        <router-link v-if="results.length && targetName"
                     :to="`/projects/${encodeURIComponent(targetName)}/train`"
                     class="btn btn-secondary">
          <Icon name="rocket" size="sm" />
          <span>Train on this</span>
        </router-link>
      </div>

      <div v-if="job" class="job">
        <div class="job-line">{{ job.message || job.status }}</div>
        <div v-if="running" class="job-bar">
          <div class="job-fill" :style="{ width: percent + '%' }"></div>
        </div>
        <dl v-if="job.profile" class="measured">
          <div>
            <dt>Glove</dt><dd>{{ Math.round(job.profile.glove_level) }}</dd>
            <i class="swatch" :style="swatch(job.profile.glove_level)"></i>
          </div>
          <div>
            <dt>Backlight</dt><dd>{{ Math.round(job.profile.bg_level) }}</dd>
            <i class="swatch" :style="swatch(job.profile.bg_level)"></i>
          </div>
          <div><dt>Noise</dt><dd>{{ job.profile.noise_sigma }}</dd></div>
          <div class="good"><dt>Made</dt><dd>{{ job.made || 0 }}</dd></div>
          <div :class="{ warn: job.refused }">
            <dt>Refused</dt><dd>{{ job.refused || 0 }}</dd>
          </div>
          <div v-if="job.no_glove" class="warn">
            <dt>No glove</dt><dd>{{ job.no_glove }}</dd>
          </div>
        </dl>
        <p v-if="job.no_glove" class="hint">
          Those pictures were skipped because the glove could not be found in
          them — either they carry no box and no model was chosen, or the model
          found none. Nothing was placed anywhere for the sake of placing it.
        </p>
        <p v-if="job.refused" class="hint">
          Refused means too faint to see against this colour's own noise —
          thrown out rather than written in. A defect nobody can see teaches
          the detector that ordinary rubber is a tear.
        </p>
      </div>
    </section>

    <!-- ── results ────────────────────────────────────────────────────── -->
    <section v-if="results.length" class="card">
      <div class="card-head">
        <h2>Made, and already boxed</h2>
        <span class="pill">{{ results.length }}</span>
      </div>
      <p class="hint">
        Click any of them to see the good glove it was made from, side by side.
        The box comes from what changed by more than this line's own noise, so
        the same defect is boxed smaller on a darker glove.
      </p>

      <div class="gallery">
        <figure v-for="img in results" :key="img.filename" class="shot"
                @click="preview = img">
          <div class="frame">
            <img :src="imageUrl(targetName, img.filename)" :alt="img.filename"
                 loading="lazy" />
            <svg v-if="img.width" class="overlay"
                 :viewBox="`0 0 ${img.width} ${img.height}`"
                 preserveAspectRatio="none">
              <rect v-for="(box, i) in img.boxes || []" :key="i"
                    :x="box[0]" :y="box[1]" :width="box[2]" :height="box[3]"
                    class="box" />
            </svg>
            <span class="zoom"><Icon name="search" size="sm" /></span>
          </div>
          <figcaption>
            <span v-for="tag in tagsOf(img)" :key="tag" class="tag">{{ tag }}</span>
          </figcaption>
        </figure>
      </div>
    </section>

    <!-- ── before and after ───────────────────────────────────────────── -->
    <div v-if="preview" class="lightbox" @click="preview = null">
      <div class="lightbox-inner" @click.stop>
        <div class="compare">
          <figure>
            <figcaption>Good glove, as it came off the line</figcaption>
            <div class="frame">
              <img v-if="preview.original_name"
                   :src="imageUrl(targetName, preview.original_name)" alt="before" />
              <p v-else class="missing">The original is not in this project.</p>
            </div>
          </figure>
          <figure>
            <figcaption>With the defect, and its box</figcaption>
            <div class="frame">
              <img :src="imageUrl(targetName, preview.filename)" alt="after" />
              <svg v-if="preview.width" class="overlay"
                   :viewBox="`0 0 ${preview.width} ${preview.height}`"
                   preserveAspectRatio="none">
                <rect v-for="(box, i) in preview.boxes || []" :key="i"
                      :x="box[0]" :y="box[1]" :width="box[2]" :height="box[3]"
                      class="box" />
              </svg>
            </div>
          </figure>
        </div>
        <div class="lightbox-bar">
          <span v-for="tag in tagsOf(preview)" :key="tag" class="tag">{{ tag }}</span>
          <button class="btn btn-secondary" @click="preview = null">Close</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import Icon from '@/components/Icon.vue'
import { errorMessage, projectService, trainingService } from '@/services'

const projects = ref([])
const sourceName = ref('')
const targetName = ref('')
const library = ref(null)
const tags = ref([])
const perImage = ref(1)
const goodImages = ref([])
const results = ref([])
const job = ref(null)
const error = ref('')
const collecting = ref(false)
const uploading = ref(false)
const uploadPercent = ref(0)
const uploadCount = ref(0)
const dragging = ref(false)
const preview = ref(null)
const fileInput = ref(null)
const makingNew = ref(false)
const newName = ref('')
const creating = ref(false)
const downloading = ref(false)
const models = ref([])
const gloveModel = ref('')

// Classes whose boxes cover the glove are statements about the whole piece,
// not marks on it. Kept apart rather than mixed in: ticking one produces
// pictures that look made and teach the wrong thing.
const usableTags = computed(
  () => (library.value?.per_tag || []).filter((row) => !row.whole_glove))
const wholeGloveTags = computed(
  () => (library.value?.per_tag || []).filter((row) => row.whole_glove))
const importingModel = ref(false)
const modelNote = ref('')
const modelLabels = ref([])
const importingSet = ref(false)
const datasetOver = ref(false)
const datasetNote = ref('')
const datasetInput = ref(null)

/**
 * A labelled dataset, brought in where it is needed.
 *
 * The list above holds projects, and somebody arriving with an export from
 * another tool has none. This makes the project, imports into it and selects
 * it -- three steps that otherwise happen on two other pages.
 */
const importDataset = async (file) => {
  if (!file || importingSet.value) return
  importingSet.value = true
  datasetNote.value = ''
  error.value = ''
  try {
    const now = new Date()
    const stamp = `${now.getFullYear()}`
      + `${String(now.getMonth() + 1).padStart(2, '0')}`
      + `${String(now.getDate()).padStart(2, '0')}`
    let name = `defects-${stamp}`
    let suffix = 2
    const taken = new Set(projects.value.map((p) => p.name))
    while (taken.has(name)) name = `defects-${stamp}-${suffix++}`

    await projectService.create(name)
    const result = await projectService.importDataset(name, file)
    await refreshProjects()
    sourceName.value = name

    if (result.job) {
      datasetNote.value = `Reading ${result.job.total || ''} annotation(s)…`
      await waitForDataset(name)
    } else {
      datasetNote.value = result.message || 'Imported.'
    }
    await loadLibrary()
  } catch (err) {
    datasetNote.value = errorMessage(err, 'That dataset could not be read')
  } finally {
    importingSet.value = false
  }
}

const waitForDataset = (name) => new Promise((resolve) => {
  const tick = async () => {
    try {
      const { job } = await projectService.datasetImportStatus(name)
      if (job?.status === 'running') {
        datasetNote.value = job.message || 'Reading…'
        setTimeout(tick, 1200)
        return
      }
      datasetNote.value = job?.message || 'Imported.'
      await refreshProjects()
    } catch {
      datasetNote.value = 'Imported.'
    }
    resolve()
  }
  tick()
})

const onDatasetPick = (event) => {
  importDataset(event.target.files?.[0])
  event.target.value = ''
}

const onDatasetDrop = (event) => {
  datasetOver.value = false
  importDataset(event.dataTransfer?.files?.[0])
}

/**
 * Bring a detector in from here as well as from the annotation toolbar.
 *
 * Not because it can supply defects -- it cannot, weights hold no pictures --
 * but because the route that does work starts with one: a model puts boxes on
 * defect photographs, and the boxed photographs are then a source to collect
 * from. Making somebody go elsewhere for the first step of a job that starts
 * on this page is the kind of friction this page exists to remove.
 */
const importModel = async (event) => {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file || importingModel.value) return
  importingModel.value = true
  modelNote.value = ''
  try {
    const record = await trainingService.importModel(file)
    const detail = record.detail || {}
    modelLabels.value = record.labels || []
    const parts = [detail.source ? `Recognised as ${detail.source}` : 'Imported']
    if (modelLabels.value.length) {
      parts.push(`${modelLabels.value.length} class name(s), listed below`)
    } else {
      parts.push('no class names — add its labels.txt in the annotator')
    }
    parts.push('it is now in the model list on every project, for auto-labelling')
    modelNote.value = parts.join(' — ')
  } catch (err) {
    modelNote.value = errorMessage(err, 'That model could not be imported')
  } finally {
    importingModel.value = false
  }
}

const running = computed(() => job.value?.status === 'running')
const canRun = computed(
  () => tags.value.length > 0 && targetName.value && goodImages.value.length > 0)
const percent = computed(() => {
  if (!job.value?.total) return 0
  return Math.round(((job.value.done || 0) / job.value.total) * 100)
})
const whyNotReady = computed(() => {
  if (!goodImages.value.length) return 'Add this line\'s good gloves above.'
  if (!tags.value.length) return 'Collect some defects and choose which to use.'
  return ''
})

const imageUrl = (project, filename) =>
  `/api/projects/${encodeURIComponent(project)}/images/${encodeURIComponent(filename)}/raw`

// The index already carries every image's boxes, so the gallery draws them
// from the one request it makes rather than asking per picture.
const tagsOf = (img) => [...new Set((img.boxes || []).map((b) => b[4]))]

// The measured glove and backlight levels are greys, and showing them as greys
// says more at a glance than the number does.
const swatch = (level) => {
  const v = Math.max(0, Math.min(255, Math.round(level || 0)))
  return { background: `rgb(${v},${v},${v})` }
}

const toggleTag = (tag) => {
  const at = tags.value.indexOf(tag)
  if (at === -1) tags.value.push(tag)
  else tags.value.splice(at, 1)
}

const refreshProjects = async () => {
  projects.value = (await projectService.list()).projects || []
}

const loadLibrary = async () => {
  library.value = null
  tags.value = []
  if (!sourceName.value) return
  try {
    library.value = await projectService.defectLibrary(sourceName.value)
    tags.value = (library.value.per_tag || [])
      .filter((row) => !row.whole_glove).map((row) => row.tag)
  } catch {
    library.value = { total: 0, per_tag: [] }
  }
}

const collect = async () => {
  if (!sourceName.value || collecting.value) return
  collecting.value = true
  error.value = ''
  try {
    library.value = await projectService.collectDefectLibrary(sourceName.value,
                                                              { per_label: 40 })
    // Everything usable chosen to start with: unticking a few beats hunting
    // for them, and the ones that cannot be made are not ticked at all.
    tags.value = (library.value.per_tag || [])
      .filter((row) => !row.whole_glove).map((row) => row.tag)
  } catch (err) {
    error.value = errorMessage(err, 'Those defects could not be collected')
  } finally {
    collecting.value = false
  }
}

const onTargetChange = async () => {
  if (targetName.value === '__new__') {
    targetName.value = ''
    makingNew.value = true
    return
  }
  makingNew.value = false
  await loadTarget()
}

const createTarget = async () => {
  const name = newName.value.trim()
  if (!name || creating.value) return
  creating.value = true
  error.value = ''
  try {
    await projectService.create(name)
    await refreshProjects()
    targetName.value = name
    makingNew.value = false
    newName.value = ''
    await loadTarget()
  } catch (err) {
    error.value = errorMessage(err, 'That project could not be created')
  } finally {
    creating.value = false
  }
}

/**
 * Somewhere to put the pictures, without being asked for one first.
 *
 * Requiring a project to exist before a file can be dropped is friction for no
 * reason: a project is only where the images and their boxes are kept, and
 * this has nothing to do with training. Dropping files with none chosen makes
 * one, named after the day, and the picker then shows which.
 */
const ensureTarget = async () => {
  if (targetName.value) return targetName.value
  const now = new Date()
  const stamp = `${now.getFullYear()}`
    + `${String(now.getMonth() + 1).padStart(2, '0')}`
    + `${String(now.getDate()).padStart(2, '0')}`
  let name = `defect-studio-${stamp}`
  let suffix = 2
  const taken = new Set(projects.value.map((p) => p.name))
  while (taken.has(name)) name = `defect-studio-${stamp}-${suffix++}`
  await projectService.create(name)
  await refreshProjects()
  targetName.value = name
  return name
}

const loadTarget = async () => {
  goodImages.value = []
  results.value = []
  job.value = null
  if (!targetName.value) return
  try {
    const status = await projectService.defectSynthStatus(targetName.value)
    job.value = status.job || null
    await refreshImages()
    if (job.value?.status === 'running') poll()
  } catch (err) {
    error.value = errorMessage(err, 'That project could not be read')
  }
}

// A good glove is one with no defect on it -- which is not the same as one
// with no boxes. An export holds both kinds together and boxes the glove in
// both, so treating any boxed picture as already dealt with left nothing to
// work on and reported an empty project that was full.
const GLOVE_CLASSES = ['good', 'good2', 'nonbad', 'glove', 'ok']
const isGoodGlove = (image) => {
  if (image.augmented) return false
  const tags = image.tags || []
  return !tags.length || tags.every((t) => GLOVE_CLASSES.includes(String(t).toLowerCase()))
}

const refreshImages = async () => {
  if (!targetName.value) return
  const { images } = await projectService.images(targetName.value)
  const all = images || []
  goodImages.value = all.filter(isGoodGlove)
  const batch = job.value?.batch
  results.value = batch
    ? all.filter((i) => i.batch === batch && i.annotated)
    : []
}

const upload = async (files) => {
  const picked = Array.from(files || []).filter((f) => f.type.startsWith('image/'))
  if (!picked.length) return
  uploading.value = true
  uploadCount.value = picked.length
  uploadPercent.value = 0
  error.value = ''
  try {
    const into = await ensureTarget()
    await projectService.uploadImages(into, picked,
                                      (p) => { uploadPercent.value = p })
    await refreshImages()
  } catch (err) {
    error.value = errorMessage(err, 'Those images could not be added')
  } finally {
    uploading.value = false
  }
}

const onPick = (event) => {
  upload(event.target.files)
  event.target.value = ''
}

const onDrop = (event) => {
  dragging.value = false
  upload(event.dataTransfer?.files)
}

const run = async () => {
  error.value = ''
  results.value = []
  try {
    const { job: started } = await projectService.startDefectSynth(
      targetName.value,
      { source_project: sourceName.value, tags: tags.value,
        per_image: perImage.value,
        model_path: gloveModel.value || undefined })
    job.value = started
    poll()
  } catch (err) {
    error.value = errorMessage(err, 'The run could not be started')
  }
}

const stop = async () => {
  try {
    await projectService.cancelDefectSynth(targetName.value)
  } catch (err) {
    error.value = errorMessage(err, 'It could not be stopped')
  }
}

const download = async () => {
  if (!targetName.value || downloading.value) return
  downloading.value = true
  try {
    const blob = await projectService.exportDataset(targetName.value)
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${targetName.value}-with-defects.zip`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  } catch (err) {
    error.value = errorMessage(err, 'The dataset could not be packed')
  } finally {
    downloading.value = false
  }
}

let timer = null
const poll = () => {
  clearTimeout(timer)
  timer = setTimeout(async () => {
    try {
      const { job: latest } = await projectService.defectSynthStatus(targetName.value)
      job.value = latest
      if (latest?.status === 'running') poll()
      else await refreshImages()
    } catch {
      /* the next poll picks it up */
    }
  }, 1200)
}

onMounted(async () => {
  try {
    await refreshProjects()
    try {
      models.value = (await trainingService.listTrainedModels())
        .filter((m) => m.checkpoint === 'best')
    } catch { /* the picker simply offers none */ }
  } catch (err) {
    error.value = errorMessage(err, 'The projects could not be listed')
  }
})

onBeforeUnmount(() => clearTimeout(timer))
</script>

<style scoped>
.studio { padding: 1.5rem; max-width: 78rem; }

/* ── hero ─────────────────────────────────────────────────────────── */
.hero {
  display: flex;
  gap: 1.1rem;
  align-items: flex-start;
  flex-wrap: wrap;
  padding: 1.3rem 1.5rem;
  margin-bottom: 1.1rem;
  border-radius: var(--radius-xl, 18px);
  border: 1px solid var(--border-color, var(--border));
  background:
    radial-gradient(90% 180% at 0% 0%, var(--accent-softer), transparent 55%),
    var(--surface, var(--bg-subtle));
}
.hero-mark {
  display: grid;
  place-items: center;
  width: 2.7rem;
  height: 2.7rem;
  flex: none;
  border-radius: 14px;
  color: #fff;
  background: var(--grad-accent, var(--accent));
  box-shadow: var(--shadow-accent, none);
}
.hero-text { flex: 1 1 24rem; }
.hero-text h1 {
  margin: 0 0 0.35rem;
  font-size: 1.35rem;
  color: var(--text-primary, var(--text));
}
.hero-text p {
  margin: 0;
  max-width: 48rem;
  font-size: 0.82rem;
  line-height: 1.65;
  color: var(--text-tertiary, var(--text-3));
}
.flow { display: flex; gap: 0.5rem; margin: 0; padding: 0; list-style: none; }
.flow li {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 5.2rem;
  padding: 0.55rem 0.7rem;
  border-radius: 12px;
  border: 1px solid var(--border-color, var(--border));
  background: var(--bg);
}
.flow b {
  font-size: 1.2rem;
  font-variant-numeric: tabular-nums;
  color: var(--text-tertiary, var(--text-3));
}
.flow span {
  font-size: 0.64rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-tertiary, var(--text-3));
}
.flow li.done { border-color: var(--accent); background: var(--accent-softer); }
.flow li.done b { color: var(--accent); }

.banner {
  margin-bottom: 1rem;
  padding: 0.65rem 0.85rem;
  border-radius: 10px;
  font-size: 0.82rem;
  color: var(--danger);
  border: 1px solid var(--danger);
}

/* ── cards ────────────────────────────────────────────────────────── */
.card {
  padding: 1.15rem;
  margin-bottom: 1rem;
  border: 1px solid var(--border-color, var(--border));
  border-radius: var(--radius-xl, 18px);
  background: var(--surface, var(--bg-subtle));
}
.card-head {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.6rem;
}
.card-head h2 {
  margin: 0;
  font-size: 1rem;
  color: var(--text-primary, var(--text));
}
.pill {
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 700;
  background: var(--accent-softer);
  color: var(--accent);
}

/* ── the drop zone, which is the point of the page ────────────────── */
.drop-card { padding: 0.9rem; }
.drop {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.3rem;
  padding: 2.2rem 1.5rem;
  border: 2px dashed var(--border-strong, var(--border));
  border-radius: 16px;
  text-align: center;
  cursor: pointer;
  background:
    radial-gradient(70% 120% at 50% 0%, var(--accent-softer), transparent 70%);
  transition: border-color 0.18s ease, background 0.18s ease;
}
.drop:hover { border-color: var(--accent); }
.drop.over { border-color: var(--accent); background: var(--accent-soft, var(--accent-softer)); }
.drop.busy { cursor: default; }
.drop h2 {
  margin: 0.5rem 0 0;
  font-size: 1.05rem;
  color: var(--text-primary, var(--text));
}
.drop p {
  margin: 0.25rem 0 0;
  max-width: 36rem;
  font-size: 0.8rem;
  line-height: 1.6;
  color: var(--text-tertiary, var(--text-3));
}
.drop-ring {
  display: grid;
  place-items: center;
  width: 3rem;
  height: 3rem;
  border-radius: 999px;
  color: var(--accent);
  background: var(--accent-softer);
  border: 1px solid var(--accent-soft, var(--accent-softer));
}
.hidden-input { display: none; }
.upload-bar {
  width: min(22rem, 70%);
  height: 6px;
  margin-top: 0.7rem;
  border-radius: 999px;
  background: var(--border-color, var(--border));
  overflow: hidden;
}
.upload-bar div {
  height: 100%;
  background: var(--grad-accent, var(--accent));
  transition: width 0.25s ease;
}

.drop-foot {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-top: 0.8rem;
  padding: 0 0.3rem;
}
.count-note { font-size: 0.76rem; color: var(--text-tertiary, var(--text-3)); }
.count-note b { color: var(--accent); }
.new-project { display: flex; gap: 0.4rem; }
.new-project .control { width: 14rem; }

/* ── fields ───────────────────────────────────────────────────────── */
.inline-field { display: flex; align-items: center; gap: 0.45rem; }
.inline-field.grow { flex: 1 1 20rem; }
.inline-field > span {
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  white-space: nowrap;
  color: var(--text-tertiary, var(--text-3));
}
.control {
  flex: 1;
  background: var(--bg);
  border: 1px solid var(--border-color, var(--border));
  border-radius: 10px;
  padding: 0.45rem 0.65rem;
  font-family: inherit;
  font-size: 0.84rem;
  color: var(--text-primary, var(--text));
}
.control.narrow { flex: none; width: 3.6rem; }

.hint {
  margin: 0.65rem 0 0;
  font-size: 0.78rem;
  line-height: 1.6;
  color: var(--text-tertiary, var(--text-3));
}
.hint.inline { margin: 0; }

/* ── defect chips ─────────────────────────────────────────────────── */
.source-row { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.85rem; }
.chip {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.65rem;
  border-radius: 999px;
  border: 1px solid var(--border-color, var(--border));
  background: var(--bg);
  cursor: pointer;
  font-family: inherit;
  font-size: 0.77rem;
  color: var(--text-tertiary, var(--text-3));
  transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
}
.chip:hover { border-color: var(--border-strong, var(--accent)); }
.chip-name { font-weight: 600; }
.chip-count { opacity: 0.6; font-variant-numeric: tabular-nums; }
.chip.on {
  color: var(--text-primary, var(--text));
  border-color: var(--accent);
  background: var(--accent-softer);
}
.dot {
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 999px;
  flex: none;
  background: var(--text-tertiary, var(--text-3));
}
/* Light through, versus more or less rubber. */
.chip.hole .dot, .chip.tear .dot, .dot.hole { background: var(--amber, #e0a63c); }
.chip.thin .dot, .chip.thick .dot { background: var(--cyan, #57b6d8); }
.chip.stain .dot, .chip.particle .dot, .dot.stain { background: var(--accent); }

.mini-drop {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-top: 0.8rem;
  padding: 0.7rem 0.85rem;
  border: 1.5px dashed var(--border-strong, var(--border));
  border-radius: 12px;
  cursor: pointer;
  font-size: 0.78rem;
  line-height: 1.55;
  color: var(--text-tertiary, var(--text-3));
  transition: border-color 0.15s ease, background 0.15s ease;
}
.mini-drop:hover { border-color: var(--accent); }
.mini-drop.over { border-color: var(--accent); background: var(--accent-softer); }
.mini-drop.busy { cursor: default; }
.mini-drop b { color: var(--text-primary, var(--text)); }

.hint.top { margin: 0 0 0.7rem; }
.chips.quiet { margin-top: 0.6rem; }
.chip.flat { cursor: default; padding: 0.2rem 0.5rem; font-size: 0.7rem; }

.aside {
  margin-top: 0.8rem;
  padding: 0.7rem 0.85rem;
  border-radius: 12px;
  border: 1px solid var(--border-color, var(--border));
  background: var(--bg);
}
.aside > summary {
  cursor: pointer;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-primary, var(--text));
}
.aside p, .aside ol {
  margin: 0.55rem 0 0;
  font-size: 0.78rem;
  line-height: 1.6;
  color: var(--text-tertiary, var(--text-3));
}
.aside ol { padding-left: 1.1rem; }
.aside li { margin-bottom: 0.2rem; }
.aside .btn { margin-top: 0.7rem; }

.unusable { margin-top: 0.9rem; }
.chip.off {
  cursor: default;
  opacity: 0.55;
  border-style: dashed;
}
.inline-field.wide { width: 100%; margin-bottom: 0.2rem; }

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 1.1rem;
  margin-top: 0.7rem;
  font-size: 0.72rem;
  color: var(--text-tertiary, var(--text-3));
}
.legend span { display: flex; align-items: center; gap: 0.35rem; }

/* ── thumbnails ───────────────────────────────────────────────────── */
.strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin-top: 0.8rem;
  align-items: center;
}
.strip figure { margin: 0; }
.strip img {
  width: 4rem;
  height: 4rem;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid var(--border-color, var(--border));
  background: #000;
}
.more { font-size: 0.76rem; color: var(--text-tertiary, var(--text-3)); }

/* ── run ──────────────────────────────────────────────────────────── */
.run-row { display: flex; gap: 0.7rem; align-items: center; flex-wrap: wrap; }
.run-spacer { flex: 1 1 auto; }
.btn.big { padding: 0.6rem 1.1rem; font-size: 0.88rem; }

.job { margin-top: 0.95rem; }
.job-line { font-size: 0.82rem; color: var(--text-primary, var(--text)); }
.job-bar {
  margin-top: 0.5rem;
  height: 6px;
  border-radius: 999px;
  background: var(--border-color, var(--border));
  overflow: hidden;
}
.job-fill {
  height: 100%;
  background: var(--grad-accent, var(--accent));
  transition: width 0.3s ease;
}
.measured { display: flex; flex-wrap: wrap; gap: 1.5rem; margin: 0.95rem 0 0; }
.measured div {
  display: grid;
  grid-template-columns: auto auto;
  align-items: center;
  gap: 0 0.45rem;
}
.measured dt {
  grid-column: 1 / -1;
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-tertiary, var(--text-3));
}
.measured dd {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: var(--text-primary, var(--text));
}
.measured .good dd { color: var(--accent); }
.measured .warn dd { color: var(--amber, #e0a63c); }
.swatch {
  width: 1rem;
  height: 1rem;
  border-radius: 4px;
  border: 1px solid var(--border-color, var(--border));
}

/* ── results ──────────────────────────────────────────────────────── */
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(11rem, 1fr));
  gap: 0.75rem;
  margin-top: 0.9rem;
}
.shot { margin: 0; cursor: zoom-in; }
.frame {
  position: relative;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--border-color, var(--border));
  background: #000;
}
.frame img { display: block; width: 100%; }
.overlay { position: absolute; inset: 0; width: 100%; height: 100%; }
.box {
  fill: none;
  stroke: #3ddc9a;
  stroke-width: 6;
  vector-effect: non-scaling-stroke;
}
.zoom {
  position: absolute;
  right: 0.4rem;
  bottom: 0.4rem;
  display: grid;
  place-items: center;
  width: 1.7rem;
  height: 1.7rem;
  border-radius: 8px;
  opacity: 0;
  color: #fff;
  background: rgba(0, 0, 0, 0.55);
  transition: opacity 0.15s ease;
}
.shot:hover .zoom { opacity: 1; }
.shot figcaption {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
  margin-top: 0.4rem;
}
.tag {
  padding: 0.12rem 0.5rem;
  border-radius: 999px;
  font-size: 0.68rem;
  font-weight: 600;
  background: var(--accent-softer);
  color: var(--accent);
}

/* ── before and after ─────────────────────────────────────────────── */
.lightbox {
  position: fixed;
  inset: 0;
  z-index: 60;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  background: rgba(0, 0, 0, 0.85);
}
.lightbox-inner { width: min(94vw, 68rem); }
.compare {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr));
  gap: 0.8rem;
}
.compare figure { margin: 0; }
.compare figcaption {
  margin-bottom: 0.35rem;
  font-size: 0.74rem;
  color: rgba(255, 255, 255, 0.72);
}
.compare .frame img { max-height: 66vh; object-fit: contain; margin: 0 auto; }
.missing {
  margin: 0;
  padding: 3rem 1rem;
  text-align: center;
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.6);
}
.lightbox-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.8rem;
}
.lightbox-bar .btn { margin-left: auto; }
</style>

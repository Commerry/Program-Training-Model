<template>
  <div class="studio">
    <header class="hero">
      <div class="hero-text">
        <h1><Icon name="layers" size="md" /> Defect Studio</h1>
        <p>
          A mono camera turns glove colour into one grey level, so a model
          trained on one colour is useless on another and every line needs its
          own defects — the rare half of any dataset. This moves defects you
          already labelled onto this line's good gloves, at the right depth for
          its grey, boxed from what is actually visible afterwards.
        </p>
      </div>
      <div class="hero-flow">
        <span :class="['flow-step', { done: tags.length }]">Defects</span>
        <span class="flow-arrow" aria-hidden="true">&rarr;</span>
        <span :class="['flow-step', { done: goodImages.length }]">Good gloves</span>
        <span class="flow-arrow" aria-hidden="true">&rarr;</span>
        <span :class="['flow-step', { done: results.length }]">Ready to train</span>
      </div>
    </header>

    <div v-if="error" class="banner error">{{ error }}</div>

    <div class="columns">
      <!-- ── 1. the defects ─────────────────────────────────────────── -->
      <section class="card">
        <div class="card-head">
          <span class="num">1</span>
          <h2>Defects to use</h2>
          <span v-if="tags.length" class="pill">{{ tags.length }} selected</span>
        </div>

        <label class="field">
          <span>From project</span>
          <select v-model="sourceName" class="control" @change="loadLibrary">
            <option :value="''">Choose a labelled project…</option>
            <option v-for="p in projects" :key="p.name" :value="p.name">
              {{ p.name }} — {{ p.total_annotations || 0 }} annotated
            </option>
          </select>
        </label>

        <button class="btn btn-secondary wide" :disabled="!sourceName || collecting"
                @click="collect">
          <Icon name="download" size="sm" />
          <span>{{ collecting ? 'Reading signatures…' : 'Collect defects' }}</span>
        </button>

        <p v-if="library && !library.total" class="note">
          Nothing collected yet. This reads the optical density each box added
          to the rubber it was on — the part that survives being moved to
          another colour.
        </p>

        <div v-if="library && library.total" class="chips">
          <button v-for="row in library.per_tag" :key="row.tag"
                  :class="['chip', row.defect_class, { on: tags.includes(row.tag) }]"
                  @click="toggleTag(row.tag)">
            <span class="chip-name">{{ row.tag }}</span>
            <span class="chip-count">{{ row.count }}</span>
            <span class="chip-kind">{{ row.defect_class }}</span>
          </button>
        </div>
        <p v-if="library && library.total" class="note tiny">
          The kind decides the physics: <strong>hole</strong> and
          <strong>tear</strong> let the backlight through, so they land at the
          same brightness on every colour. The rest add optical density, so
          they land at the same <em>density</em> and a different grey.
        </p>
      </section>

      <!-- ── 2. the good gloves ─────────────────────────────────────── -->
      <section class="card">
        <div class="card-head">
          <span class="num">2</span>
          <h2>This line's good gloves</h2>
          <span v-if="goodImages.length" class="pill">{{ goodImages.length }} ready</span>
        </div>

<!--
          This has nothing to do with training and is not tied to it. A project
          is only where the pictures and their boxes are kept, so needing one
          that already exists is friction for no reason: name a new one and it
          is made here.
        -->
        <label class="field">
          <span>Keep them in</span>
          <select v-model="targetName" class="control" @change="onTargetChange">
            <option :value="''">Choose a project…</option>
            <option v-for="p in projects" :key="p.name" :value="p.name">
              {{ p.name }} — {{ p.total_images || 0 }} images
            </option>
            <option value="__new__">＋ Start a new one…</option>
          </select>
        </label>

        <div v-if="makingNew" class="step-row new-project">
          <input v-model="newName" class="control" placeholder="Name for this line, e.g. black-line-2"
                 @keyup.enter="createTarget" />
          <button class="btn btn-secondary" :disabled="!newName.trim() || creating"
                  @click="createTarget">
            <Icon name="plus" size="sm" />
            <span>{{ creating ? 'Creating…' : 'Create' }}</span>
          </button>
        </div>

        <div v-if="targetName"
             :class="['drop', { over: dragging, busy: uploading }]"
             @dragover.prevent="dragging = true"
             @dragleave.prevent="dragging = false"
             @drop.prevent="onDrop">
          <input ref="fileInput" type="file" accept="image/*" multiple
                 class="hidden-input" @change="onPick" />
          <Icon name="upload" size="md" />
          <p v-if="!uploading">
            Drop good glove photographs here, or
            <button class="link" @click="fileInput?.click()">browse</button>
          </p>
          <p v-else>Uploading… {{ uploadPercent }}%</p>
          <span class="note tiny">
            Only images with no boxes are used — anything already labelled is
            left alone.
          </span>
        </div>

        <div v-if="goodImages.length" class="strip">
          <figure v-for="img in goodImages.slice(0, 12)" :key="img.filename">
            <img :src="imageUrl(targetName, img.filename)" :alt="img.filename" />
          </figure>
          <div v-if="goodImages.length > 12" class="more">
            +{{ goodImages.length - 12 }}
          </div>
        </div>
        <p v-else-if="targetName" class="note">
          No unlabelled images in that project yet. Add this line's good gloves
          above.
        </p>
      </section>
    </div>

    <!-- ── 3. run ───────────────────────────────────────────────────── -->
    <section class="card run-card">
      <div class="card-head">
        <span class="num">3</span>
        <h2>Add the defects</h2>
        <label class="inline-field">
          <span>per image</span>
          <input v-model.number="perImage" type="number" min="1" max="4"
                 class="control tiny-control" />
        </label>
      </div>

      <div class="run-row">
        <button v-if="!running" class="btn btn-primary" :disabled="!canRun"
                @click="run">
          <Icon name="zap" size="sm" />
          <span>Add defects to {{ goodImages.length }} image(s)</span>
        </button>
        <button v-else class="btn btn-danger" @click="stop">
          <Icon name="x" size="sm" />
          <span>Stop</span>
        </button>
<!--
          Both offered, neither required. The pictures are a dataset in their
          own right: they can be trained here, or downloaded and taken to
          whatever the line actually runs.
        -->
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
        <span v-if="!canRun && !running" class="note tiny">
          {{ whyNotReady }}
        </span>
      </div>

      <div v-if="job" class="job">
        <div class="job-line">{{ job.message || job.status }}</div>
        <div v-if="running" class="job-bar">
          <div class="job-fill" :style="{ width: percent + '%' }"></div>
        </div>
        <dl v-if="job.profile" class="measured">
          <div><dt>Glove</dt><dd>{{ Math.round(job.profile.glove_level) }}</dd></div>
          <div><dt>Backlight</dt><dd>{{ Math.round(job.profile.bg_level) }}</dd></div>
          <div><dt>Noise</dt><dd>{{ job.profile.noise_sigma }}</dd></div>
          <div class="good"><dt>Made</dt><dd>{{ job.made || 0 }}</dd></div>
          <div :class="{ warn: job.refused }">
            <dt>Refused</dt><dd>{{ job.refused || 0 }}</dd>
          </div>
        </dl>
        <p v-if="job.refused" class="note tiny">
          Refused means too faint to see against this colour's own noise.
          Thrown out rather than written in: a defect nobody can see teaches
          the detector that ordinary rubber is a tear.
        </p>
      </div>
    </section>

    <!-- ── results ──────────────────────────────────────────────────── -->
    <section v-if="results.length" class="card">
      <div class="card-head">
        <span class="num"><Icon name="check" size="sm" /></span>
        <h2>Made, and boxed</h2>
        <span class="pill">{{ results.length }}</span>
      </div>
      <p class="note">
        The boxes come from what changed by more than this line's own noise, so
        the same defect is boxed smaller on a darker glove. Nothing to draw by
        hand.
      </p>

      <div class="gallery">
        <figure v-for="img in results" :key="img.filename" class="shot"
                @click="preview = img">
          <div class="shot-frame">
            <img :src="imageUrl(targetName, img.filename)" :alt="img.filename" />
            <svg v-if="img.width && img.height" class="overlay"
                 :viewBox="`0 0 ${img.width} ${img.height}`"
                 preserveAspectRatio="none">
              <rect v-for="(box, i) in img.boxes || []" :key="i"
                    :x="box[0]" :y="box[1]" :width="box[2]" :height="box[3]"
                    class="box" />
            </svg>
          </div>
          <figcaption>
            <span v-for="tag in tagsOf(img)" :key="tag" class="tag">{{ tag }}</span>
          </figcaption>
        </figure>
      </div>
    </section>

    <!-- a closer look -->
    <div v-if="preview" class="lightbox" @click="preview = null">
      <div class="lightbox-inner" @click.stop>
        <div class="shot-frame big">
          <img :src="imageUrl(targetName, preview.filename)" :alt="preview.filename" />
          <svg v-if="preview.width" class="overlay"
               :viewBox="`0 0 ${preview.width} ${preview.height}`"
               preserveAspectRatio="none">
            <rect v-for="(box, i) in preview.boxes || []" :key="i"
                  :x="box[0]" :y="box[1]" :width="box[2]" :height="box[3]"
                  class="box" />
          </svg>
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
import { errorMessage, projectService } from '@/services'

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
const dragging = ref(false)
const preview = ref(null)
const fileInput = ref(null)
const makingNew = ref(false)
const newName = ref('')
const creating = ref(false)
const downloading = ref(false)

const running = computed(() => job.value?.status === 'running')
const canRun = computed(
  () => tags.value.length > 0 && targetName.value && goodImages.value.length > 0)
const percent = computed(() => {
  if (!job.value?.total) return 0
  return Math.round(((job.value.done || 0) / job.value.total) * 100)
})

const whyNotReady = computed(() => {
  if (!tags.value.length) return 'Collect some defects and tick the ones to use.'
  if (!targetName.value) return 'Choose the project for this line.'
  if (!goodImages.value.length) return 'Add this line\'s good gloves.'
  return ''
})

const imageUrl = (project, filename) =>
  `/api/projects/${encodeURIComponent(project)}/images/${encodeURIComponent(filename)}/raw`

// The index already carries every image's boxes, so the gallery draws them
// from the one request it has rather than asking per picture.
const tagsOf = (img) => [...new Set((img.boxes || []).map((b) => b[4]))]

const toggleTag = (tag) => {
  const at = tags.value.indexOf(tag)
  if (at === -1) tags.value.push(tag)
  else tags.value.splice(at, 1)
}

const loadLibrary = async () => {
  library.value = null
  tags.value = []
  if (!sourceName.value) return
  try {
    library.value = await projectService.defectLibrary(sourceName.value)
    tags.value = (library.value.per_tag || []).map((row) => row.tag)
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
    // Everything ticked to start with: unticking a few beats hunting for them.
    tags.value = (library.value.per_tag || []).map((row) => row.tag)
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
    projects.value = (await projectService.list()).projects || []
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

const refreshImages = async () => {
  if (!targetName.value) return
  const { images } = await projectService.images(targetName.value)
  const all = images || []
  goodImages.value = all.filter((i) => !i.annotated && !i.augmented)
  // What this run produced: the batch it stamped, still boxed.
  const batch = job.value?.batch
  results.value = batch
    ? all.filter((i) => i.batch === batch && i.annotated)
    : []
}

const upload = async (files) => {
  const picked = Array.from(files || []).filter((f) => f.type.startsWith('image/'))
  if (!picked.length || !targetName.value) return
  uploading.value = true
  uploadPercent.value = 0
  error.value = ''
  try {
    await projectService.uploadImages(targetName.value, picked,
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
        per_image: perImage.value })
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
    projects.value = (await projectService.list()).projects || []
  } catch (err) {
    error.value = errorMessage(err, 'The projects could not be listed')
  }
})

onBeforeUnmount(() => clearTimeout(timer))
</script>

<style scoped>
.studio { padding: 1.5rem; max-width: 76rem; }

/* ── hero ─────────────────────────────────────────────────────────── */
.hero {
  display: flex;
  justify-content: space-between;
  gap: 1.5rem;
  align-items: flex-start;
  flex-wrap: wrap;
  padding: 1.25rem 1.4rem;
  margin-bottom: 1.2rem;
  border-radius: 16px;
  border: 1px solid var(--border-color, var(--border));
  background:
    radial-gradient(120% 140% at 0% 0%,
      var(--accent-softer, rgba(46, 167, 122, 0.16)), transparent 60%),
    var(--bg-subtle);
}
.hero-text h1 {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin: 0 0 0.5rem;
  font-size: 1.35rem;
  color: var(--text-primary, var(--text));
}
.hero-text p {
  margin: 0;
  max-width: 46rem;
  font-size: 0.82rem;
  line-height: 1.65;
  color: var(--text-tertiary, var(--text-3));
}
.hero-flow {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  color: var(--text-tertiary, var(--text-3));
}
.flow-arrow { opacity: 0.5; font-size: 0.85rem; }
.flow-step {
  padding: 0.3rem 0.65rem;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 600;
  border: 1px solid var(--border-color, var(--border));
}
.flow-step.done {
  color: var(--accent, #2ea77a);
  border-color: var(--accent, #2ea77a);
  background: var(--accent-softer, rgba(46, 167, 122, 0.12));
}

.banner {
  margin-bottom: 1rem;
  padding: 0.65rem 0.85rem;
  border-radius: 10px;
  font-size: 0.82rem;
}
.banner.error {
  color: var(--danger, #d9534f);
  border: 1px solid var(--danger, #d9534f);
}

/* ── cards ────────────────────────────────────────────────────────── */
.columns {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(23rem, 1fr));
  gap: 1rem;
}
.card {
  padding: 1.1rem;
  margin-bottom: 1rem;
  border: 1px solid var(--border-color, var(--border));
  border-radius: 16px;
  background: var(--bg-subtle);
}
.card-head {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.9rem;
}
.card-head h2 {
  margin: 0;
  font-size: 1rem;
  color: var(--text-primary, var(--text));
}
.num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.65rem;
  height: 1.65rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 700;
  background: var(--accent-softer, rgba(46, 167, 122, 0.15));
  color: var(--accent, #2ea77a);
}
.pill {
  margin-left: auto;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 600;
  background: var(--accent-softer, rgba(46, 167, 122, 0.12));
  color: var(--accent, #2ea77a);
}

.field { display: flex; flex-direction: column; gap: 0.3rem; margin-bottom: 0.7rem; }
.field > span, .inline-field > span {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-tertiary, var(--text-3));
}
.inline-field {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.control {
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border-color, var(--border));
  border-radius: 10px;
  padding: 0.5rem 0.7rem;
  font-family: inherit;
  font-size: 0.85rem;
  color: var(--text-primary, var(--text));
}
.tiny-control { width: 4rem; padding: 0.35rem 0.5rem; }
.wide { width: 100%; justify-content: center; }

.note {
  margin: 0.6rem 0 0;
  font-size: 0.78rem;
  line-height: 1.55;
  color: var(--text-tertiary, var(--text-3));
}
.note.tiny { font-size: 0.72rem; }

/* ── defect chips ─────────────────────────────────────────────────── */
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.8rem; }
.chip {
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
  padding: 0.35rem 0.6rem;
  border-radius: 999px;
  border: 1px solid var(--border-color, var(--border));
  background: var(--bg);
  cursor: pointer;
  font-family: inherit;
  font-size: 0.76rem;
  color: var(--text-tertiary, var(--text-3));
  transition: border-color 0.15s ease, color 0.15s ease;
}
.chip-name { font-weight: 600; }
.chip-count { opacity: 0.75; }
.chip-kind { font-size: 0.66rem; opacity: 0.6; }
.chip.on {
  color: var(--text-primary, var(--text));
  border-color: var(--accent, #2ea77a);
  background: var(--accent-softer, rgba(46, 167, 122, 0.12));
}
/* The physics, at a glance: light through, or more/less rubber. */
.chip.on.hole, .chip.on.tear { border-color: #e0a63c; }
.chip.on.thin, .chip.on.thick { border-color: #6aa8e0; }

/* ── drop zone ────────────────────────────────────────────────────── */
.drop {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.3rem;
  padding: 1.2rem 1rem;
  border: 1.5px dashed var(--border-color, var(--border));
  border-radius: 14px;
  text-align: center;
  color: var(--text-tertiary, var(--text-3));
  transition: border-color 0.15s ease, background 0.15s ease;
}
.drop.over {
  border-color: var(--accent, #2ea77a);
  background: var(--accent-softer, rgba(46, 167, 122, 0.08));
}
.drop.busy { opacity: 0.7; }
.drop p { margin: 0.2rem 0 0; font-size: 0.82rem; }
.hidden-input { display: none; }
.link {
  background: none;
  border: none;
  padding: 0;
  font: inherit;
  cursor: pointer;
  text-decoration: underline;
  color: var(--accent, #2ea77a);
}

/* ── thumbnails ───────────────────────────────────────────────────── */
.new-project { display: flex; gap: 0.5rem; margin-bottom: 0.7rem; }
.new-project .control { flex: 1 1 14rem; }

.strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.9rem;
  align-items: center;
}
.strip figure { margin: 0; }
.strip img {
  width: 4.2rem;
  height: 4.2rem;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid var(--border-color, var(--border));
  background: #000;
}
.more {
  font-size: 0.78rem;
  color: var(--text-tertiary, var(--text-3));
}

/* ── run ──────────────────────────────────────────────────────────── */
.run-card { margin-top: 0; }
.run-row { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
.job { margin-top: 0.9rem; }
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
  background: var(--accent, #2ea77a);
  transition: width 0.3s ease;
}
.measured { display: flex; flex-wrap: wrap; gap: 1.4rem; margin: 0.9rem 0 0; }
.measured div { display: flex; flex-direction: column; gap: 0.1rem; }
.measured dt {
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-tertiary, var(--text-3));
}
.measured dd {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text-primary, var(--text));
}
.measured .good dd { color: var(--accent, #2ea77a); }
.measured .warn dd { color: #e0a63c; }

/* ── results ──────────────────────────────────────────────────────── */
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(11rem, 1fr));
  gap: 0.7rem;
  margin-top: 0.9rem;
}
.shot { margin: 0; cursor: zoom-in; }
.shot-frame {
  position: relative;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--border-color, var(--border));
  background: #000;
}
.shot-frame img { display: block; width: 100%; }
.overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}
.box {
  fill: none;
  stroke: #3ddc9a;
  stroke-width: 6;
  vector-effect: non-scaling-stroke;
}
.shot figcaption {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
  margin-top: 0.35rem;
}
.tag {
  padding: 0.12rem 0.45rem;
  border-radius: 999px;
  font-size: 0.68rem;
  font-weight: 600;
  background: var(--accent-softer, rgba(46, 167, 122, 0.14));
  color: var(--accent, #2ea77a);
}

/* ── lightbox ─────────────────────────────────────────────────────── */
.lightbox {
  position: fixed;
  inset: 0;
  z-index: 60;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  background: rgba(0, 0, 0, 0.82);
}
.lightbox-inner { max-width: min(90vw, 60rem); width: 100%; }
.shot-frame.big img { max-height: 76vh; object-fit: contain; margin: 0 auto; }
.lightbox-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.7rem;
}
.lightbox-bar .btn { margin-left: auto; }
</style>

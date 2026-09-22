<template>
  <div class="studio">
    <header class="studio-head">
      <h1 class="studio-title">
        <Icon name="layers" size="md" />
        Defect Studio
      </h1>
      <p class="studio-sub">
        A mono camera turns glove colour into one grey level, so a model trained
        on one colour does not work on another and every line needs its own
        defects — the rare half of any dataset. This takes defects already
        labelled on one colour and puts them on this line's good gloves, at the
        right depth for its grey, with the boxes drawn from what is visible
        afterwards.
      </p>
    </header>

    <div v-if="error" class="studio-error">{{ error }}</div>

    <!-- 1. where the defects come from -->
    <section class="studio-step">
      <div class="step-head">
        <span class="step-num">1</span>
        <h2>Take defects from</h2>
      </div>
      <p class="step-note">
        A project that is already labelled — usually the colour you have been
        running longest.
      </p>
      <div class="step-row">
        <select v-model="sourceName" class="studio-select" @change="loadLibrary">
          <option :value="''">Choose a project…</option>
          <option v-for="p in projects" :key="p.name" :value="p.name">
            {{ p.name }} — {{ p.total_annotations || 0 }} annotated
          </option>
        </select>
        <button class="btn btn-secondary" :disabled="!sourceName || collecting"
                @click="collect">
          <Icon name="download" size="sm" />
          <span>{{ collecting ? 'Reading…' : 'Collect defects' }}</span>
        </button>
      </div>

      <div v-if="library" class="step-result">
        <p v-if="!library.total" class="step-note">
          Nothing collected from that project yet. <strong>Collect defects</strong>
          reads the optical signature of every box in it — the density it added
          to the rubber, which is what survives being moved to another colour.
        </p>
        <template v-else>
          <p class="step-note">
            {{ library.total }} defect(s) collected. Tick the ones to use:
          </p>
          <div class="tag-grid">
            <label v-for="row in library.per_tag" :key="row.tag" class="tag-item">
              <input type="checkbox" :value="row.tag" v-model="tags" />
              <span class="tag-name">{{ row.tag }}</span>
              <span class="tag-meta">{{ row.count }} · {{ row.defect_class }}</span>
            </label>
          </div>
        </template>
      </div>
    </section>

    <!-- 2. the line being set up -->
    <section class="studio-step">
      <div class="step-head">
        <span class="step-num">2</span>
        <h2>Put them on</h2>
      </div>
      <p class="step-note">
        The project holding this line's good gloves. Only images with no boxes
        on them are used — anything already labelled is left alone.
      </p>
      <div class="step-row">
        <select v-model="targetName" class="studio-select" @change="loadTarget">
          <option :value="''">Choose a project…</option>
          <option v-for="p in projects" :key="p.name" :value="p.name">
            {{ p.name }} — {{ p.total_images || 0 }} images
          </option>
        </select>
        <label class="studio-field">
          <span>Defects per image</span>
          <input v-model.number="perImage" type="number" min="1" max="4"
                 class="studio-number" />
        </label>
      </div>
      <p v-if="targetName" class="step-note">
        <strong>{{ goodCount }}</strong> good image(s) with no boxes yet.
        <span v-if="!goodCount">
          Import this line's good gloves into that project first.
        </span>
      </p>
    </section>

    <!-- 3. run -->
    <section class="studio-step">
      <div class="step-head">
        <span class="step-num">3</span>
        <h2>Add the defects</h2>
      </div>
      <div class="step-row">
        <button v-if="!running" class="btn btn-primary"
                :disabled="!canRun" @click="run">
          <Icon name="zap" size="sm" />
          <span>Add defects to {{ goodCount }} image(s)</span>
        </button>
        <button v-else class="btn btn-danger" @click="stop">
          <Icon name="x" size="sm" />
          <span>Stop</span>
        </button>
        <router-link v-if="job && job.status === 'finished' && targetName"
                     :to="`/projects/${encodeURIComponent(targetName)}`"
                     class="btn btn-secondary">
          <Icon name="folder" size="sm" />
          <span>Open the project</span>
        </router-link>
      </div>

      <div v-if="job" class="job">
        <div class="job-line">{{ job.message || job.status }}</div>
        <div v-if="running" class="job-bar">
          <div class="job-fill" :style="{ width: percent + '%' }"></div>
        </div>
        <dl v-if="job.profile" class="job-measured">
          <div><dt>Glove</dt><dd>{{ Math.round(job.profile.glove_level) }}</dd></div>
          <div><dt>Backlight</dt><dd>{{ Math.round(job.profile.bg_level) }}</dd></div>
          <div><dt>Noise</dt><dd>{{ job.profile.noise_sigma }}</dd></div>
          <div><dt>Made</dt><dd>{{ job.made || 0 }}</dd></div>
          <div><dt>Refused</dt><dd>{{ job.refused || 0 }}</dd></div>
        </dl>
        <p v-if="job.refused" class="step-note">
          Refused means the defect was too faint to see against this colour's
          own noise. It is thrown out rather than written in: a defect nobody
          can see teaches the detector that ordinary rubber is a tear.
        </p>
      </div>
    </section>
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
const goodCount = ref(0)
const job = ref(null)
const error = ref('')
const collecting = ref(false)

const running = computed(() => job.value?.status === 'running')
const canRun = computed(
  () => tags.value.length > 0 && targetName.value && goodCount.value > 0)
const percent = computed(() => {
  if (!job.value?.total) return 0
  return Math.round(((job.value.done || 0) / job.value.total) * 100)
})

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
    // Everything ticked by default: unticking a few is quicker than hunting
    // for the ones worth having.
    tags.value = (library.value.per_tag || []).map((row) => row.tag)
  } catch (err) {
    error.value = errorMessage(err, 'Those defects could not be collected')
  } finally {
    collecting.value = false
  }
}

const loadTarget = async () => {
  goodCount.value = 0
  job.value = null
  if (!targetName.value) return
  try {
    const { images } = await projectService.images(targetName.value)
    goodCount.value = (images || []).filter(
      (i) => !i.annotated && !i.augmented).length
    const status = await projectService.defectSynthStatus(targetName.value)
    job.value = status.job || null
    if (job.value?.status === 'running') poll()
  } catch (err) {
    error.value = errorMessage(err, 'That project could not be read')
  }
}

const run = async () => {
  error.value = ''
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
      else await loadTargetCount()
    } catch {
      /* the next poll picks it up */
    }
  }, 1200)
}

const loadTargetCount = async () => {
  if (!targetName.value) return
  try {
    const { images } = await projectService.images(targetName.value)
    goodCount.value = (images || []).filter(
      (i) => !i.annotated && !i.augmented).length
  } catch { /* leave the last count */ }
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
.studio { padding: 1.5rem; max-width: 60rem; }

.studio-head { margin-bottom: 1.5rem; }
.studio-title {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin: 0 0 0.5rem;
  font-size: 1.4rem;
  color: var(--text-primary, var(--text));
}
.studio-sub {
  margin: 0;
  max-width: 48rem;
  font-size: 0.84rem;
  line-height: 1.6;
  color: var(--text-tertiary, var(--text-3));
}

.studio-error {
  margin-bottom: 1rem;
  padding: 0.6rem 0.8rem;
  border-radius: 10px;
  font-size: 0.82rem;
  color: var(--danger, #d9534f);
  border: 1px solid var(--danger, #d9534f);
}

.studio-step {
  margin-bottom: 1.25rem;
  padding: 1rem;
  border: 1px solid var(--border-color, var(--border));
  border-radius: 14px;
  background: var(--bg-subtle);
}
.step-head {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.4rem;
}
.step-head h2 {
  margin: 0;
  font-size: 1rem;
  color: var(--text-primary, var(--text));
}
.step-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.5rem;
  height: 1.5rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 700;
  background: var(--accent-softer, rgba(46, 167, 122, 0.15));
  color: var(--accent, #2ea77a);
}
.step-note {
  margin: 0 0 0.7rem;
  font-size: 0.78rem;
  line-height: 1.55;
  color: var(--text-tertiary, var(--text-3));
}
.step-row {
  display: flex;
  gap: 0.6rem;
  align-items: flex-end;
  flex-wrap: wrap;
}
.step-result { margin-top: 0.8rem; }

.studio-select, .studio-number {
  background: var(--bg);
  border: 1px solid var(--border-color, var(--border));
  border-radius: 10px;
  padding: 0.5rem 0.7rem;
  font-family: inherit;
  font-size: 0.85rem;
  color: var(--text-primary, var(--text));
}
.studio-select { flex: 1 1 20rem; min-width: 14rem; }
.studio-number { width: 5rem; }
.studio-field { display: flex; flex-direction: column; gap: 0.3rem; }
.studio-field > span {
  font-size: 0.72rem;
  color: var(--text-tertiary, var(--text-3));
}

.tag-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(14rem, 1fr));
  gap: 0.35rem;
}
.tag-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.3rem 0.4rem;
  border-radius: 8px;
  font-size: 0.8rem;
  color: var(--text-primary, var(--text));
}
.tag-name { font-weight: 600; }
.tag-meta {
  margin-left: auto;
  font-size: 0.72rem;
  color: var(--text-tertiary, var(--text-3));
}

.job { margin-top: 0.9rem; }
.job-line {
  font-size: 0.82rem;
  color: var(--text-primary, var(--text));
}
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
.job-measured {
  display: flex;
  flex-wrap: wrap;
  gap: 1.2rem;
  margin: 0.8rem 0 0;
}
.job-measured div { display: flex; flex-direction: column; gap: 0.15rem; }
.job-measured dt {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-tertiary, var(--text-3));
}
.job-measured dd {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-primary, var(--text));
}
</style>

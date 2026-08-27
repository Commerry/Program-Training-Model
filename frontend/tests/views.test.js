/**
 * Renders every view with stubbed services.
 *
 * The build only checks that templates parse. This mounts them, which is what
 * actually catches undefined bindings, bad property access on API payloads and
 * exceptions thrown from onMounted.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { createRouter, createWebHistory } from 'vue-router'

// ── API fixtures: shaped exactly like the backend responses ─────────────────
const PROJECT = {
  name: 'demo',
  description: 'a demo project',
  path: '/data/projects/demo',
  created_at: '2026-01-02T10:00:00',
  updated_at: '2026-01-05T10:00:00',
  total_images: 40,
  annotated_images: 30,
  total_annotations: 90,
  tags: { cat: { boxes: 50, images: 20 }, dog: { boxes: 40, images: 18 } }
}

const IMAGE = {
  filename: '20260101_000000_000000_0000.jpg',
  annotated: true,
  regions_count: 2,
  tags: ['cat'],
  width: 640,
  height: 480,
  augmented: false,
  size_kb: 120.5
}

const RUN = {
  project_name: 'demo',
  model_name: 'demo_yolo11s_20260105',
  model_type: 'yolo11s',
  status: 'completed',
  epochs: 100,
  completed_epochs: 100,
  batch_size: 16,
  img_size: 640,
  classes: ['cat', 'dog'],
  train_images: 32,
  val_images: 8,
  metrics: { mAP50: 0.87, mAP50_95: 0.62, precision: 0.9, recall: 0.85, train_loss: 0.4 },
  best_model: '/data/projects/demo/training/runs/x/weights/best.pt',
  exported_models: { pt: '/x/best.pt', onnx: '/x/best.onnx' },
  started_at: '2026-01-05T09:00:00',
  completed_at: '2026-01-05T11:15:00'
}

const STATUS = {
  ...RUN,
  status: 'completed',
  current_epoch: 100,
  total_epochs: 100,
  weights: 'yolo11s.pt',
  dataset_path: '/x/dataset',
  metrics_history: [{ epoch: 1, mAP50: 0.2 }, { epoch: 2, mAP50: 0.6 }],
  error: null,
  pid: null
}

const OVERVIEW = {
  project_count: 2,
  total_images: 80,
  annotated_images: 60,
  total_annotations: 180,
  training_runs: 3,
  completed_runs: 2,
  failed_runs: 1,
  active_runs: [{ project_name: 'demo', model_name: 'run', current_epoch: 5, total_epochs: 100 }],
  average_map50: 0.81,
  recent_runs: [RUN]
}

const SUMMARY = {
  success: true,
  total_images: 40,
  annotated_images: 30,
  total_boxes: 90,
  tags: PROJECT.tags,
  classes: ['cat', 'dog'],
  num_classes: 2,
  readiness_score: 72.5,
  warnings: ['Some tags have few images.'],
  recommendations: ['Dataset looks ready for training.']
}

vi.mock('@/services', () => ({
  http: { get: vi.fn(() => Promise.resolve({ data: { version: '3.0.0', projects_root: '/d', auth_required: true } })) },
  errorMessage: (e, f) => e?.message || f || 'error',
  setUnauthorizedHandler: vi.fn(),
  authService: {
    me: vi.fn(() => Promise.resolve({ success: true, user: null })),
    login: vi.fn(() => Promise.resolve({ success: true, user: { username: 'admin' } })),
    register: vi.fn(() => Promise.resolve({ success: true, user: { username: 'bob' } })),
    logout: vi.fn(() => Promise.resolve({ success: true })),
    updateProfile: vi.fn(() => Promise.resolve({ success: true, user: { username: 'admin' } }))
  },
  projectService: {
    list: vi.fn(() => Promise.resolve({ success: true, projects: [PROJECT] })),
    get: vi.fn(() => Promise.resolve({ success: true, project: PROJECT })),
    create: vi.fn(() => Promise.resolve({ success: true, project: PROJECT })),
    remove: vi.fn(() => Promise.resolve({ success: true })),
    tags: vi.fn(() => Promise.resolve({ success: true, tags: ['cat', 'dog'], tags_detail: PROJECT.tags })),
    datasetSummary: vi.fn(() => Promise.resolve(SUMMARY)),
    images: vi.fn(() => Promise.resolve({ success: true, images: [IMAGE] })),
    uploadImages: vi.fn(() => Promise.resolve({ success: true, imported_count: 1, imported: [IMAGE], rejected: [] })),
    imageData: vi.fn(() => Promise.resolve({
      success: true,
      data: {
        filename: IMAGE.filename,
        image: 'data:image/jpeg;base64,AAAA',
        width: 640,
        height: 480,
        annotations: { filename: IMAGE.filename, regions: [{ tag: 'cat', x: 10, y: 10, width: 50, height: 50 }], annotated: true }
      }
    })),
    deleteImage: vi.fn(() => Promise.resolve({ success: true })),
    imageUrl: (p, f) => `/api/projects/${p}/images/${f}/raw`,
    saveAnnotations: vi.fn(() => Promise.resolve({ success: true, saved_count: 1 })),
    tones: vi.fn(() => Promise.resolve({ success: true, tones: ['gray'] })),
    augment: vi.fn(() => Promise.resolve({ success: true, message: 'ok', created_count: 5 })),
    prepareDataset: vi.fn(() => Promise.resolve({ success: true, dataset: {} })),
    exportDataset: vi.fn(() => Promise.resolve(new Blob(['x']))),
    importDataset: vi.fn(() => Promise.resolve({ success: true, message: 'imported' }))
  },
  trainingService: {
    options: vi.fn(() => Promise.resolve({ success: true, model_types: ['yolo11s'] })),
    start: vi.fn(() => Promise.resolve({ success: true, config: STATUS, dataset: { train_images: 32, val_images: 8, train_boxes: 70, val_boxes: 20, empty_classes: [], skipped_count: 0 } })),
    status: vi.fn(() => Promise.resolve({ success: true, status: STATUS, has_run: true })),
    stop: vi.fn(() => Promise.resolve({ success: true })),
    reset: vi.fn(() => Promise.resolve({ success: true })),
    logs: vi.fn(() => Promise.resolve({ success: true, logs: ['[10:00:00] Epoch 1/100'] })),
    models: vi.fn(() => Promise.resolve({ success: true, models: [{ name: 'best.pt', run: 'x', path: '/x/best.pt', format: 'pt', size_mb: 5.2, modified: '2026-01-05T11:00:00' }] })),
    downloadUrl: (p, path) => `/api/projects/${p}/training/models/download?path=${path}`,
    history: vi.fn(() => Promise.resolve({ success: true, history: [RUN] })),
    allHistory: vi.fn(() => Promise.resolve({ success: true, history: [RUN, { ...RUN, status: 'failed', error: 'CUDA out of memory', metrics: {} }] })),
    overview: vi.fn(() => Promise.resolve(OVERVIEW)),
    testModel: vi.fn(() => Promise.resolve({ success: true, results: [] }))
  }
}))

// fabric touches canvas APIs happy-dom does not implement
vi.mock('fabric', () => ({
  fabric: {
    Canvas: class {
      constructor() { this.freeDrawingBrush = {} }
      on() {} off() {} clear() {} dispose() {} renderAll() {}
      setWidth() {} setHeight() {} setBackgroundImage() {}
      getObjects() { return [] }
      setActiveObject() {} remove() {} add() {} discardActiveObject() {}
    },
    Image: { fromURL: (url, cb) => cb({ width: 640, height: 480 }) },
    Rect: class { constructor(o) { Object.assign(this, o) } set() {} },
    Text: class { constructor(o) { Object.assign(this, o) } set() {} }
  }
}))

const VIEWS = {
  DashboardView: () => import('@/views/DashboardView.vue'),
  ProjectsView: () => import('@/views/ProjectsView.vue'),
  ProjectDetailView: () => import('@/views/ProjectDetailView.vue'),
  AnnotateView: () => import('@/views/AnnotateView.vue'),
  TrainView: () => import('@/views/TrainView.vue'),
  DatasetsView: () => import('@/views/DatasetsView.vue'),
  ModelsView: () => import('@/views/ModelsView.vue'),
  HistoryView: () => import('@/views/HistoryView.vue'),
  AnalyticsView: () => import('@/views/AnalyticsView.vue'),
  TestModelView: () => import('@/views/TestModelView.vue'),
  SettingsView: () => import('@/views/SettingsView.vue'),
  LoginView: () => import('@/views/LoginView.vue'),
  RegisterView: () => import('@/views/RegisterView.vue'),
  NotFoundView: () => import('@/views/NotFoundView.vue')
}

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'Dashboard', component: { template: '<div/>' } },
    { path: '/projects', name: 'Projects', component: { template: '<div/>' } },
    { path: '/projects/:name', name: 'ProjectDetail', component: { template: '<div/>' } },
    { path: '/projects/:name/annotate/:filename', name: 'Annotate', component: { template: '<div/>' } },
    { path: '/projects/:name/train', name: 'Train', component: { template: '<div/>' } },
    { path: '/datasets', name: 'Datasets', component: { template: '<div/>' } },
    { path: '/models', name: 'Models', component: { template: '<div/>' } },
    { path: '/history', name: 'History', component: { template: '<div/>' } },
    { path: '/analytics', name: 'Analytics', component: { template: '<div/>' } },
    { path: '/test-model', name: 'TestModel', component: { template: '<div/>' } },
    { path: '/settings', name: 'Settings', component: { template: '<div/>' } },
    { path: '/login', name: 'Login', component: { template: '<div/>' } },
    { path: '/register', name: 'Register', component: { template: '<div/>' } }
  ]
})

const flush = async () => {
  for (let i = 0; i < 8; i += 1) await new Promise((r) => setTimeout(r, 0))
}

describe('every view mounts without runtime errors', () => {
  beforeEach(async () => {
    setActivePinia(createPinia())
    await router.push({ name: 'ProjectDetail', params: { name: 'demo', filename: IMAGE.filename } })
      .catch(() => router.push('/'))
    await router.isReady()
  })

  for (const [name, loader] of Object.entries(VIEWS)) {
    it(name, async () => {
      const errors = []
      const spy = vi.spyOn(console, 'error').mockImplementation((...a) => errors.push(a.join(' ')))
      const warns = []
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation((...a) => warns.push(a.join(' ')))

      const component = (await loader()).default
      const wrapper = mount(component, {
        global: {
          plugins: [router],
          stubs: { Icon: { template: '<i/>', props: ['name', 'size'] } }
        },
        attachTo: document.body
      })

      await flush()
      const html = wrapper.html()
      wrapper.unmount()
      spy.mockRestore()
      warnSpy.mockRestore()

      const real = [...errors, ...warns].filter(
        (m) => !m.includes('Failed to resolve component') &&
               !m.includes('Extraneous non-props')
      )
      expect(real, `${name} logged: ${real.join(' | ')}`).toEqual([])
      expect(html.length, `${name} rendered nothing`).toBeGreaterThan(50)
    })
  }
})

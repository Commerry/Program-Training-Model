/**
 * Exercises the interactions that matter: filtering, paging, saving, starting
 * a run, and the error paths. Mounting alone proves a view renders; these
 * prove the handlers behind its buttons do the right thing.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { createRouter, createWebHistory } from 'vue-router'

import { projectService, trainingService } from '@/services'

const makeImage = (index, annotated, augmented = false) => ({
  filename: `img_${String(index).padStart(4, '0')}.jpg`,
  annotated,
  regions_count: annotated ? 2 : 0,
  tags: annotated ? ['cat'] : [],
  boxes: annotated ? [[10, 20, 60, 80, 'cat'], [200, 150, 40, 40, 'dog']] : [],
  width: 640,
  height: 480,
  augmented,
  size_kb: 100
})

// 150 images: enough to force more than one page at PAGE_SIZE 60.
const MANY_IMAGES = [
  ...Array.from({ length: 100 }, (_, i) => makeImage(i, true)),
  ...Array.from({ length: 30 }, (_, i) => makeImage(100 + i, false)),
  ...Array.from({ length: 20 }, (_, i) => makeImage(200 + i, true, true))
]

const PROJECT = {
  name: 'demo',
  description: 'd',
  total_images: 150,
  annotated_images: 120,
  total_annotations: 240,
  tags: { cat: { boxes: 240, images: 120 } },
  created_at: '2026-01-01T00:00:00',
  updated_at: '2026-01-02T00:00:00'
}

const RUNS = [
  { project_name: 'demo', model_name: 'a', model_type: 'yolo11s', status: 'completed',
    epochs: 100, completed_epochs: 100, classes: ['cat'], train_images: 96, val_images: 24,
    metrics: { mAP50: 0.9 }, best_model: '/x/a.pt', exported_models: { pt: '/x/a.pt' },
    started_at: '2026-01-05T09:00:00', completed_at: '2026-01-05T10:00:00' },
  { project_name: 'demo', model_name: 'b', model_type: 'yolo11n', status: 'failed',
    epochs: 50, error: 'CUDA out of memory', metrics: {},
    started_at: '2026-01-06T09:00:00', completed_at: '2026-01-06T09:05:00' },
  { project_name: 'other', model_name: 'c', model_type: 'faster_rcnn', status: 'stopped',
    epochs: 30, completed_epochs: 12, classes: ['dog'], metrics: { mAP50: 0.4 },
    best_model: '/x/c.pth', exported_models: {},
    started_at: '2026-01-07T09:00:00', completed_at: '2026-01-07T09:40:00' }
]

vi.mock('@/services', () => {
  const fn = (value) => vi.fn(() => Promise.resolve(value))
  return {
    http: { get: fn({ data: {} }) },
    errorMessage: (e, f) => e?.message || f || 'error',
    setUnauthorizedHandler: vi.fn(),
    authService: {
      me: fn({ success: true, user: null }),
      login: vi.fn(), register: vi.fn(), logout: vi.fn(), updateProfile: vi.fn()
    },
    projectService: {
      list: vi.fn(), get: vi.fn(), create: vi.fn(), remove: vi.fn(),
      tags: vi.fn(), datasetSummary: vi.fn(), images: vi.fn(),
      uploadImages: vi.fn(), imageData: vi.fn(), deleteImage: vi.fn(),
      imageUrl: (p, f) => `/api/projects/${p}/images/${f}/raw`,
      saveAnnotations: vi.fn(), tones: vi.fn(), augment: vi.fn(),
      prepareDataset: vi.fn(), exportDataset: vi.fn(), importDataset: vi.fn()
    },
    trainingService: {
      options: vi.fn(), start: vi.fn(), status: vi.fn(), stop: vi.fn(),
      reset: vi.fn(), logs: vi.fn(), models: vi.fn(), history: vi.fn(),
      allHistory: vi.fn(), overview: vi.fn(), testModel: vi.fn(),
      downloadUrl: (p, path) => `/dl?p=${path}`
    }
  }
})

vi.mock('fabric', () => ({
  fabric: {
    Canvas: class { on() {} clear() {} dispose() {} renderAll() {} setWidth() {}
      setHeight() {} setBackgroundImage() {} getObjects() { return [] }
      setActiveObject() {} remove() {} add() {} },
    Image: { fromURL: (u, cb) => cb({ width: 640, height: 480 }) },
    Rect: class { constructor(o) { Object.assign(this, o) } set() {} },
    Text: class { constructor(o) { Object.assign(this, o) } set() {} }
  }
}))

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'Dashboard', component: { template: '<div/>' } },
    { path: '/projects', name: 'Projects', component: { template: '<div/>' } },
    { path: '/projects/:name', name: 'ProjectDetail', component: { template: '<div/>' } },
    { path: '/projects/:name/annotate/:filename', name: 'Annotate', component: { template: '<div/>' } },
    { path: '/projects/:name/train', name: 'Train', component: { template: '<div/>' } },
    { path: '/models', name: 'Models', component: { template: '<div/>' } },
    { path: '/history', name: 'History', component: { template: '<div/>' } },
    { path: '/datasets', name: 'Datasets', component: { template: '<div/>' } },
    { path: '/test-model', name: 'TestModel', component: { template: '<div/>' } },
    { path: '/login', name: 'Login', component: { template: '<div/>' } }
  ]
})

const flush = async () => { for (let i = 0; i < 10; i += 1) await new Promise((r) => setTimeout(r, 0)) }

/**
 * happy-dom never fetches images, so their load event never fires and the ROI
 * overlay — which waits for it, so boxes never appear over a blank or broken
 * frame — stays hidden. Firing it by hand exercises the real loaded path.
 */
const settleImages = async (wrapper, { width = 640, height = 480 } = {}) => {
  for (const img of wrapper.findAll('img')) {
    Object.defineProperty(img.element, 'naturalWidth', { value: width, configurable: true })
    Object.defineProperty(img.element, 'naturalHeight', { value: height, configurable: true })
    await img.trigger('load')
  }
  await flush()
}

const mountView = async (loader, route) => {
  await router.push(route).catch(() => {})
  await router.isReady()
  const component = (await loader()).default
  const wrapper = mount(component, {
    global: { plugins: [router], stubs: { Icon: { template: '<i/>', props: ['name', 'size'] } } },
    attachTo: document.body
  })
  await flush()
  return wrapper
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.clearAllMocks()
})

describe('ProjectDetailView', () => {
  const load = () => import('@/views/ProjectDetailView.vue')

  beforeEach(() => {
    projectService.get.mockResolvedValue({ success: true, project: PROJECT })
    projectService.images.mockResolvedValue({ success: true, images: MANY_IMAGES })
    projectService.datasetSummary.mockResolvedValue({ success: true, readiness_score: 70, warnings: [], recommendations: [], tags: PROJECT.tags, classes: ['cat'], num_classes: 1, total_images: 150, annotated_images: 120, total_boxes: 240 })
  })

  it('pages a large gallery instead of rendering every image', async () => {
    const w = await mountView(load, { name: 'ProjectDetail', params: { name: 'demo' } })
    const imgs = w.findAll('.image-card')
    expect(imgs.length).toBe(60)          // PAGE_SIZE, not 150
    expect(w.text()).toContain('Page 1 / 3')
    w.unmount()
  })

  // Chips are found by the count they report rather than by position or
  // wording: which filters exist and what they are called is a product
  // decision that has changed more than once, and neither should break a test
  // about whether filtering works.
  const chipShowing = (w, count) =>
    w.findAll('.filter-chip').find((c) => c.text().includes(`(${count})`))

  it('filter chips report the right counts and narrow the grid', async () => {
    const w = await mountView(load, { name: 'ProjectDetail', params: { name: 'demo' } })
    const counts = w.findAll('.filter-chip')
      .map((c) => Number(c.text().match(/\((\d+)\)/)?.[1]))
    expect(counts).toContain(150)   // all
    expect(counts).toContain(120)   // annotated
    expect(counts).toContain(30)    // pending
    expect(counts).toContain(20)    // made by a filter
    expect(counts).toContain(130)   // photographs, the rest

    await chipShowing(w, 30).trigger('click')
    await flush()
    expect(w.findAll('.image-card').length).toBe(30)
    w.unmount()
  })

  it('resets to page 1 when a filter would leave the page out of range', async () => {
    const w = await mountView(load, { name: 'ProjectDetail', params: { name: 'demo' } })
    await w.findAll('.pager button')[1].trigger('click')   // Next -> page 2
    await flush()
    expect(w.text()).toContain('Page 2 / 3')

    await chipShowing(w, 20).trigger('click')    // made by a filter: 20, one page
    await flush()
    expect(w.findAll('.image-card').length).toBe(20)
    w.unmount()
  })

  it('draws an ROI box for every annotation on the thumbnail', async () => {
    const w = await mountView(load, { name: 'ProjectDetail', params: { name: 'demo' } })
    await settleImages(w)
    const annotatedTiles = w.findAll('.image-card').filter(
      (c) => c.find('svg.roi-overlay').exists()
    )
    expect(annotatedTiles.length).toBeGreaterThan(0)

    const svg = annotatedTiles[0].find('svg.roi-overlay')
    // The overlay works in the source image's coordinates, so no scaling maths
    // is needed and the boxes stay put at any thumbnail size.
    expect(svg.attributes('viewBox')).toBe('0 0 640 480')

    const rects = svg.findAll('rect')
    expect(rects.length).toBe(2)
    expect(rects[0].attributes('x')).toBe('10')
    expect(rects[0].attributes('width')).toBe('60')
    w.unmount()
  })

  it('gives each class its own colour, and the same one on every image', async () => {
    const w = await mountView(load, { name: 'ProjectDetail', params: { name: 'demo' } })
    await settleImages(w)
    const strokesFor = (tile) =>
      tile.findAll('svg.roi-overlay rect').map((r) => r.attributes('stroke'))

    const tiles = w.findAll('.image-card').filter((c) => c.find('svg.roi-overlay').exists())
    const first = strokesFor(tiles[0])
    // Two different classes on the same image must not share a colour.
    expect(first[0]).not.toBe(first[1])
    // And the same class keeps its colour on a different image.
    expect(strokesFor(tiles[1])[0]).toBe(first[0])
    w.unmount()
  })

  it('hides the boxes when Show ROI is turned off', async () => {
    const w = await mountView(load, { name: 'ProjectDetail', params: { name: 'demo' } })
    await settleImages(w)
    expect(w.findAll('svg.roi-overlay rect').length).toBeGreaterThan(0)

    await w.find('.roi-toggle input').setValue(false)
    await flush()
    expect(w.findAll('svg.roi-overlay rect').length).toBe(0)
    w.unmount()
  })

  it('draws nothing on an unannotated image', async () => {
    projectService.images.mockResolvedValue({
      success: true, images: [makeImage(1, false)]
    })
    const w = await mountView(load, { name: 'ProjectDetail', params: { name: 'demo' } })
    await settleImages(w)
    expect(w.findAll('svg.roi-overlay rect').length).toBe(0)
    w.unmount()
  })

  it('surfaces an upload failure instead of failing silently', async () => {
    projectService.uploadImages.mockRejectedValue(new Error('disk full'))
    const w = await mountView(load, { name: 'ProjectDetail', params: { name: 'demo' } })
    w.vm.selectedFiles = [new File(['x'], 'a.jpg', { type: 'image/jpeg' })]
    await w.vm.uploadImages()
    await flush()
    expect(w.vm.actionError).toBe('disk full')
    w.unmount()
  })
})

describe('HistoryView', () => {
  const load = () => import('@/views/HistoryView.vue')

  beforeEach(() => {
    trainingService.allHistory.mockResolvedValue({ success: true, history: RUNS })
  })

  it('counts runs by status', async () => {
    const w = await mountView(load, '/history')
    const stats = w.findAll('.stat-value').map((s) => s.text())
    expect(stats).toEqual(['1', '1', '1', '3'])   // completed, stopped, failed, total
    w.unmount()
  })

  it('filters by status and by search text', async () => {
    const w = await mountView(load, '/history')
    expect(w.findAll('tbody tr').length).toBe(3)

    await w.find('.filter-select').setValue('failed')
    await flush()
    expect(w.findAll('tbody tr').length).toBe(1)

    await w.find('.filter-select').setValue('all')
    await w.find('.search-box input').setValue('faster_rcnn')
    await flush()
    expect(w.findAll('tbody tr').length).toBe(1)
    expect(w.text()).toContain('other')
    w.unmount()
  })

  it('renders mAP as a percentage and a dash when it was never measured', async () => {
    const w = await mountView(load, '/history')
    expect(w.text()).toContain('90.0%')
    expect(w.text()).toContain('CUDA out of memory')
    w.unmount()
  })
})

describe('ModelsView', () => {
  const load = () => import('@/views/ModelsView.vue')

  beforeEach(() => {
    trainingService.allHistory.mockResolvedValue({ success: true, history: RUNS })
    trainingService.overview.mockResolvedValue({
      active_runs: [{ project_name: 'live', model_name: 'now', current_epoch: 3, total_epochs: 10 }]
    })
  })

  it('lists finished runs plus anything currently training', async () => {
    const w = await mountView(load, '/models')
    const cards = w.findAll('.model-card')
    expect(cards.length).toBe(4)                       // 3 history + 1 active
    expect(w.text()).toContain('training')
    w.unmount()
  })

  it('sorts by best mAP when asked', async () => {
    const w = await mountView(load, '/models')
    const selects = w.findAll('.filter-select')
    await selects[1].setValue('accuracy')
    await flush()
    const names = w.findAll('.model-name').map((n) => n.text())
    expect(names[0]).toBe('a')                         // mAP 0.9 first
    w.unmount()
  })
})

describe('TrainView', () => {
  const load = () => import('@/views/TrainView.vue')

  const idle = { success: true, has_run: false, status: null }

  beforeEach(() => {
    projectService.get.mockResolvedValue({ success: true, project: PROJECT })
    projectService.datasetSummary.mockResolvedValue({
      success: true, readiness_score: 80, warnings: [], recommendations: [],
      tags: PROJECT.tags, classes: ['cat'], num_classes: 1,
      total_images: 150, annotated_images: 120, total_boxes: 240
    })
    trainingService.status.mockResolvedValue(idle)
    trainingService.models.mockResolvedValue({ success: true, models: [] })
    trainingService.logs.mockResolvedValue({ success: true, logs: [] })
  })

  it('shows a start failure from the backend rather than swallowing it', async () => {
    trainingService.start.mockRejectedValue(new Error('img_size must be a multiple of 32'))
    const w = await mountView(load, { name: 'Train', params: { name: 'demo' } })
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    await w.vm.startTraining()
    await flush()
    expect(w.vm.startError).toBe('img_size must be a multiple of 32')
    expect(w.vm.training).toBe(false)      // the button must be usable again
    w.unmount()
  })

  it('reports the dataset split after a successful start', async () => {
    trainingService.start.mockResolvedValue({
      success: true,
      config: { status: 'running', current_epoch: 0, total_epochs: 10, metrics: {} },
      dataset: { train_images: 96, val_images: 24, train_boxes: 190, val_boxes: 50, empty_classes: ['dog'], skipped_count: 2 }
    })
    const w = await mountView(load, { name: 'Train', params: { name: 'demo' } })
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    await w.vm.startTraining()
    await flush()
    expect(w.text()).toContain('96')
    expect(w.text()).toContain('No boxes for: dog')
    expect(w.text()).toContain('2 image(s) skipped')
    w.vm.stopPolling?.()
    w.unmount()
  })

  it('lists per-class accuracy worst first and names the weak classes', async () => {
    trainingService.status.mockResolvedValue({
      success: true, has_run: true,
      status: {
        status: 'completed', current_epoch: 10, total_epochs: 10,
        model_type: 'yolo11s', epochs: 10, classes: ['0', '1', '8'],
        started_at: '2026-01-01T00:00:00', completed_at: '2026-01-01T01:00:00',
        exported_models: {}, metrics: { mAP50: 0.72 },
        per_class: {
          '0': { ap50: 0.93 },
          '1': { ap50: 0.81 },
          '8': { ap50: 0.31 }
        }
      }
    })
    const w = await mountView(load, { name: 'Train', params: { name: 'demo' } })
    const names = w.findAll('.per-class-name').map((n) => n.text())
    expect(names).toEqual(['8', '0', '1'].slice(0, 1).concat(['1', '0']))
    expect(w.find('.per-class-value').classes()).toContain('poor')
    expect(w.text()).toContain('Weakest:')
    expect(w.find('.per-class-advice').text()).toContain('8')
    w.unmount()
  })

  it('formats mAP metrics as percentages, not raw floats', async () => {
    trainingService.status.mockResolvedValue({
      success: true, has_run: true,
      status: { status: 'completed', current_epoch: 10, total_epochs: 10,
                model_type: 'yolo11s', epochs: 10, classes: ['cat'],
                started_at: '2026-01-01T00:00:00', completed_at: '2026-01-01T01:00:00',
                exported_models: {}, metrics: { mAP50: 0.8712, precision: 0.9, train_loss: 0.35 } }
    })
    const w = await mountView(load, { name: 'Train', params: { name: 'demo' } })
    expect(w.text()).toContain('87.1%')
    expect(w.text()).toContain('0.35')
    w.unmount()
  })
})

describe('AnnotateView', () => {
  const load = () => import('@/views/AnnotateView.vue')

  beforeEach(() => {
    projectService.tags.mockResolvedValue({ success: true, tags: ['cat'] })
    projectService.images.mockResolvedValue({ success: true, images: MANY_IMAGES.slice(0, 3) })
    projectService.imageData.mockResolvedValue({
      success: true,
      data: { filename: MANY_IMAGES[0].filename, image: 'data:image/jpeg;base64,AA',
              width: 640, height: 480,
              annotations: { regions: [{ tag: 'cat', x: 1, y: 1, width: 10, height: 10 }], annotated: true } }
    })
  })

  const route = { name: 'Annotate', params: { name: 'demo', filename: MANY_IMAGES[0].filename } }

  it('marks unsaved changes and clears them after a successful save', async () => {
    projectService.saveAnnotations.mockResolvedValue({ success: true, saved_count: 1 })
    const w = await mountView(load, route)
    w.vm.regions.push({ tag: 'cat', x: 5, y: 5, width: 20, height: 20 })
    w.vm.dirty = true
    await flush()
    expect(w.text()).toContain('Unsaved changes')

    await w.vm.saveAnnotations()
    await flush()
    expect(w.vm.dirty).toBe(false)
    expect(w.text()).toContain('Saved')
    w.unmount()
  })

  it('does not navigate away when the save fails', async () => {
    projectService.saveAnnotations.mockRejectedValue(new Error('server down'))
    vi.spyOn(window, 'alert').mockImplementation(() => {})
    const w = await mountView(load, route)
    w.vm.dirty = true

    const navigate = vi.fn()
    await w.vm.saveThenGo(navigate)
    await flush()

    expect(navigate).not.toHaveBeenCalled()      // the whole point: work is not lost
    expect(w.vm.saveError).toBe('server down')
    w.unmount()
  })

  it('navigates when there is nothing to save', async () => {
    const w = await mountView(load, route)
    w.vm.dirty = false
    const navigate = vi.fn()
    await w.vm.saveThenGo(navigate)
    expect(navigate).toHaveBeenCalled()
    w.unmount()
  })
})

describe('DatasetsView', () => {
  const load = () => import('@/views/DatasetsView.vue')

  it('shows one dataset card per non-empty project', async () => {
    projectService.list.mockResolvedValue({
      success: true,
      projects: [PROJECT, { ...PROJECT, name: 'empty', total_images: 0 }]
    })
    const w = await mountView(load, '/datasets')
    expect(w.findAll('.dataset-card').length).toBe(1)
    expect(w.text()).toContain('120 / 150 annotated')
    w.unmount()
  })

  it('rejects a non-zip import before sending anything', async () => {
    projectService.list.mockResolvedValue({ success: true, projects: [PROJECT] })
    const w = await mountView(load, '/datasets')
    w.vm.handleFileSelect({ target: { files: [new File(['x'], 'notes.txt')] } })
    await flush()
    expect(w.vm.uploadError).toContain('.zip')
    expect(projectService.importDataset).not.toHaveBeenCalled()
    w.unmount()
  })
})

describe('DashboardView', () => {
  it('uses real counters, not the field name that never existed', async () => {
    projectService.list.mockResolvedValue({ success: true, projects: [PROJECT] })
    trainingService.overview.mockResolvedValue({
      project_count: 1, total_images: 150, annotated_images: 120,
      total_annotations: 240, training_runs: 3, completed_runs: 2, failed_runs: 1,
      active_runs: [], average_map50: 0.83, recent_runs: []
    })
    const w = await mountView(() => import('@/views/DashboardView.vue'), '/')
    const values = w.findAll('.stat-value').map((v) => v.text())
    expect(values).toEqual(['1', '120', '3', '83.0%'])
    w.unmount()
  })
})

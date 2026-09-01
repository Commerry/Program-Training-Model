import http from './http'

const base = (name) => `/projects/${encodeURIComponent(name)}`

export const projectService = {
  // ── projects ──────────────────────────────────────────────────────────
  list: () =>
    http.get('/projects').then((r) => r.data),

  create: (name, description = '') =>
    http.post('/projects', { name, description }).then((r) => r.data),

  get: (name) =>
    http.get(base(name)).then((r) => r.data),

  remove: (name) =>
    http.delete(base(name)).then((r) => r.data),

  tags: (name) =>
    http.get(`${base(name)}/tags`).then((r) => r.data),

  rescan: (name, options = {}) =>
    http.post(`${base(name)}/rescan`, options).then((r) => r.data),

  datasetSummary: (name) =>
    http.get(`${base(name)}/dataset-summary`).then((r) => r.data),

  // ── images ────────────────────────────────────────────────────────────
  images: (name) =>
    http.get(`${base(name)}/images`).then((r) => r.data),

  uploadImages: (name, files, onProgress) => {
    const form = new FormData()
    for (const file of files) form.append('images', file)
    return http
      .post(`${base(name)}/images`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (event) => {
          if (onProgress && event.total) {
            onProgress(Math.round((event.loaded / event.total) * 100))
          }
        }
      })
      .then((r) => r.data)
  },

  imageData: (name, filename) =>
    http.get(`${base(name)}/images/${encodeURIComponent(filename)}`).then((r) => r.data),

  /** Delete a list of images, or every generated copy, in one request. */
  deleteImages: (name, options) =>
    http.post(`${base(name)}/images/delete`, options).then((r) => r.data),

  deleteImage: (name, filename) =>
    http.delete(`${base(name)}/images/${encodeURIComponent(filename)}`).then((r) => r.data),

  /** URL for an <img src>. Served directly by the backend, not fetched here. */
  imageUrl: (name, filename) =>
    `/api${base(name)}/images/${encodeURIComponent(filename)}/raw`,

  /**
   * Run a model over one image and get regions back, without saving anything.
   *
   * Separate from auto-labelling on purpose: that is a bulk background pass,
   * this is one image on demand, so a model can be tried on a single picture
   * before it is turned loose on a thousand.
   */
  detectOnImage: (name, filename, options) =>
    http
      .post(`${base(name)}/images/${encodeURIComponent(filename)}/detect`, options)
      .then((r) => r.data),

  /** Try a model on a few images and report what it would draw. Writes nothing. */
  previewAutoLabel: (name, options) =>
    http.post(`${base(name)}/auto-label/preview`, options).then((r) => r.data),

  classAccuracy: (name) =>
    http.get(`${base(name)}/class-accuracy`).then((r) => r.data),

  saveAnnotations: (name, filename, regions) =>
    http
      .post(`${base(name)}/images/${encodeURIComponent(filename)}/annotations`, { regions })
      .then((r) => r.data),

  // ── augmentation ──────────────────────────────────────────────────────
  tones: (name) =>
    http.get(`${base(name)}/augment-color/tones`).then((r) => r.data),

  augment: (name, options = {}) =>
    http.post(`${base(name)}/augment-color`, options).then((r) => r.data),

  // ── auto-labelling ────────────────────────────────────────────────────
  startAutoLabel: (name, options = {}) =>
    http.post(`${base(name)}/auto-label`, options).then((r) => r.data),

  autoLabelStatus: (name) =>
    http.get(`${base(name)}/auto-label`).then((r) => r.data),

  cancelAutoLabel: (name) =>
    http.post(`${base(name)}/auto-label/cancel`).then((r) => r.data),

  // ── importing a dataset labelled elsewhere ────────────────────────────
  // Read from a folder on the machine running the server rather than uploaded:
  // six thousand pictures is not a browser file picker's job.
  previewDatasetImport: (name, folder) =>
    http.post(`${base(name)}/dataset-import/preview`, { folder }).then((r) => r.data),

  startDatasetImport: (name, folder, options = {}) =>
    http.post(`${base(name)}/dataset-import`, { folder, ...options }).then((r) => r.data),

  datasetImportStatus: (name) =>
    http.get(`${base(name)}/dataset-import`).then((r) => r.data),

  cancelDatasetImport: (name) =>
    http.post(`${base(name)}/dataset-import/cancel`).then((r) => r.data),

  // ── dataset ───────────────────────────────────────────────────────────
  prepareDataset: (name) =>
    http.post(`${base(name)}/prepare-dataset`).then((r) => r.data),

  exportDataset: (name) =>
    http.post(`${base(name)}/export`, {}, { responseType: 'blob' }).then((r) => r.data),

  importDataset: (name, file) => {
    const form = new FormData()
    form.append('file', file)
    return http
      .post(`${base(name)}/import-dataset`, form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      .then((r) => r.data)
  }
}

export default projectService

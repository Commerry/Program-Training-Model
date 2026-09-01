import http from './http'

const base = (name) => `/projects/${encodeURIComponent(name)}/training`

export const trainingService = {
  options: (name) =>
    http.get(`${base(name)}/options`).then((r) => r.data),

  start: (name, config) =>
    http.post(`${base(name)}/start`, config).then((r) => r.data),

  status: (name) =>
    http.get(`${base(name)}/status`).then((r) => r.data),

  stop: (name) =>
    http.post(`${base(name)}/stop`).then((r) => r.data),

  reset: (name) =>
    http.post(`${base(name)}/reset`).then((r) => r.data),

  logs: (name, lastN = 200) =>
    http.get(`${base(name)}/logs`, { params: { last_n: lastN } }).then((r) => r.data),

  models: (name) =>
    http.get(`${base(name)}/models`).then((r) => r.data),

  /** Download URL for a weights file. The backend checks it is inside the project. */
  downloadUrl: (name, path) =>
    `/api${base(name)}/models/download?path=${encodeURIComponent(path)}`,

  history: (name) =>
    http.get(`${base(name)}/history`).then((r) => r.data),

  // ── cross-project ─────────────────────────────────────────────────────
  allHistory: () =>
    http.get('/history').then((r) => r.data),

  overview: () =>
    http.get('/overview').then((r) => r.data),

  /**
   * Boxes for one frame. Used by the webcam feed, several times a second.
   *
   * Returns coordinates only: the page already has the pixels on screen, so
   * asking the server to send an annotated copy back would cost more than the
   * prediction and would still arrive a frame late.
   */
  detectFrame: (model, frameBlob, options = {}) => {
    const form = new FormData()
    if (model && model.path) form.append('model_path', model.path)
    else form.append('model', model)
    form.append('frame', frameBlob, 'frame.jpg')
    form.append('score_threshold', String(options.scoreThreshold ?? 0.5))
    form.append('label_names', options.labelNames ?? '')
    form.append('img_size', String(options.imgSize ?? 640))
    return http
      .post('/models/detect', form,
        { headers: { 'Content-Type': 'multipart/form-data' } })
      .then((r) => r.data)
  },

  /**
   * Analyse a whole video. Returns a job immediately; poll videoStatus.
   *
   * Only a model this server trained can be used, because the analysis outlives
   * the request that started it and an uploaded copy would be gone before the
   * worker got to it.
   */
  analyseVideo: (model, videoFile, options = {}) => {
    const form = new FormData()
    form.append('model_path', model.path)
    form.append('video', videoFile)
    form.append('score_threshold', String(options.scoreThreshold ?? 0.5))
    form.append('label_names', options.labelNames ?? '')
    form.append('img_size', String(options.imgSize ?? 640))
    form.append('sample_fps', String(options.sampleFps ?? 5))
    return http
      .post('/models/video', form,
        { headers: { 'Content-Type': 'multipart/form-data' } })
      .then((r) => r.data.job)
  },

  /** URL for downloading a finished analysis as CSV. Followed by the browser. */
  videoCsvUrl: (jobId) => `/api/models/video/${encodeURIComponent(jobId)}/csv`,

  /**
   * The last test as a spreadsheet, with the annotated images in it.
   *
   * Posted rather than requested because testing images is stateless: there
   * is no run on the server to refer back to, so the results travel up with
   * the request.
   */
  exportTestResults: (payload) =>
    http.post('/models/test/export', payload, { responseType: 'blob' })
      .then((r) => r.data),

  videoStatus: (jobId) =>
    http.get(`/models/video/${encodeURIComponent(jobId)}`).then((r) => r.data.job),

  stopVideo: (jobId) =>
    http.post(`/models/video/${encodeURIComponent(jobId)}/stop`).then((r) => r.data.job),

  /** Every model this installation has trained, newest first. */
  listTrainedModels: () => http.get('/models').then((r) => r.data.models || []),

  /**
   * Bring in a detector built somewhere else, so it can pre-label a project
   * that has nothing yet. It is stored as a folder: an ONNX carries no class
   * names, and nothing in it records how it wants to be fed.
   */
  importModel: (modelFile, options = {}) => {
    const form = new FormData()
    form.append('model', modelFile)
    if (options.name) form.append('name', options.name)
    if (options.labelsFile) form.append('labels_file', options.labelsFile)
    if (options.onnxConventions) form.append('onnx_conventions', options.onnxConventions)
    return http
      .post('/models/import', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      .then((r) => r.data)
  },

  /** The class names for a model imported without them. */
  setImportedLabels: (folderName, labelsFile) => {
    const form = new FormData()
    form.append('labels_file', labelsFile)
    return http
      .post(`/models/imported/${encodeURIComponent(folderName)}/labels`, form,
            { headers: { 'Content-Type': 'multipart/form-data' } })
      .then((r) => r.data)
  },

  listImportedModels: () => http.get('/models/imported').then((r) => r.data.models || []),

  /**
   * Run a model over some images.
   *
   * `model` is either an uploaded File or, for one the server trained itself,
   * a { path } naming it — sending the path avoids downloading the weights and
   * posting them straight back to the machine that wrote them.
   */
  testModel: (model, imageFiles, options = {}) => {
    const form = new FormData()
    if (model && model.path) form.append('model_path', model.path)
    else form.append('model', model)
    for (const file of imageFiles) form.append('images', file)
    form.append('score_threshold', String(options.scoreThreshold ?? 0.5))
    form.append('label_names', options.labelNames ?? '')
    form.append('img_size', String(options.imgSize ?? 640))
    // An ONNX carries no class names. An export that has them keeps them in a
    // labels.txt beside the model, which does not come along with the model
    // file on its own.
    if (options.labelsFile) form.append('labels_file', options.labelsFile)
    if (options.onnxConventions) {
      form.append('onnx_conventions', options.onnxConventions)
    }
    return http
      .post('/models/test', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      .then((r) => r.data)
  }
}

export default trainingService

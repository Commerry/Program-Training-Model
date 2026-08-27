/**
 * Drive the video and webcam modes of the model test screen in a real browser.
 *
 * Both depend on browser APIs no unit test can stand in for: getUserMedia, a
 * <video> element that decodes a real file, canvas.toBlob, and the timing loop
 * that keeps only one request in flight. Chromium is launched with a fake
 * camera so the webcam path runs unattended.
 *
 *   node frontend/tests/browser-live.mjs [baseUrl] [clip.webm]
 *
 * Expects a built UI being served and at least one trained model on the
 * server. The clip is optional; without it the video mode is skipped rather
 * than reported as broken. webm/VP8 is the container to use, because it is the
 * one both Chromium and this OpenCV build handle -- H.264 is not, the openh264
 * library the mp4 writer wants is not reliably installed.
 */
import { existsSync } from 'node:fs'
import { chromium } from 'playwright'

const BASE = process.argv[2] || 'http://127.0.0.1:5178'
const CLIP = process.argv[3] || ''

const IGNORED = [/favicon/i, /Download the Vue Devtools/i, /\[vite\]/i]

let checks = 0
let failures = 0
const findings = []

const check = (label, condition, detail = '') => {
  checks += 1
  if (condition) {
    console.log(`  PASS ${label}`)
  } else {
    failures += 1
    console.log(`  FAIL ${label}${detail ? `  -> ${detail}` : ''}`)
    findings.push(`${label}${detail ? `: ${detail}` : ''}`)
  }
}

const run = async () => {
  const browser = await chromium.launch({
    args: [
      // A moving synthetic pattern on a virtual camera, so getUserMedia
      // succeeds with no hardware and no permission prompt.
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
    ],
  })
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    permissions: ['camera'],
  })
  const page = await context.newPage()
  // Poll loops below read elements that may legitimately be absent. Without a
  // short timeout each miss costs the full 30-second default and a 90-turn
  // loop runs for the better part of an hour.
  page.setDefaultTimeout(4000)

  let consoleErrors = []
  let failed = []
  page.on('console', (m) => {
    if (m.type() !== 'error') return
    if (IGNORED.some((p) => p.test(m.text()))) return
    consoleErrors.push(m.text())
  })
  page.on('pageerror', (e) => consoleErrors.push(`uncaught: ${e.message}`))
  page.on('response', (r) => {
    if (r.url().includes('/api/') && r.status() >= 400 && !r.url().includes('/auth/me')) {
      failed.push(`${r.status()} ${r.request().method()} ${r.url().replace(BASE, '')}`)
    }
  })

  const models = await (await context.request.get(`${BASE}/api/models`)).json()
  const trained = (models.models || [])[0]
  if (!trained) {
    console.log('no trained model on this server; nothing to drive')
    await browser.close()
    process.exit(0)
  }
  console.log(`using ${trained.name} from ${trained.project}/${trained.run}\n`)

  await page.goto(`${BASE}/test-model`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(800)

  console.log('== the three modes are offered ==')
  const tabs = page.locator('.mode-tab')
  check('three modes on screen', await tabs.count() === 3, await tabs.count())
  check('images is the one selected first',
    (await tabs.first().getAttribute('aria-selected')) === 'true')

  await page.locator('.trained-item').first().click()
  await page.waitForTimeout(200)

  // ── Webcam ────────────────────────────────────────────────────────────────
  console.log('\n== webcam ==')
  consoleErrors = []
  failed = []
  await tabs.nth(2).click()
  await page.waitForTimeout(300)
  check('switching to the camera mode raises nothing',
    consoleErrors.length === 0, consoleErrors.slice(0, 2).join(' | '))

  const blocked = await page.locator('.error-box').count() > 0
  check('the camera is available in this context', !blocked,
    blocked ? (await page.locator('.error-box').innerText()).slice(0, 100) : '')

  const runButton = page.locator('.run-btn')
  check('the start button is enabled once a model is picked',
    await runButton.isEnabled())

  await runButton.click()
  await page.waitForTimeout(6000)

  const liveVideo = page.locator('.live-video')
  check('the camera feed is playing',
    await liveVideo.evaluate((el) => el.videoWidth > 0 && !el.paused))
  check('running the feed raises no console error', consoleErrors.length === 0,
    consoleErrors.slice(0, 2).join(' | '))
  check('the server accepted every frame', failed.length === 0,
    failed.slice(0, 2).join(' | '))

  const frameCalls = await page.evaluate(() =>
    performance.getEntriesByType('resource')
      .filter((e) => e.name.includes('/models/detect')).length)
  console.log(`    ${frameCalls} frames sent in about 6 s`)
  check('frames are being sent continuously', frameCalls >= 3, frameCalls)

  // The loop must never have more than one request outstanding, or a slow
  // model makes the feed fall further behind the longer it runs.
  const overlapping = await page.evaluate(() => {
    const calls = performance.getEntriesByType('resource')
      .filter((e) => e.name.includes('/models/detect'))
      .sort((a, b) => a.startTime - b.startTime)
    let overlaps = 0
    for (let i = 1; i < calls.length; i += 1) {
      if (calls[i].startTime < calls[i - 1].responseEnd - 1) overlaps += 1
    }
    return overlaps
  })
  check('only one request is in flight at a time', overlapping === 0,
    `${overlapping} overlapped`)

  const stats = await page.locator('.field-note').last().innerText().catch(() => '')
  if (stats) console.log(`    ${stats.replace(/\s+/g, ' ').trim()}`)

  await runButton.click()
  await page.waitForTimeout(800)
  check('stopping releases the camera',
    await liveVideo.evaluate((el) => el.srcObject === null))

  // Leaving the page must release it too, or the browser keeps the in-use
  // indicator lit and the camera light on until the tab is closed.
  await runButton.click()
  await page.waitForTimeout(2500)
  await page.goto(`${BASE}/projects`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(800)
  check('navigating away tears the feed down',
    await page.locator('video').count() === 0)

  // ── Video ────────────────────────────────────────────────────────────────
  console.log('\n== video ==')
  await page.goto(`${BASE}/test-model`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(800)
  await page.locator('.trained-item').first().click()
  await page.locator('.mode-tab').nth(1).click()
  await page.waitForTimeout(300)

  consoleErrors = []
  failed = []

  if (!CLIP || !existsSync(CLIP)) {
    console.log('    (no clip given as the third argument; skipped)')
  } else {
    await page.locator('input[type="file"]').nth(1).setInputFiles(CLIP)
    await page.waitForTimeout(500)
    check('the chosen clip is shown', await page.locator('.chosen-name').count() > 0)

    await page.locator('.run-btn').click()
    let finished = false
    for (let i = 0; i < 90; i += 1) {
      await page.waitForTimeout(1000)
      const bar = page.locator('.live-bar')
      if (await bar.count() === 0) continue
      const text = await bar.innerText().catch(() => '')
      if (text && !/วิเคราะห์แล้ว/.test(text)) { finished = true; break }
    }
    check('the analysis finished', finished)
    check('it finished without a console error', consoleErrors.length === 0,
      consoleErrors.slice(0, 2).join(' | '))
    check('the server accepted the video', failed.length === 0,
      failed.slice(0, 2).join(' | '))

    const summary = await page.locator('.live-bar').innerText().catch(() => '')
    console.log(`    ${summary.replace(/\s+/g, ' ').trim()}`)

    const playback = page.locator('.live-video')
    check('the clip plays in the page',
      await playback.evaluate((el) => el.videoWidth > 0),
      'the browser could not decode it')
  }

  // ── The overlay itself ───────────────────────────────────────────────────
  // Whether a model detects anything is a property of that model, not of this
  // page: the projects on a given server may only carry a smoke-test model
  // that finds nothing, and then the checks below would report the page as
  // broken. So the boxes are supplied directly and what is measured is the
  // part that belongs to the page — that a sample is drawn in the frame's own
  // coordinates, and that scrubbing moves to a different sample.
  console.log('\n== the overlay, with known boxes ==')
  const FRAMES = [
    { time_s: 0.0, frame: 0,  detections: [
      { label_id: 0, label_name: 'left', score: 0.9, box: [10, 20, 90, 100] }] },
    { time_s: 2.0, frame: 20, detections: [
      { label_id: 1, label_name: 'right', score: 0.8, box: [400, 60, 520, 200] }] },
  ]

  await page.route('**/api/models/video', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ success: true, job: {
        id: 'stub', status: 'running', message: '', frames_done: 0,
        frames_total: 0, detection_count: 0, frames: [],
        width: 640, height: 480, fps: 10, duration_s: 5, sample_fps: 5,
      } }),
    })
  })
  await page.route('**/api/models/video/stub', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ success: true, job: {
        id: 'stub', status: 'completed', message: 'stubbed',
        frames_done: 2, frames_total: 2, detection_count: 2, elapsed_s: 0.1,
        label_names: ['left', 'right'], frames: FRAMES,
        width: 640, height: 480, fps: 10, duration_s: 5, sample_fps: 5,
      } }),
    })
  })

  await page.goto(`${BASE}/test-model`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(700)
  await page.locator('.trained-item').first().click()
  await page.locator('.mode-tab').nth(1).click()
  await page.waitForTimeout(200)

  if (CLIP && existsSync(CLIP)) {
    await page.locator('input[type="file"]').nth(1).setInputFiles(CLIP)
    await page.waitForTimeout(400)
    await page.locator('.run-btn').click()
    await page.waitForTimeout(2500)

    const stage = page.locator('.live-video')
    await stage.evaluate((el) => { el.currentTime = 0.1 })
    await page.waitForTimeout(500)
    const early = await page.locator('.detection-overlay rect').first()
      .getAttribute('x').catch(() => null)

    await stage.evaluate((el) => { el.currentTime = 3.0 })
    await page.waitForTimeout(500)
    const late = await page.locator('.detection-overlay rect').first()
      .getAttribute('x').catch(() => null)

    check('a box is drawn for the sample at the playhead', early === '10', `x=${early}`)
    check('scrubbing forward moves to the next sample', late === '400', `x=${late}`)
    check('the overlay uses the frame coordinate system',
      (await page.locator('.detection-overlay').getAttribute('viewBox')) === '0 0 640 480',
      await page.locator('.detection-overlay').getAttribute('viewBox'))
    // An SVG <text> is not an HTMLElement, so innerText is not available on it.
    const labelText = await page.locator('.detection-overlay text').first()
      .evaluate((el) => el.textContent)
    check('the class name is shown on the box',
      labelText.trim().startsWith('right'), labelText)
  } else {
    console.log('    (no clip; skipped)')
  }

  await browser.close()
  console.log(`\n${checks} checks, ${failures} failed`)
  if (findings.length) {
    console.log('\nfindings:')
    findings.forEach((f) => console.log(`  - ${f}`))
  }
  process.exit(failures ? 1 : 0)
}

run().catch((err) => {
  console.error(err)
  process.exit(2)
})

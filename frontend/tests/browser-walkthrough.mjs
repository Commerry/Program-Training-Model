/**
 * Drive the whole application in a real browser against a real backend.
 *
 * The component tests mount views with the network stubbed, so they prove the
 * markup renders but say nothing about whether a page works once it is talking
 * to the actual server: a field renamed on one side, a route that 404s, a
 * computed that throws on a shape the stub never produced. Those only show up
 * here.
 *
 * Every console error and every failed request is collected per page and
 * reported at the end, so one broken page does not hide the rest.
 *
 *   node frontend/tests/browser-walkthrough.mjs [baseUrl]
 *
 * Expects the app already serving on the URL given (default http://127.0.0.1:5178).
 */
import { chromium } from 'playwright'

const BASE = process.argv[2] || 'http://127.0.0.1:5178'

// Noise that says nothing about the application being broken.
const IGNORED = [
  /favicon/i,
  /Download the Vue Devtools/i,
  /\[vite\]/i,
  /ResizeObserver loop/i,
]

const PAGES = [
  // '/' is a redirect to the dashboard, so name where it is expected to land.
  { path: '/', name: 'Dashboard', lands: '/dashboard' },
  { path: '/projects', name: 'Projects' },
  { path: '/datasets', name: 'Datasets' },
  { path: '/train', name: 'Train' },
  { path: '/models', name: 'Models' },
  { path: '/test-model', name: 'Test model' },
  { path: '/analytics', name: 'Analytics' },
  { path: '/history', name: 'History' },
  { path: '/settings', name: 'Settings' },
]

const findings = []
let checks = 0
let failures = 0

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
  const browser = await chromium.launch()
  const context = await browser.newContext({ viewport: { width: 1440, height: 960 } })
  const page = await context.newPage()

  let consoleErrors = []
  let failedRequests = []

  page.on('console', (msg) => {
    if (msg.type() !== 'error') return
    const text = msg.text()
    if (IGNORED.some((pattern) => pattern.test(text))) return
    consoleErrors.push(text)
  })
  page.on('pageerror', (err) => consoleErrors.push(`uncaught: ${err.message}`))
  page.on('response', (response) => {
    const url = response.url()
    if (!url.includes('/api/')) return
    // 401 on /auth/me before sign-in is the normal startup path, not a failure.
    if (response.status() >= 400 && !url.includes('/auth/me')) {
      failedRequests.push(`${response.status()} ${response.request().method()} ` +
        url.replace(BASE, ''))
    }
  })

  const visit = async (path) => {
    consoleErrors = []
    failedRequests = []
    await page.goto(`${BASE}${path}`, { waitUntil: 'networkidle', timeout: 30000 })
    await page.waitForTimeout(700)
  }

  console.log(`walking ${BASE}\n`)

  // ── every page loads clean ────────────────────────────────────────────────
  for (const { path, name, lands } of PAGES) {
    console.log(`== ${name} (${path}) ==`)
    try {
      await visit(path)
    } catch (err) {
      check(`${name} loads`, false, err.message.split('\n')[0])
      continue
    }
    check(`${name} loads`, true)

    // Without this every check below can pass against the login screen — which
    // is how an earlier version of this file reported a clean run while never
    // once getting past the front door.
    check(`${name} is the page asked for, not a redirect`,
      new URL(page.url()).pathname === (lands || path),
      `landed on ${new URL(page.url()).pathname}`)

    check(`${name} renders content`,
      (await page.locator('body').innerText()).trim().length > 40)
    check(`${name} has no console error`, consoleErrors.length === 0,
      consoleErrors.slice(0, 2).join(' | '))
    check(`${name} has no failing request`, failedRequests.length === 0,
      failedRequests.slice(0, 3).join(' | '))

    // A page that renders its own error banner is reporting a real problem.
    const banner = page.locator('.error, .error-banner, .alert-error').first()
    if (await banner.count()) {
      const text = (await banner.innerText().catch(() => '')).trim()
      check(`${name} shows no error banner`, !text, text.slice(0, 120))
    }
  }

  // ── the page bodies must not scroll sideways ──────────────────────────────
  console.log('\n== layout ==')
  for (const { path, name } of PAGES) {
    await visit(path)
    const overflows = await page.evaluate(() =>
      document.documentElement.scrollWidth > document.documentElement.clientWidth + 1)
    check(`${name} does not scroll horizontally`, !overflows)
  }

  // ── the cursor only appears where text can be typed ───────────────────────
  console.log('\n== cursor ==')
  for (const { path, name } of PAGES) {
    await visit(path)
    const wrong = await page.evaluate(() => {
      const bad = []
      for (const el of document.querySelectorAll('h1, h2, h3, h4, p, span, label, td, th, li')) {
        if (!el.offsetParent) continue
        if (el.closest('input, textarea, [contenteditable="true"]')) continue
        const cursor = getComputedStyle(el).cursor
        if (cursor === 'text' || cursor === 'auto') {
          bad.push(`${el.tagName.toLowerCase()}.${el.className || '-'}: ${cursor}`)
        }
      }
      return bad.slice(0, 3)
    })
    check(`${name} shows no text cursor on non-editable text`, wrong.length === 0,
      wrong.join(' | '))
  }

  // ── inputs must still show one ────────────────────────────────────────────
  await visit('/settings')
  const inputCursor = await page.evaluate(() => {
    const el = document.querySelector('input[type="text"], input[type="password"], input:not([type])')
    return el ? getComputedStyle(el).cursor : 'no input found'
  })
  check('a text input still shows the typing cursor', inputCursor === 'text', inputCursor)

  // ── a project page shows its images with ROI boxes ────────────────────────
  // Projects open through router.push on a click handler rather than an
  // anchor, so ask the API which one has images and go there directly.
  console.log('\n== project detail ==')
  const projects = await (await context.request.get(`${BASE}/api/projects`)).json()
  const withImages = (projects.projects || []).find((p) => p.annotated_images > 0)

  if (withImages) {
    console.log(`  using "${withImages.name}" (${withImages.annotated_images} annotated)`)
    await visit(`/projects/${encodeURIComponent(withImages.name)}`)
    await page.waitForTimeout(1500)
    check('the project page has no console error', consoleErrors.length === 0,
      consoleErrors.slice(0, 2).join(' | '))
    check('the project page has no failing request', failedRequests.length === 0,
      failedRequests.slice(0, 3).join(' | '))

    const thumbs = await page.locator('img[src*="/raw"]').count()
    check('sample images are displayed', thumbs > 0, `${thumbs} images`)
    const rois = await page.locator('svg rect').count()
    check('ROI boxes are drawn over them', rois > 0, `${rois} boxes`)

    // Every drawn box must have real dimensions; a zero-sized rect is what a
    // wrong coordinate space produces, and it looks like nothing at all.
    const collapsed = await page.evaluate(() =>
      [...document.querySelectorAll('svg rect')]
        .filter((r) => r.getBBox().width < 1 || r.getBBox().height < 1).length)
    check('no ROI box collapsed to nothing', collapsed === 0, `${collapsed} collapsed`)

    // ── the annotate screen, the largest view in the app ───────────────────
    console.log('\n== annotate ==')
    const images = await (await context.request.get(
      `${BASE}/api/projects/${encodeURIComponent(withImages.name)}/images`)).json()
    const annotated = (images.images || []).find((i) => i.annotated)
    if (annotated) {
      await visit(`/projects/${encodeURIComponent(withImages.name)}` +
        `/annotate/${encodeURIComponent(annotated.filename)}`)
      await page.waitForTimeout(1500)
      check('the annotate page has no console error', consoleErrors.length === 0,
        consoleErrors.slice(0, 2).join(' | '))
      check('the annotate page has no failing request', failedRequests.length === 0,
        failedRequests.slice(0, 3).join(' | '))
      const canvasOrImage = await page.locator('canvas, img[src*="/raw"]').count()
      check('the image to annotate is on screen', canvasOrImage > 0)
      const existing = await page.locator('svg rect, .region-box, .roi-box').count()
      check('the boxes already saved for it are shown', existing > 0, `${existing}`)
    }
  } else {
    console.log('  (no project with annotated images — skipped)')
  }

  // ── the test-model screen offers models the server trained ────────────────
  console.log('\n== model picker ==')
  await visit('/test-model')
  const picker = page.locator('.trained-item')
  const pickerCount = await picker.count()
  console.log(`  ${pickerCount} trained model(s) offered`)
  if (pickerCount > 0) {
    await picker.first().click()
    await page.waitForTimeout(300)
    check('picking a trained model selects it',
      await picker.first().evaluate((el) => el.classList.contains('trained-item--on')))

    // Feed it an image and press the button: the point of the picker is that a
    // model the server trained can be tried without downloading the weights
    // and posting them back to the machine that wrote them.
    //
    // The image is one of the project's own, fetched from the server, rather
    // than a base64 blob pasted in here — an earlier version used a hand-typed
    // PNG with a broken CRC, and the 400 the server correctly returned for it
    // read like a bug in the application.
    let probe = null
    if (withImages) {
      const list = await (await context.request.get(
        `${BASE}/api/projects/${encodeURIComponent(withImages.name)}/images`)).json()
      const sample = (list.images || [])[0]
      if (sample) {
        const raw = await context.request.get(
          `${BASE}/api/projects/${encodeURIComponent(withImages.name)}` +
          `/images/${encodeURIComponent(sample.filename)}/raw`)
        probe = { name: sample.filename, mimeType: 'image/jpeg', buffer: await raw.body() }
      }
    }

    const fileInputs = page.locator('input[type="file"]')
    if (!probe) {
      check('an image was available to test with', false, 'no project image')
    } else {
      await fileInputs.nth(1).setInputFiles(probe)
    }
    await page.waitForTimeout(400)

    const runButton = page.locator('button', { hasText: /ทดสอบโมเดล|Run|Test/i }).last()
    if (await runButton.isEnabled()) {
      consoleErrors = []
      failedRequests = []
      await runButton.click()
      await page.waitForTimeout(9000)
      check('running the picked model raises no error', consoleErrors.length === 0,
        consoleErrors.slice(0, 2).join(' | '))
      check('the server accepted the picked model', failedRequests.length === 0,
        failedRequests.slice(0, 2).join(' | '))
      const shown = (await page.locator('body').innerText()).replace(/\s+/g, ' ')
      check('a result came back', /detect|ผลลัพธ์|result|0 |box/i.test(shown))
    } else {
      check('the run button enables once a model and an image are chosen', false,
        'still disabled')
    }
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

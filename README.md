# Vision Training Platform

A self-hosted object-detection training tool in the spirit of Azure Custom
Vision: create a project, upload images, draw and tag boxes, train a YOLO or
Faster R-CNN model, and export the weights.

```
Train-Model-Webapp-main/
├── backend/              Flask API and training workers (Python)
│   ├── app.py            entry point — python backend/app.py
│   ├── config.py         paths, ports, env configuration
│   ├── models.py         user accounts (the only database table)
│   ├── api/              HTTP routes, one module per area
│   ├── services/         all filesystem and dataset logic
│   ├── training/         worker processes and model libraries
│   ├── scripts/          doctor.py, migrate_layout.py
│   └── tests/            runnable checks (see Testing below)
├── frontend/             Vue 3 single-page app
│   └── src/
│       ├── services/     one module per API area
│       ├── stores/       Pinia stores (auth, projects)
│       ├── views/        one per route
│       ├── components/   shared UI
│       └── utils/        formatting helpers
│   └── tests/            component tests (vitest)
└── data/
    ├── projects/         your datasets — one folder per project
    ├── weights/          cached pretrained checkpoints
    └── instance/         SQLite database and the generated secret key
```

## Installing on another machine

Needs **Python 3.10+** and **Node 18+**. Everything else the installer handles.

### Windows

```powershell
git clone https://github.com/Commerry/Program-Training-Model.git
cd Program-Training-Model

# PowerShell blocks unsigned scripts by default; allow them for this session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\start.ps1 -Install
```

That installs both dependency sets and starts the backend and the Vite dev
server in two windows. Open <http://localhost:64030>.

If `python` or `npm` is missing, install them first — from
[python.org](https://www.python.org/downloads/windows/) (tick **Add python.exe
to PATH**) and [nodejs.org](https://nodejs.org/) — then reopen PowerShell so the
new PATH is picked up.

### Linux

```bash
git clone https://github.com/Commerry/Program-Training-Model.git
cd Program-Training-Model

# Debian / Ubuntu — the OpenCV wheel needs libGL and libglib at run time
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nodejs npm libgl1 libglib2.0-0

python3 -m venv .venv
source .venv/bin/activate

chmod +x start.sh
./start.sh --install
```

Open <http://localhost:64030>.

`start.sh` uses `.venv/bin/python` when that directory exists, so the virtualenv
is picked up on later runs without activating it by hand.

On Fedora or RHEL the package names differ:

```bash
sudo dnf install -y python3 python3-pip nodejs npm mesa-libGL glib2
```

If `npm` is not available and you would rather not install Node, skip it: build
the UI once anywhere that does have Node, copy `frontend/dist` across, and run
`./start.sh --network`, which serves that build straight from the backend.

### Both platforms, step by step

If you would rather not use the launcher:

```bash
pip install -r backend/requirements.txt
cd frontend && npm install && npm run build && cd ..

python backend/app.py        # serves the API and the built UI on :64031
```

Or run the dev server separately for live reloading:

```bash
python backend/app.py                    # terminal 1 — API on :64031
cd frontend && npm run dev               # terminal 2 — UI on :64030
```

### First run

An `admin` account is created on a fresh database and its password is printed
to the backend console. Set `ADMIN_PASSWORD` before the first start to choose
your own.

Copy `.env.example` to `.env` to change ports, move the projects directory, or
turn authentication off for a single-user machine. It is read at startup.

The server binds to `127.0.0.1` with the debugger off. Both are deliberate: see
below for opening it up.

### Use the GPU — do this before your first real training run

```bash
python backend/scripts/doctor.py             # packages, GPU, disk
python backend/scripts/doctor.py my-project  # also validates that project's dataset
```

This is the single most valuable command in the repo. A plain
`pip install torch` — which `requirements.txt` performs — installs a
**CPU-only** build. Training then still works, reports a falling loss, and
finishes tens of times slower with nothing on screen to suggest anything is
wrong. One run measured here took roughly 35 hours on a CPU against about 1.2
hours on a GTX 1650.

If `doctor.py` reports a GPU that PyTorch cannot use, reinstall it:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Use the index URL matching your driver — `cu121`, `cu124` and `cu126` are the
common ones; check <https://pytorch.org/get-started/locally/>. Then run
`doctor.py` again; it should report CUDA.

On Linux the NVIDIA driver has to be installed at the system level first
(`nvidia-smi` should print your card). The CUDA toolkit itself does not need to
be installed separately — the PyTorch wheel bundles what it needs.

### Opening it to other machines

The dev setup listens on this machine only. To reach the app from another
computer by IP:

```powershell
.\start.ps1 -Network        # Windows
```

```bash
./start.sh --network        # Linux
```

That builds the frontend and serves it from the backend on **one port**, then
prints the addresses to open. One port matters: it is far easier to allow
through a firewall, needs no Node process on the serving machine, and keeps the
UI and API same-origin so the login cookie works with no CORS configuration.

A firewall will otherwise report `ERR_CONNECTION_REFUSED` even though the
server is running:

```powershell
.\start.ps1 -Firewall                                  # Windows, admin PowerShell
```

```bash
sudo ufw allow 64031/tcp                               # Ubuntu / Debian
sudo firewall-cmd --add-port=64031/tcp --permanent && sudo firewall-cmd --reload
```

`start.sh --network` detects which of the two is running and prints the right
line for you.

**Before leaving it open**, sign in and change the admin password in Settings.
The default lets anyone on the network in. The session signing key needs no
attention: one is generated on first run and kept in `data/instance/secret_key`.

To go back to local-only, run the launcher with no arguments.

### Moving a dataset to the new machine

Cloning the repository does **not** bring your images with it — they are large
and excluded on purpose. Annotations *are* committed, since they represent work
that cannot be regenerated, but the images they describe are not.

To carry a whole project across, on the machine that has it:

```bash
python backend/scripts/pack_project.py --list        # what this machine can see
python backend/scripts/pack_project.py my-project    # writes my-project_dataset.zip
```

Copy the zip over, then on the new machine:

```bash
python backend/scripts/pack_project.py --restore my-project_dataset.zip
```

Reload the page, or press **Check again** on the project.

## What was fixed after that

The two faults below are the ones that made the tool unusable. Three more
turned up when every preset and every page was measured rather than assumed.

**Two of the 26 colour filters were the same filter.** `gamma_low` and
`gamma_high` both worked out to `pixel ** 0.5`, so they emitted images that
were identical pixel for pixel — one preset wasted, and a duplicate row added
to the dataset for every source image. `gamma_low` now darkens and `gamma_high`
brightens, as their names say.

**The sharpen filter halved the brightness of everything it touched.** Its
kernel summed to 0.5 instead of 1, so it sharpened *and* darkened by 49%,
duplicating what the `dark` preset already did. It now leaves exposure alone
(measured shift: 0.2%) while raising edge energy from 115 to 887.

**Three filters buried the object in noise.** `laplacian`, `sobel` and
`adaptive_thresh` fed the raw frame into a derivative or a local threshold, so
sensor grain out-responded the object being labelled — the annotation ended up
pointing at something less distinct than its background. They smooth first now,
which is what Canny already did internally and why `canny_overlay` never had
the problem.

**A filter that erases the object is now dropped rather than trained on.**
Presets suit different data — black-hat looks for sunken digits and finds
nothing on raised ones — and an image whose label points at a blank patch
teaches the detector that the class looks like empty background. Every
generated image is measured against its own source before being kept, and the
project page reports which presets were dropped and how many.

**Setting `REQUIRE_AUTH=0` did nothing.** The backend accepted every request,
but the router still bounced every page to a login form, so the documented way
to run this open on a single machine locked you out of it. The client now reads
`auth_required` from `/api/health`, and defaults to requiring a login if it
cannot reach the server, so a failure fails closed.

**Testing a model meant uploading it back to the machine that wrote it.** The
test screen only accepted a file from disk, so trying a model you had just
trained meant finding `best.pt` under
`data/projects/<name>/training/runs/<run>/weights` and posting it to the server
that had produced it. It now lists what this installation has trained and runs
the chosen one directly; the path is resolved against the projects tree first,
so it cannot be pointed at anything else on the filesystem.

## Testing a model: images, video, or the camera

The model test screen runs a model three ways. All three take the same model,
threshold and class names; only the source differs.

**Images** — pick some files, get them back annotated.

**Video** — pick a clip and the server samples it (five frames a second by
default) and returns what it found at each instant. The clip itself is never
sent back: the browser plays the file you already chose and draws the boxes
over it, scrubbing included. That is not only cheaper, it is the only version
that works everywhere. Writing H.264 from OpenCV needs an openh264 library that
is not reliably installed — on the machine this was built on the mp4 writer
reported "Incorrect library version loaded" and produced a file no browser
would accept — so a server that returned an annotated video would work on one
install and silently fail on the next. The cost is that there is no annotated
file to download afterwards.

**Camera** — the page grabs a frame, sends it, draws the boxes, and repeats.
Measured here: 22 frames a second against a small YOLO on a CPU. Only one
request is ever in flight; firing on a timer regardless would queue requests
behind each other the moment inference is slower than the interval and the feed
would fall further behind the longer it ran.

Two things are worth knowing about the camera:

- **It needs https or localhost.** `getUserMedia` is only available in a secure
  context, so opening the app from another machine over plain http gives no
  camera at all. Browsers report that as the API simply being absent, so the
  page says so rather than leaving a button that does nothing. Use the machine
  running the server, at `http://localhost:<port>`.
- **The model is kept in memory between frames.** Loading a small YOLO costs
  131 ms against 40 ms to predict, so reloading per frame would cap the feed
  near 6 fps. Models are cached by path, size and modification time, so
  retraining into the same filename is noticed rather than served stale, and an
  uploaded model is stored under the hash of its contents so repeated frames
  from one session hit the same entry.

A model is also run at the size it was built for rather than at the screen's
default. An ONNX export has its resolution compiled into the graph: asking a
320-export for 640 raised `Got invalid dimensions for input: images` from
onnxruntime and reached the browser as an opaque 500. A `.pt` records what it
was trained at, and honouring that matters too — a model trained at 320 and run
at 640 detects far less, with nothing on screen to say why.

## Why the earlier runs were wasted

Two independent faults, both now fixed. `python backend/scripts/why_old_runs_failed.py`
reproduces them from the original code and prints the numbers.

**The model detected nothing.** The dataset builder read image dimensions from
a field that was never populated, so every box was normalised by 1x1 instead of
by the real pixel size. A box at (210, 150) on a 640x480 photo was written to
the label file as `258.0 210.0 96.0 120.0` — YOLO requires all four values in
`[0, 1]`, so every label was clipped to the same degenerate corner. Training ran,
reported a falling loss, and learned nothing.

**A run took days.** `pip install torch` gives a CPU-only build. With one
present, training silently falls back to the CPU: roughly 35 hours for 100
epochs over 1785 images at 640px, against about 1.2 hours on a GTX 1650.

## How it works

### Data layout

Each project is a directory under `data/projects/`:

```
my-project/
├── project.json            name, description, cached counts
├── index.json              gallery cache, rebuilt when it goes stale
├── images/                 source images, never modified
├── annotations/            one JSON per image, boxes in image pixels
├── training/
│   ├── training_config.json   live status of the current/last run
│   ├── training.log
│   ├── history.json           one record per finished run
│   ├── dataset/               generated YOLO split (rebuilt each run)
│   └── runs/<name>/weights/   checkpoints and exports
└── exports/                dataset zips
```

The annotation files are the source of truth. `project.json` and `index.json`
are caches derived from them.

`index.json` exists because the gallery once opened all 2232 annotation files
on every request — 18.8 seconds cold on Windows, where real-time antivirus adds
about 8 ms to each first open. Keeping the per-image summary in one file makes
that request 45 ms. The cache records how many image and annotation files it
was built from plus the newest annotation timestamp; if the directory no longer
matches, it is rebuilt. Training always rebuilds it first, so a run never
depends on a possibly stale cache.

### Training

Training runs in a **separate process**. The web server writes a config file,
launches a worker, and then only reads the status the worker writes back. A run
that crashes or exhausts GPU memory cannot take the API down with it, and the
UI polls the same file for progress.

Two model families are supported:

| Model | Worker | Good for |
| --- | --- | --- |
| YOLOv8 / v9 / v10 / v11, RT-DETR | `training/yolo_worker.py` | Almost everything. Fast, accurate, exports cleanly. |
| Faster R-CNN ResNet50-FPN | `training/frcnn_worker.py` | Small objects where a two-stage detector helps. Considerably slower. |

Both train on the same prepared train/val split, so their metrics are
comparable.

### Auto-labelling

Once a project has one completed run, **Auto-label** runs that model over every
image that still has no annotations and writes its predictions in as ordinary,
editable boxes. Correcting a drawn box takes a few seconds; drawing one takes
far longer, so this is the difference between days and hours on a large set.

Inference uses the image size the model was trained at, not a fixed 640 — a
detector is sensitive to object scale, and the mismatch measurably suppresses
detections. Everything written is tagged `auto_labelled` so it can be reviewed,
and the run that produced it is recorded alongside.

### Reading the results

After a run, the training page lists **accuracy per class**, worst first. One
overall mAP cannot distinguish a detector that reads every digit adequately
from one that reads nine perfectly and never sees an 8; the per-class list
names the classes worth collecting more images for.

### Seeing the annotations

Every gallery thumbnail draws its ROI boxes, one colour per class. Colours come
from the class's position in the project's sorted class list stepped by the
golden angle, so a given class keeps its colour on every image and no two are
close together — hashing the class name instead put `0` and `1` one degree
apart, which is useless for a ten-digit project.

The boxes come from the same request that lists the images: the index stores a
compact `[x, y, w, h, tag]` per region, which costs about 84 KB across 2232
images and saves one request per thumbnail. The overlay is an SVG in the
image's own coordinate system, so it needs no scaling maths and stays correct
at any thumbnail size and aspect ratio.

The **Show ROI** toggle above the grid turns the outlines off for anyone
reviewing image quality rather than labels; the choice is remembered per
browser.

### Annotating quickly

| Key | Action |
| --- | --- |
| `D` / `V` | Draw box / select |
| `1`-`9`, `0` | Pick the first ten classes |
| `A` / `S` (or arrows) | Previous / next image |
| `Del` | Delete the selected box |
| `Esc` | Cancel the current action |
| `Ctrl+S` | Save |

Navigation saves first and stays put if the save fails, so work is never lost
to a dropped connection.

### Stopping a run

`Stop` asks the worker to finish the epoch it is on. The worker sees the
request between epochs and exits through its normal completion path, so the
exports still run and the run is recorded in `history.json`. Only if it has
not exited after 90 seconds is it terminated, and the weights written so far
are kept either way.

### Train/val split

The split is grouped by source image: colour-augmented copies always land on the
same side as the original they came from, and augmented images are kept out of
validation entirely. Without this, validation would be scoring the model on
near-duplicates of its own training data and every run would look excellent.

## Testing

Everything, in one command:

```bash
python backend/tests/run_all.py            # every backend suite (~3 min)
python backend/tests/run_all.py --full     # also trains a model and detects with it
python backend/tests/run_all.py --list     # what each suite covers
```

Individually, if you want to run just one:

```bash
python backend/tests/test_api.py                 # API surface, dataset build, auth
python backend/tests/test_edge_cases.py          # odd images, unicode names, bad input
python backend/tests/test_concurrency.py         # simultaneous writers and readers
python backend/tests/test_regressions.py         # one check per bug that was fixed
python backend/tests/test_missing_files.py       # a project whose images are gone
python backend/tests/test_autolabel.py           # trains a model, then labels with it
python backend/tests/test_augment.py             # the colour filters, and the boxes
                                                 #   they must not move
python backend/tests/test_train_end_to_end.py    # a real run, then detection with it
python backend/tests/test_video_webcam.py        # one frame at a time, and a video
python backend/tests/bench_scale.py              # timings against a 2232-image project
```

The frontend:

```bash
cd frontend && npm test                    # every view mounts and behaves
python frontend/tests/check-css-vars.py    # no undefined CSS custom properties
```

Each script prints PASS/FAIL per check and exits non-zero on failure. They
create their own temporary projects directory and clean it up, so none of them
touch `data/projects`.

`test_train_end_to_end.py` is the one that proves the thing that was broken:
it builds a dataset, runs a real training subprocess to completion, and then
asks the resulting weights to find the object in an image the model never saw,
checking the box lands on it. A pipeline that trains on wrong labels completes,
writes weights, and fails exactly that last check.

### In a browser, against a running server

```bash
python backend/app.py                                       # in one terminal
node frontend/tests/browser-walkthrough.mjs http://127.0.0.1:64031
node frontend/tests/browser-live.mjs http://127.0.0.1:64031 clip.webm
```

`browser-live.mjs` drives the camera and video modes, launching Chromium with a
fake camera so it runs unattended. The clip argument is optional; use webm/VP8,
the one container both Chromium and this OpenCV build handle.

This drives the real application in Chromium: every page, every console error,
every failed request, plus the ROI overlays, the annotate screen, the text
cursor rules and the model picker. The component tests stub the network, so
they cannot see a field renamed on one side or a route that 404s — this can.
It needs a build to be present (`cd frontend && npm run build`).

`bench_scale.py` is the one to run if the app feels slow; it reports each
request against a budget.

## Appearance

The interface is dark by default. Every colour, radius, shadow and easing is a
token in `frontend/src/assets/styles/main.css`; views reference tokens rather
than literals, so retuning the palette is a single-file change.

Type is **Space Grotesk** for Latin with **IBM Plex Sans Thai** for Thai, at a
16px base. Both are bundled in `frontend/src/assets/fonts` (632 KB total) and
declared in `fonts.css`, so the app renders identically with no internet and
makes no request to a font CDN. Each face is limited to the code points it
covers, so a screen with no Thai text never downloads the Thai file.

Other pairings ship with it. Switch at runtime or from the console:

```js
document.documentElement.dataset.font = 'plex'   // or inter, jakarta, manrope, grotesk, sarabun, system
```

Cursors follow what an element does, not what it contains: headings, labels and
figures show an arrow and are not selectable, while inputs, log lines, file
paths and error messages keep a text cursor and stay copyable. The browser's
default of an I-beam over every piece of text made static headings look
editable.

The UI carries no emoji. Icons come from `components/Icon.vue`, a single
lookup of 52 SVG paths — an unknown name renders a dashed placeholder rather
than nothing, which is how twenty missing icons had been showing as blank
boxes.

`python frontend/tests/check-css-vars.py` fails on any `var(--x)` that nothing
defines — an undefined custom property silently invalidates its whole
declaration, which is how a gradient can vanish with no error anywhere.

## Notes and known limits

- **Pretrained weights download on the first run.** `yolo11s.pt` and friends are
  fetched from the ultralytics release assets, and Faster R-CNN pulls its
  backbone from the torchvision hub. The first training run needs a network
  connection; later ones do not.
- **`.blob` export needs `pip install blobconverter`,** and the conversion is
  performed by Intel's hosted service, so it needs network access. `.blob` files
  run only on Luxonis OAK hardware and cannot be tested in the browser — test
  the `.onnx` or `.pt` export instead.
- **CPU training is very slow.** `doctor.py` tells you whether CUDA was found,
  and distinguishes "no GPU" from "GPU present but PyTorch is a CPU-only
  build". On CPU, prefer `yolo11n` at `imgsz=320` for a first sanity run.
- **Pick a batch size that fits your VRAM.** On a 4 GB card, `yolo11n`/`yolo11s`
  at `batch_size` 4-8 and `imgsz` 640 is a safe starting point; Faster R-CNN
  needs `batch_size` 2. If a run dies with no error in the log, it almost
  always ran out of GPU memory — lower the batch size and retry.
- **Faster R-CNN exports `.pth`, not `.pt`.** A `torch.save(model)` pickle can
  only be loaded with this exact source tree present and cannot be opened by
  the model tester, so it is deliberately not produced. Test the `.pth` or the
  `.onnx`.
- **An augmentation run is capped at 5,000 generated images.** It happens
  inside the request, and every source image is decoded and re-encoded once per
  preset per variant; all 26 presets over 100 images at 3 variants each is
  already 7,800 files. If you hit the cap, pick fewer presets or a subset of
  images.
- **`npm audit` reports issues in `canvas`,** an optional native dependency of
  `fabric@5` used by the annotation canvas. It is never sent to the browser.
  Clearing it means moving to `fabric@6`, whose API differs enough to require
  rewriting the annotator.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `SECRET_KEY` | generated | Session signing. Created on first run and stored in `data/instance/secret_key`. |
| `PROJECTS_ROOT` | `data/projects` | Where datasets live. |
| `BACKEND_PORT` | `64031` | API port. |
| `FRONTEND_PORT` | `64030` | Dev server port. |
| `REQUIRE_AUTH` | `1` | Set `0` to drop the login requirement. |
| `BACKEND_HOST` | `127.0.0.1` | Interface to bind. `0.0.0.0` serves other machines. |
| `FRONTEND_HOST` | `0.0.0.0` | Interface the Vite dev server binds to. |
| `FLASK_DEBUG` | `0` | The Werkzeug debugger runs arbitrary code; leave it off. |
| `ADMIN_USERNAME` / `ADMIN_EMAIL` / `ADMIN_PASSWORD` | `admin` / … / generated | First-run account. |
| `DATABASE_URL` | SQLite in `data/instance` | Account database. |

`PROJECTS_ROOT` also falls back to the legacy `training_module/projects/`
directory if it still exists, so an older checkout keeps working. To move that
data into the current layout:

```bash
python backend/scripts/migrate_layout.py --dry-run
python backend/scripts/migrate_layout.py
```

## API

All routes are under `/api` and return `{"success": true, ...}` or
`{"success": false, "message": "..."}` with a matching HTTP status.

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/auth/login`, `/auth/register`, `/auth/logout` | Session |
| `GET`/`PUT` | `/auth/me`, `/auth/profile` | Account |
| `GET`/`POST` | `/projects` | List / create |
| `GET`/`DELETE` | `/projects/<name>` | Read / delete |
| `GET`/`POST` | `/projects/<name>/images` | List / upload |
| `GET` | `/projects/<name>/images/<file>/raw` | Image bytes |
| `POST` | `/projects/<name>/images/<file>/annotations` | Save boxes |
| `GET` | `/projects/<name>/dataset-summary` | Stats + readiness score |
| `POST` | `/projects/<name>/augment-color` | Generate tone variants |
| `POST`/`GET` | `/projects/<name>/auto-label` | Start / poll model-assisted labelling |
| `POST` | `/projects/<name>/auto-label/cancel` | Stop a labelling pass |
| `POST` | `/projects/<name>/prepare-dataset` | Build the split without training |
| `POST` | `/projects/<name>/export`, `/import-dataset` | Dataset zips |
| `POST` | `/projects/<name>/training/start`, `/stop`, `/reset` | Run control |
| `GET` | `/projects/<name>/training/status`, `/logs`, `/models` | Progress |
| `GET` | `/projects/<name>/training/models/download?path=` | Download weights |
| `POST` | `/models/test` | Run an uploaded model on uploaded images |
| `GET` | `/history`, `/overview` | Cross-project reporting |

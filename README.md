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

### Updating

```powershell
.\start.ps1 -Update        # Windows
```

```bash
./start.sh --update        # Linux
```

One command, because doing it by hand is four steps in an order that fails
quietly when you get it wrong. A server still bound to the port keeps serving
the old code, so endpoints the update added answer 404 from the process that is
still running; and the backend serves `frontend/dist` rather than
`frontend/src`, so skipping the rebuild leaves the browser on the previous
interface while the API has already moved on.

`-Update` stops whatever holds the ports, pulls, reinstalls only if
`requirements.txt` or `package.json` actually changed, rebuilds the interface,
and then carries on with whatever else was asked for -- so
`.\start.ps1 -Update -Network` updates and then serves on one port.

Your annotations are tracked on purpose, being work that cannot be regenerated,
so boxes drawn since the last update show up as local changes and would
otherwise stop the pull. They are set aside and put back around it, and if they
cannot be restored automatically the update stops and says where to find them
rather than pressing on.

The first update on an older checkout has to be done by hand, since a launcher
that predates this switch does not have it:

```bash
git pull && cd frontend && npm run build && cd ..
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

## Taking a model to a new site

A model trained at one factory does not work at the next. The lighting is
different, the camera is different, the background is different — the object
has not changed but everything around it has. Retraining from the stock
checkpoint every time is thousands of images and hours per site, and it throws
away everything the previous model learned about the object itself.

A run can now continue from a model this installation already trained. Pick it
on the training screen; the rest is unchanged. What the numbers looked like on
a deliberately shifted "site B" — same twelve images, same epochs, one starting
from stock and one continuing:

```
from stock  ->  mAP50 0.995,  detects on 0/2 validation images,  best 0.00
continued   ->  mAP50 0.995,  detects on 2/2 validation images,  best 0.98
```

Both report a perfect mAP. Only one is usable, and the self-check is what says
which — the same reason it exists.

Two things worth knowing:

- **Keep the `.pt`.** An ONNX or TorchScript export is a compiled graph with no
  trainable head: it runs and cannot be continued. Someone who archives only
  the `.onnx` from a site has no way back to fine-tuning. The `.pt` is the file
  worth keeping.
- **The class lists are compared before the run**, and what it says is recorded
  with it. Continuing onto a different set of classes works — the backbone
  carries over, which is most of the value — but the detection head is rebuilt,
  so what the old model knew about a class the new project does not have is
  discarded.

## Feeding the GPU

Training passed `workers=0`, because ultralytics' dataloader workers used to
deadlock when the trainer runs as a spawned subprocess on Windows, which is how
this application launches it. Measured through that exact path with the
ultralytics in `requirements.txt`, a run at 4 completes normally and produces a
model indistinguishable from one trained at 0 — so it is now chosen rather than
pinned, and exposed on the training screen. On a fast card a single process
decoding JPEGs is the bottleneck and the card idles between batches, which
looks exactly like a slow GPU and gives no hint of the real cause. If a run
never reaches epoch 1, set it back to 0; that is the only failure it can cause.

**Caching the decoded images in memory is deliberately not offered.** It looks
like free speed. On one dataset it reproducibly ruins the model instead.
Trained twice each way on the same images with the same seed and epochs,
differing only in that setting, all four runs reported `mAP50 = 0.995`. Asked
about ten images from the same generator that none of them had seen:

```
cache off  ->  3/10 above 0.25          (both repeats)
cache ram  ->  0/10, every score 0.01   (both repeats)
```

It is not universal — on an easier set the cached run detected perfectly well.
What makes it worth refusing anyway is that when it goes wrong it goes wrong
silently, and the number that should warn you reads 0.995.

That number misleads for a reason worth knowing: **mAP measures ranking, not
usable confidence.** It integrates over all confidence levels, so a model that
answers 0.01 everywhere while ranking the right boxes first scores near 1.0 and
detects nothing at any threshold a person would use. Which is why every run now
also checks itself at a practical threshold — see below.

Passing `cache` explicitly still reaches it, for anyone re-measuring on a newer
ultralytics.

## Every run checks whether its own weights detect anything

The application shipped once with training that completed, reported a falling
loss, and produced a model that found nothing. The caching failure above is the
same shape. In both cases every number the tool had said the run was fine.

So a finished run now loads the weights it just wrote, points them at images
from its own validation split, and records what came back:

```
Self-check: found 5 object(s) on 5/5 validation images (best 0.98).
```

or, for the cached run above, alongside its `mAP50 = 0.995`:

```
Self-check: found NOTHING on 4 validation images at 0.25. These weights will
not detect anything as they are, whatever the metrics above say.
```

It proves nothing about accuracy — a model that detects is not necessarily a
good one — but a model that detects nothing on the very images it was scored
against is not usable, whatever the numbers say. The result is kept with the
run and in the history, and `backend/tests/test_model_is_usable.py` checks both
directions in about 90 seconds, which is short enough to run every time rather
than only under `--full`. Both trainers do it: YOLO through ultralytics,
Faster R-CNN through torchvision, each supplying its own way of loading the
weights it just wrote.

## Adding images to a project that already has some

While a project is being built, "everything unlabelled" and "the images I just
added" are the same set. The moment it is being extended — a second site, a new
shift, a run of parts the model got wrong — they stop being: a handful of
pictures skipped months ago are still unlabelled, and running a model over
those alongside today's import mixes two decisions into one review.

**Every upload is numbered.** The gallery separates them, each tile says which
one it came from, and a copy made by a filter belongs to the batch its source
came from — so filtering by batch shows a photograph together with everything
made from it.

**Photographs and generated copies filter separately.** The chips are
`ทั้งหมด / รูปถ่ายจริง / จากฟิลเตอร์ / ตีกรอบแล้ว / ยังไม่ตีกรอบ`, and a tile
made by a filter is marked as one. Copies are worth training on and are never
worth re-drawing, so telling them apart at a glance matters.

**Auto-labelling can be pointed at one upload.** With more than one batch
present it defaults to the newest, which after adding images is almost always
what was meant. Everything unlabelled is still available as a choice.

While building this, saving a box turned out to erase the batch number the
upload had written, because the save rewrote the annotation from scratch. The
fields that describe where an image came from — its batch, when it arrived,
its original filename, whether a model drew it — now survive a save. Only the
boxes and the dimensions are the save's to own.

## Finding out whether auto-labelling would work

Auto-labelling only touches images that have no boxes yet, which is right when
it is doing the work and wrong when the question is whether it can. On a
project that is already fully annotated it had nothing to do and refused,
and the only way to see the model's output was to let it overwrite work drawn
by hand.

**Try it first** runs the model over a few images and reports what it would
draw, writing nothing. It prefers images that already have boxes, because those
come with the answer to compare against:

```
Image                     Drawn by hand   Model found   Best score
20260827_...0000.jpg            2              0           -
20260827_...0001.jpg            3              0           -

Found nothing on any of 4 images above 0.25. Either this model has not
learned these objects, or the threshold is too high.
```

There is also an explicit **Also re-label images that already have boxes**,
which does replace them — the destructive option, next to the one that answers
the same question for free.

## Trying a model on one image

The annotate screen has a **Detect** button (or `F`) beside a model chooser. It
runs the chosen model over the image on screen and drops what it found in as
ordinary editable boxes. Nothing is saved.

This is deliberately not auto-labelling. That is a bulk background pass over
everything unannotated, and turning a model loose on a thousand pictures is a
poor way to find out whether it is worth trusting. One image, on demand, while
somebody is looking at it, answers that question in a few seconds — and the
same button is the fastest way to label an image once the answer is yes.

## How well each class is doing

The project page listed how many boxes each class had, which says how much work
went in and nothing about whether any of it worked. Each class now also carries
the figure from the newest finished run:

```
block: 347  92%      circle: 128  71%      8: 96  34%
```

A single overall mAP does not help here. On a ten-class detector, "0.72" and
"everything is fine except 8" call for completely different next actions, and
only the second says what to go and photograph.

A class the validation split never contained shows no figure at all rather than
0%. Not measured is not the same as scored zero, and showing zero would send
someone to fix a model that was never tested on it.

### best.pt is not always the best weights

ultralytics picks `best.pt` by fitness, which is mostly mAP50-95 — a ranking
measure. An epoch whose precision has collapsed to 0.008 can still rank well
and win. Seen on an ordinary 24-image run:

```
Epoch 21:  precision 1.000   recall 0.948   mAP50 0.995
Epoch 23:  precision 0.008   recall 1.000   mAP50 0.995   <- chosen as "best"

best.pt  ->  detects on 0/5 validation images
last.pt  ->  detects on 2/5, best score 0.40
```

`best.pt` is not silently redefined — quietly handing back different weights
than the ones named is its own kind of dishonesty — but when it fails the
self-check the last-epoch weights are checked too, and if those work the run
says so and names them.

## Annotating faster

**Copy the boxes from the previous image** — the button, or `C`. A camera
bolted above a line photographs the same object in nearly the same place every
time, so on those projects almost every box is a small nudge away from the one
before it; redrawing each from scratch is the bulk of the work and none of the
value. The boxes are appended rather than replacing what is there, scaled if
the two images differ in size, and left unsaved, so a wrong copy costs nothing.

**Pre-label with a model from another project.** Auto-labelling used to require
a completed run in the same project, which is a problem exactly when it would
help most: a brand-new project has no model, and the reason to pre-label is
that there is nothing labelled yet. Any model this installation has trained can
now be chosen. One trained on a similar job usually gets the boxes close enough
to be worth correcting, and correcting a drawn box is far faster than drawing
one.

A pass that labels nothing now says why — whether the model found nothing above
the threshold, or there was nothing to do — rather than reporting a count of
zero and leaving you to guess.

## The advice before you train

The readiness figure is accompanied by advice that names things rather than
describing them. "The smallest class has 2 images" leaves someone with nine
classes no idea what to go and photograph; the project page now says which
classes are short and by how much, which of them to start with, and what the
imbalance is between the largest and the smallest.

It also sizes the run to the data. A large model on a small set memorises it
rather than learning from it, and few images benefit from more passes, so a
project under 60 annotated images is told to use yolo11n or yolo11s and to
raise the epochs rather than accept the default.

## Testing a model this application did not train

Any model built by other tooling follows its own conventions, and ultralytics
raises from deep inside its own backend when it meets one it cannot wrap: an
IndexError on an empty input list, a protobuf parse failure, an output shape
its decoder does not know. Every one of those reached the browser as

```
Internal server error. Check the server log for details.
```

which is no use to anyone who is not holding the server log. Loading is now
caught wherever it happens and reported with the file's real name — the one
that was chosen, not the temporary name it was saved under — and the underlying
error:

```
"model.onnx" could not be loaded. InvalidProtobuf: Load model failed:
Protobuf parsing failed. The file does not parse as ONNX at all — it may be
truncated or may not be the file you meant.
```

A file that cannot be read as an image, or is too small for a detector to
letterbox, is reported the same way rather than crashing the request; and one
bad image among thirty-six no longer loses the other thirty-five. The results
page lists what was left out and why.

### When the metadata is the only thing wrong

Ultralytics reads the metadata an exporter left in the file and trusts it. A
model built elsewhere carries whatever that tooling wrote, and one export
failed with

```
TypeError: empty(): argument 'size' failed to unpack the object at pos 2
           with error "type must be tuple of ints, but got str"
```

which is a metadata value going straight into torch. Nothing was wrong with
the model — the note attached to it was the wrong shape. A detector nobody can
run because of a string in a metadata field is a poor outcome, so an ONNX that
ultralytics refuses is now driven directly with onnxruntime, working from the
graph and ignoring the metadata entirely.

It understands the layout every YOLOv8/v9/v10/v11 export shares: an image in as
`[1, 3, H, W]`, one tensor out of `[1, 4 + classes, anchors]`. Checked against
ultralytics on a model both can load — the same detections at every threshold
tried, with boxes agreeing to within a pixel, and suppression done per class as
ultralytics does it. Anything genuinely different in shape is reported rather
than guessed at.

The one thing lost on this path is the class names, which live in the metadata
being ignored. Type them into **Label Names** and they are used; leave it empty
and the classes come back numbered.

## A test run as a spreadsheet

The results grid answers "did it work" while you are looking at it. Handing
that answer to somebody else, or filing it beside a batch of parts, needs a
file — and a file of numbers with no pictures is not evidence of anything.

**Export to Excel** writes an `.xlsx` with the annotated images in the sheet,
so a row can be read without opening anything else. Two sheets, because two
questions get asked of it:

| # | Image | File | Reading | Objects | Confidence | Lowest | Notes |
|---|-------|------|---------|---------|------------|--------|-------|
| 1 | *(picture)* | good.jpg | 250 | 3 | 91% | 88% | |
| 2 | *(picture)* | unsure.jpg | 8 | 1 | 31% | 31% | Every detection is below 0.50: 8 at 0.31. Check by eye. |
| 3 | *(picture)* | empty.jpg | | 0 | | | Nothing found. Either the object is absent or the model missed it. |

The second sheet is one row per detection, for sorting and filtering.

Three cases are kept apart rather than collapsed into a number, because they
call for different actions: a clean read, something found but not confidently,
and nothing found at all. Rows are coloured to match, and anything under 0.50
is flagged in both sheets. That figure is not the score threshold — the
threshold already decided what to report at all; this is the line under which a
reading should be checked by a person before it is acted on.

**Clicking a result opens it.** The grid says how the run went overall; the
panel says exactly what the model claimed about one picture, which is the
question asked of anything that looks wrong.

## Getting things out, and getting rid of them

**A video analysis downloads as CSV.** The page draws the boxes over the clip,
which answers "did it work". It does not answer "what did it read at 12.4
seconds", and copying that out of a browser is not a thing anyone should have
to do. One row per detection, carrying the reading of its frame as well as the
individual box, so a spreadsheet can be filtered either way. A frame where
nothing was found still gets a row: the difference between "not looked at" and
"looked, saw nothing" is worth keeping.

**Generated copies delete in one go.** Filters write a hundred images in
seconds and removing them was one request per file. The project page offers to
delete every filtered copy at once, leaving the photographs you annotated
alone — which is the case that actually comes up, when a preset turns out to
suit the data badly.

## Results come back in reading order

A detector returns boxes in the order it found them, which for most models is
by confidence. That is the wrong order for anything being read rather than
counted: a display showing **250** comes back as 0, 2, 5 or 5, 0, 2 depending
on which digit the model happened to be surest about, and the number is gone.

Every path that returns detections -- still images, video frames, the camera --
now sorts them the way a person reads. Boxes are grouped into lines first, by
whether they overlap vertically, then read left to right within each line;
sorting purely by x would be right for one line and wrong the moment there are
two, putting a digit low on the left before one high on the right.

Each detection carries its `line` and `position`, and each result carries a
`reading`: the labels as one string, with lines separated by a newline. For a
project whose classes are the characters on a display, that string is the
answer, and the list of boxes is the working.

The tolerance for "same line" is half the shorter box's height, which is
forgiving enough for digits that sit slightly high or low and strict enough to
keep two rows apart.

## Augmentation: what happens automatically, and what you ask for

Two different things go by that name here, and keeping them apart decides what
is worth doing.

**Every epoch, automatically.** The trainer varies each image as it reads it:
hue, saturation, brightness, position, scale, and it erases a patch. This costs
no disk and gives a fresh variant each epoch, which generalises better than any
fixed set of files. It was already happening — nothing in this application
passed ultralytics any settings, so it used its own defaults.

One of those defaults was wrong for this tool's usual job. `fliplr=0.5` mirrors
half of every epoch. Mirror a **2** and you have something that is not a 2,
while the label riding along with it still says it is. A project whose classes
are digits or letters was therefore being taught something false a share of the
time. The run now decides from the project's own class list: mirroring is off
when every class reads as a character or a number, on otherwise, and the
training screen says which it chose and why. You can override it.

**Before training, on request.** The screen offers to write filtered copies of
the annotated images, reusing their boxes. Fifteen annotated photographs and
six presets gives ninety images. Worth doing when you have few images; the
count is shown before you start, because each copy makes every epoch longer.

Only the presets that add something are offered. Brightness, colour and
contrast are already applied fresh every epoch at no cost, so writing them out
as files buys nothing and slows each epoch down. What is offered is the
structural work the trainer does not do: CLAHE, adaptive thresholds,
morphological top-hat and black-hat, edge maps, unsharp masking — the
transformations that matter for characters stamped or engraved into metal.

A generated image whose filter hid the annotated object is dropped rather than
trained on, and copies of an image that landed in validation are held out of
both splits: training on them would leak the validation image, and validating
on them would flatter the score. Both are reported rather than left to be
noticed as a shortfall in the image count.

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
python backend/tests/test_train_augment.py       # what a run does to each image
python backend/tests/test_reading_order.py       # detections in reading order
python backend/tests/test_model_is_usable.py     # trained weights actually detect
python backend/tests/test_bulk_and_export.py     # bulk delete, and CSV export
python backend/tests/test_import_batches.py      # one upload from another
python backend/tests/test_report_export.py       # the spreadsheet, pictures and all
python backend/tests/test_fine_tune.py           # continuing from a trained model
python backend/tests/test_detect_and_accuracy.py # detect on one image, class accuracy
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

The interface is in English throughout. Some screens carried Thai for a while;
none do now, and three files also had their comment separators repaired, where
box-drawing characters had been mangled into Thai code points by a round trip
through the wrong encoding.


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

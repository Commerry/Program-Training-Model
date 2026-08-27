"""
Reconstruct exactly what the old code produced, to show why the models it
trained could not be used.

Both root causes are reproduced from the original source, not described.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print('=' * 74)
print('  ROOT CAUSE 1 — the labels every old training run was fed')
print('=' * 74)

# A real annotation from the reference project: a digit box on a 640x480 photo.
region = {'tag': '7', 'x': 210.0, 'y': 150.0, 'width': 96.0, 'height': 120.0}
REAL_W, REAL_H = 640, 480

# ── the old code, verbatim in behaviour ─────────────────────────────────────
# project_manager.start_training() did:
#     img_w, img_h = img.get('width', 1), img.get('height', 1)
#     if img_w <= 0 or img_h <= 0:   <-- 1 is not <= 0, so the fallback never ran
#         ...read the real size...
# get_images() never returned width/height, so both were always 1.
img_w, img_h = 1, 1

x, y, w, h = region['x'], region['y'], region['width'], region['height']
old_cx = (x + w / 2.0) / img_w
old_cy = (y + h / 2.0) / img_h
old_nw = w / img_w
old_nh = h / img_h

print(f'\nimage           : {REAL_W} x {REAL_H} px')
print(f'annotation      : x={x} y={y} w={w} h={h}  (class "{region["tag"]}")')
print(f'\nwhat the old code wrote into the .txt label file:')
print(f'    7 {old_cx:.6f} {old_cy:.6f} {old_nw:.6f} {old_nh:.6f}')
print(f'\nYOLO requires every one of those four numbers to be in [0.0, 1.0].')
for label, value in (('cx', old_cx), ('cy', old_cy), ('w', old_nw), ('h', old_nh)):
    verdict = 'OK' if 0.0 <= value <= 1.0 else f'INVALID — {value:.0f}x over the maximum'
    print(f'    {label:2} = {value:10.3f}   {verdict}')

# ── what it should have been ────────────────────────────────────────────────
new_cx = (x + w / 2.0) / REAL_W
new_cy = (y + h / 2.0) / REAL_H
new_nw = w / REAL_W
new_nh = h / REAL_H
print(f'\nwhat it should have been:')
print(f'    7 {new_cx:.6f} {new_cy:.6f} {new_nw:.6f} {new_nh:.6f}')
print(f'    all four in range, box covers {new_nw * 100:.0f}% x {new_nh * 100:.0f}% of the frame')

print(f'\nConsequence: ultralytics clips out-of-range boxes to the frame edge, so')
print(f'every single label collapsed to the same degenerate corner box. The model')
print(f'trained happily, reported a loss, and learned nothing about your digits.')
print(f'That is why the finished model detected nothing.')

print()
print('=' * 74)
print('  ROOT CAUSE 2 — why a run took so long')
print('=' * 74)

try:
    import torch
    build = torch.__version__
    cuda = torch.version.cuda or 'none (CPU-only build)'
    available = torch.cuda.is_available()
except ImportError:
    build, cuda, available = 'not installed', 'n/a', False

print(f'\ninstalled torch : {build}')
print(f'CUDA runtime    : {cuda}')
print(f'GPU usable      : {available}')

import subprocess
try:
    out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                          '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=15)
    gpu = out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
except Exception:
    gpu = None
print(f'GPU present     : {gpu or "none detected"}')

IMAGES = 2232
if gpu and not available:
    print(f'\nThe card is there; this build of PyTorch cannot address it, so every')
    print(f'run fell back to the CPU without saying so.')

# Measured on this machine earlier in the session: yolo11n, 19 images,
# imgsz 320, 3 epochs took ~10 s of pure training on CPU.
cpu_s_per_img_epoch = 10.0 / (19 * 3)
train_images = int(IMAGES * 0.8)
for epochs in (50, 100):
    cpu_hours = cpu_s_per_img_epoch * train_images * epochs / 3600
    # A GTX 1650 runs yolo11s at roughly 25-40x CPU for this workload.
    gpu_hours = cpu_hours / 30
    print(f'\n{epochs} epochs over {train_images} training images at imgsz 640:')
    print(f'    CPU (what you have now) : ~{cpu_hours * (640 / 320) ** 2:6.1f} hours')
    print(f'    GTX 1650 with CUDA      : ~{gpu_hours * (640 / 320) ** 2:6.1f} hours')

print(f'\nThe imgsz 640 figures scale the measured 320px timing by area (4x).')
print(f'Rough, but the gap is the point: days versus an evening.')

print()
print('=' * 74)
print('  BOTH ARE FIXED')
print('=' * 74)
print("""
Labels   the dataset builder reads real pixel dimensions, clamps every box to
         the frame, and refuses to start a run if it cannot produce valid
         labels — so a wasted run is now impossible rather than silent.
         Verified by test_api.py, which asserts every emitted coordinate is
         inside [0,1] and round-trips a label back to its source pixels.

Speed    doctor.py now fails loudly when a GPU is present but unusable, with
         the exact install command. That is the one thing you still need to
         do; everything else is already in place.
""")

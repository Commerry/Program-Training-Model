#!/usr/bin/env python
"""
Environment and dataset check.

Run before the first training session:

    python backend/scripts/doctor.py               # environment only
    python backend/scripts/doctor.py <project>     # environment + that project
"""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECTS_ROOT  # noqa: E402

OK, WARN, FAIL = '  [ok]  ', '  [warn]', '  [FAIL]'

REQUIRED = [
    ('flask', 'flask'),
    ('flask_cors', 'flask-cors'),
    ('flask_sqlalchemy', 'flask-sqlalchemy'),
    ('flask_login', 'flask-login'),
    ('flask_bcrypt', 'flask-bcrypt'),
    ('cv2', 'opencv-python'),
    ('numpy', 'numpy'),
    ('PIL', 'pillow'),
    ('yaml', 'pyyaml'),
    ('psutil', 'psutil'),
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('ultralytics', 'ultralytics'),
]
OPTIONAL = [
    ('onnx', 'onnx', 'ONNX export'),
    ('onnxruntime', 'onnxruntime', 'running ONNX models'),
    ('blobconverter', 'blobconverter', '.blob export for Luxonis OAK devices'),
]


def check_packages():
    print('Python packages')
    missing = []
    for module, package in REQUIRED:
        try:
            importlib.import_module(module)
            print(f'{OK} {package}')
        except ImportError:
            print(f'{FAIL} {package} is missing')
            missing.append(package)

    for module, package, purpose in OPTIONAL:
        try:
            importlib.import_module(module)
            print(f'{OK} {package}')
        except ImportError:
            print(f'{WARN} {package} not installed — needed for {purpose}')

    if missing:
        print(f'\n  Install with: pip install {" ".join(missing)}')
    return not missing


def _nvidia_gpu_present():
    """
    Detect an NVIDIA GPU without relying on torch.

    A CPU-only torch build reports no CUDA even on a machine with a perfectly
    good GPU, and that is by far the most common reason training is
    unexpectedly slow. Asking the driver directly tells the two cases apart.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    line = (result.stdout or '').strip().splitlines()
    return line[0].strip() if line else None


def _vram_advice(total_gb):
    if total_gb < 5:
        return ('With under 5 GB of VRAM use yolo11n or yolo11s with '
                'batch_size 4-8 at imgsz 640. Faster R-CNN needs batch_size 2.')
    if total_gb < 9:
        return 'Around 8 GB of VRAM: yolo11s/m at batch_size 8-16, imgsz 640.'
    return 'Plenty of VRAM: batch_size 16-32 at imgsz 640 is comfortable.'


def check_torch():
    print('\nCompute device')
    try:
        import torch
    except ImportError:
        print(f'{FAIL} PyTorch is not installed')
        return False

    print(f'{OK} torch {torch.__version__}')

    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(index).total_memory / 1e9
        print(f'{OK} CUDA {torch.version.cuda} — {torch.cuda.get_device_name(index)} '
              f'({total:.1f} GB)')
        print(f'       {_vram_advice(total)}')
        return True

    gpu = _nvidia_gpu_present()
    if gpu:
        # The important case: the hardware is there but the installed wheel
        # cannot use it. Training would silently fall back to the CPU and take
        # days instead of minutes.
        print(f'{FAIL} An NVIDIA GPU is present ({gpu}) but this PyTorch build '
              'cannot use it.')
        print(f'       Installed build: torch {torch.__version__} '
              f'(CUDA runtime: {torch.version.cuda or "none — CPU-only build"})')
        print('       Training will fall back to the CPU and be 20-50x slower.')
        print('')
        print('       Install a CUDA build instead:')
        print('         pip uninstall -y torch torchvision')
        print('         pip install torch torchvision '
              '--index-url https://download.pytorch.org/whl/cu124')
        print('')
        print('       Then re-run this script; it should report CUDA above.')
        return False

    print(f'{WARN} No CUDA device found. Training will run on the CPU, which is '
          '20-50x slower.')
    print('       For a first check, use yolo11n at imgsz 320 with few epochs.')
    return True


def check_disk():
    print('\nStorage')
    try:
        usage = shutil.disk_usage(PROJECTS_ROOT if PROJECTS_ROOT.exists()
                                  else PROJECTS_ROOT.parent)
    except OSError as exc:
        print(f'{WARN} Could not read disk usage: {exc}')
        return
    free_gb = usage.free / 1e9
    marker = OK if free_gb >= 10 else WARN
    print(f'{marker} {free_gb:.1f} GB free at {PROJECTS_ROOT}')
    if free_gb < 10:
        print('       A training run needs a few GB for the dataset copy, '
              'checkpoints and exports.')


def check_project(name):
    from services import projects
    from services.projects import ProjectError

    print(f'\nProject "{name}"')
    try:
        summary = projects.dataset_summary(name)
    except ProjectError as exc:
        print(f'{FAIL} {exc.message}')
        return False

    print(f'{OK} {summary["total_images"]} images, '
          f'{summary["annotated_images"]} annotated, '
          f'{summary["total_boxes"]} boxes, '
          f'{summary["num_classes"]} classes')
    print(f'       Readiness: {summary["readiness_score"]}%')
    for warning in summary['warnings']:
        print(f'{WARN} {warning}')

    print('\n  Building the train/val split...')
    from services import dataset as dataset_service
    try:
        report = dataset_service.build_yolo_dataset(name)
    except ProjectError as exc:
        print(f'{FAIL} {exc.message}')
        return False

    print(f'{OK} {report["train_images"]} train / {report["val_images"]} val images')
    print(f'{OK} {report["train_boxes"]} train / {report["val_boxes"]} val boxes')
    for cls in report['empty_classes']:
        print(f'{WARN} class "{cls}" has no boxes in the prepared dataset')
    if report['skipped_count']:
        print(f'{WARN} {report["skipped_count"]} images skipped:')
        for entry in report['skipped'][:10]:
            print(f'         {entry["filename"]}: {entry["reason"]}')
    print(f'{OK} data.yaml: {report["data_yaml"]}')
    return True


def main():
    print(f'Projects directory: {PROJECTS_ROOT}\n')
    healthy = check_packages()
    # A GPU that PyTorch cannot use is a hard failure: training would appear to
    # work while taking days instead of minutes.
    healthy = check_torch() and healthy
    check_disk()

    if len(sys.argv) > 1:
        healthy = check_project(sys.argv[1]) and healthy
    else:
        from services import projects
        names = [p['name'] for p in projects.list_projects()]
        print(f'\nProjects found: {", ".join(names) if names else "(none)"}')
        if names:
            print('Re-run with a project name to check its dataset, e.g.:')
            print(f'    python backend/scripts/doctor.py {names[0]}')

    print('\nAll required checks passed.' if healthy else
          '\nFix the failures above before training.')
    return 0 if healthy else 1


if __name__ == '__main__':
    raise SystemExit(main())

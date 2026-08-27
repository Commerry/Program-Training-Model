#!/usr/bin/env python
"""
Move projects from the old training_module/projects/ folder to data/projects/.

Only needed once, and only if an installation still has data in the old place.
The backend reads either location, so this is about tidiness, not correctness.

    python backend/scripts/migrate_layout.py --dry-run
    python backend/scripts/migrate_layout.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import REPO_ROOT  # noqa: E402

LEGACY = REPO_ROOT / 'training_module' / 'projects'
TARGET = REPO_ROOT / 'data' / 'projects'


def main():
    dry_run = '--dry-run' in sys.argv

    if not LEGACY.exists():
        print(f'Nothing to migrate: {LEGACY} does not exist.')
        return 0

    projects = [p for p in LEGACY.iterdir()
                if p.is_dir() and (p / 'project.json').exists()]
    if not projects:
        print(f'No projects found under {LEGACY}.')
        return 0

    print(f'Source: {LEGACY}')
    print(f'Target: {TARGET}\n')

    conflicts = [p.name for p in projects if (TARGET / p.name).exists()]
    if conflicts:
        print('These projects already exist at the target and will NOT be touched:')
        for name in conflicts:
            print(f'  - {name}')
        print()

    movable = [p for p in projects if p.name not in conflicts]
    for project in movable:
        size = sum(f.stat().st_size for f in project.rglob('*') if f.is_file())
        print(f'  {project.name}  ({size / 1e6:.1f} MB)')

    if dry_run:
        print(f'\nDry run: {len(movable)} project(s) would be moved.')
        return 0

    TARGET.mkdir(parents=True, exist_ok=True)
    for project in movable:
        print(f'Moving {project.name}...')
        shutil.move(str(project), str(TARGET / project.name))

    # The old index file is no longer used — projects are discovered by
    # scanning the directory — so it is left behind rather than moved.
    leftovers = [p.name for p in LEGACY.iterdir()] if LEGACY.exists() else []
    print(f'\nMoved {len(movable)} project(s).')
    if leftovers:
        print(f'Left in place at {LEGACY}: {", ".join(leftovers)}')
        print('Delete that folder once you have confirmed everything works.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

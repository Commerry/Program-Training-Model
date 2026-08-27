#!/usr/bin/env python
"""
Package a project's images and annotations into one zip, for moving it between
machines.

Run this on the machine that HAS the dataset:

    python backend/scripts/pack_project.py test01
    python backend/scripts/pack_project.py test01 --out D:\\test01.zip

It works against the old layout (training_module/projects) and the current one
(data/projects), so it can be dropped into an older checkout unchanged.

Copy the resulting zip across and unpack it with:

    python backend/scripts/pack_project.py --restore D:\\test01.zip

Why this exists: cloning the repo or using GitHub's "Download ZIP" does not
carry the dataset. .gitignore excludes the images on purpose, and it used to
exclude the annotations too, so a transfer that way arrives with a project.json
claiming thousands of annotated images and nothing behind it.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Both layouts, newest first.
CANDIDATE_ROOTS = [
    REPO / 'data' / 'projects',
    REPO / 'training_module' / 'projects',
]

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def find_project(name):
    for root in CANDIDATE_ROOTS:
        candidate = root / name
        if (candidate / 'project.json').exists():
            return candidate
    searched = '\n'.join(f'    {r / name}' for r in CANDIDATE_ROOTS)
    raise SystemExit(f'Project "{name}" not found. Looked in:\n{searched}')


def pack(name, out_path=None):
    project = find_project(name)
    images = sorted(p for p in (project / 'images').glob('*')
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    annotations = sorted((project / 'annotations').glob('*.json'))

    if not images and not annotations:
        raise SystemExit(
            f'"{name}" has no images and no annotations at {project}.\n'
            'There is nothing to pack — this is the empty side of the transfer.'
        )

    out = Path(out_path) if out_path else REPO / f'{name}_dataset.zip'
    print(f'project     : {project}')
    print(f'images      : {len(images)}')
    print(f'annotations : {len(annotations)}')
    print(f'writing     : {out}')

    total = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(project / 'project.json', f'{name}/project.json')
        for index, image in enumerate(images, 1):
            zf.write(image, f'{name}/images/{image.name}')
            total += image.stat().st_size
            if index % 250 == 0:
                print(f'  {index}/{len(images)} images...')
        for annotation in annotations:
            zf.write(annotation, f'{name}/annotations/{annotation.name}')

    size = out.stat().st_size
    print(f'\ndone: {size / 1e6:.1f} MB '
          f'(from {total / 1e6:.1f} MB of images)')
    print(f'\nCopy {out.name} to the other machine, then run:')
    print(f'    python backend/scripts/pack_project.py --restore "{out.name}"')
    return out


def restore(zip_path):
    archive = Path(zip_path)
    if not archive.is_file():
        raise SystemExit(f'No such file: {archive}')

    target_root = CANDIDATE_ROOTS[0]
    target_root.mkdir(parents=True, exist_ok=True)

    images = annotations = 0
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        project_name = names[0].split('/')[0] if names else None
        if not project_name:
            raise SystemExit('The archive is empty.')

        destination = target_root / project_name
        print(f'restoring "{project_name}" into {destination}')

        for member in names:
            if member.endswith('/'):
                continue
            relative = Path(member)
            out_file = target_root / relative
            # Guard against a crafted archive escaping the projects directory.
            if not out_file.resolve().is_relative_to(target_root.resolve()):
                print(f'  skipped (path escapes the target): {member}')
                continue
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(out_file, 'wb') as dst:
                dst.write(src.read())
            if '/images/' in member:
                images += 1
            elif '/annotations/' in member:
                annotations += 1

    print(f'\nrestored {images} images and {annotations} annotations')
    print('Open the project in the app and press "Check again" — or just '
          'reload the page.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project', nargs='?', help='project name to pack')
    parser.add_argument('--out', help='where to write the zip')
    parser.add_argument('--restore', metavar='ZIP',
                        help='unpack a zip made by this script')
    parser.add_argument('--list', action='store_true',
                        help='show the projects this machine can see')
    args = parser.parse_args()

    if args.list or (not args.project and not args.restore):
        print('Projects visible from this checkout:\n')
        seen = False
        for root in CANDIDATE_ROOTS:
            if not root.is_dir():
                continue
            for entry in sorted(root.iterdir()):
                meta = entry / 'project.json'
                if not meta.is_file():
                    continue
                seen = True
                images = len([p for p in (entry / 'images').glob('*')
                              if p.is_file()]) if (entry / 'images').is_dir() else 0
                annotations = len(list((entry / 'annotations').glob('*.json'))) \
                    if (entry / 'annotations').is_dir() else 0
                try:
                    recorded = json.loads(meta.read_text(encoding='utf-8-sig')) \
                        .get('total_images', 0)
                except Exception:  # noqa: BLE001
                    recorded = '?'
                flag = '' if images else '   <-- no image files present'
                print(f'  {entry.name:24} on disk: {images:5} images, '
                      f'{annotations:5} annotations   recorded: {recorded}{flag}')
        if not seen:
            print('  (none)')
        print(f'\nLooked in: {", ".join(str(r) for r in CANDIDATE_ROOTS)}')
        return 0

    if args.restore:
        restore(args.restore)
        return 0

    pack(args.project, args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())

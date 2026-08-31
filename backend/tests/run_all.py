#!/usr/bin/env python
"""
Run every check in one command, on Windows or Linux.

    python backend/tests/run_all.py            everything except the slow ones
    python backend/tests/run_all.py --full     including a real training run
    python backend/tests/run_all.py --list     just show what would run

The suites are separate processes on purpose: each one builds its own temporary
projects directory and its own database, so a crash in one cannot leave state
behind that quietly changes the result of the next.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
BACKEND = TESTS.parent
REPO = BACKEND.parent

# (name, path, slow, what it covers)
SUITES = [
    ('api', TESTS / 'test_api.py', False,
     'every endpoint, path safety, dataset build'),
    ('edge cases', TESTS / 'test_edge_cases.py', False,
     'odd project names, reserved filenames, unicode'),
    ('concurrency', TESTS / 'test_concurrency.py', False,
     'simultaneous writes and locked files'),
    ('regressions', TESTS / 'test_regressions.py', False,
     'defects that shipped once already'),
    ('missing files', TESTS / 'test_missing_files.py', False,
     'a project whose images are gone'),
    ('auto-label', TESTS / 'test_autolabel.py', False,
     'model-assisted labelling'),
    ('augmentation', TESTS / 'test_augment.py', False,
     'the colour filters and the boxes they must not move'),
    ('end to end', TESTS / 'test_train_end_to_end.py', True,
     'a real training run, then detecting with the model it produced'),
    ('live detection', TESTS / 'test_video_webcam.py', True,
     'one frame at a time for a webcam, and a whole video'),
    ('train augment', TESTS / 'test_train_augment.py', False,
     'what training does to each image, and what it writes first'),
    ('reading order', TESTS / 'test_reading_order.py', False,
     'detections come back in the order a person reads them'),
    ('model is usable', TESTS / 'test_model_is_usable.py', False,
     'a finished run produces weights that actually detect'),
    ('bulk and export', TESTS / 'test_bulk_and_export.py', False,
     'deleting many images at once, and a video analysis as CSV'),
    ('import batches', TESTS / 'test_import_batches.py', False,
     'telling one upload from another, and labelling just the new one'),
    ('report export', TESTS / 'test_report_export.py', False,
     'a model test written out as a spreadsheet, pictures included'),
    ('fine tune', TESTS / 'test_fine_tune.py', True,
     'continuing from a trained model instead of starting over'),
    ('detect and accuracy', TESTS / 'test_detect_and_accuracy.py', True,
     'detecting on one image, and accuracy per class'),
]


def run(name, path, description):
    print(f'\n{"=" * 72}\n{name}  --  {description}\n{"=" * 72}')
    started = time.time()
    result = subprocess.run([sys.executable, str(path)], cwd=str(REPO))
    elapsed = time.time() - started
    ok = result.returncode == 0
    print(f'{"PASSED" if ok else "FAILED"}  ({elapsed:.0f}s)')
    return ok, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--full', action='store_true',
                        help='include the slow suites (a real training run, minutes)')
    parser.add_argument('--list', action='store_true',
                        help='show the suites and exit')
    args = parser.parse_args()

    selected = [s for s in SUITES if args.full or not s[2]]

    if args.list:
        for name, path, slow, description in SUITES:
            mark = ' (slow, needs --full)' if slow else ''
            print(f'  {name:14} {description}{mark}')
        return 0

    missing = [s[0] for s in selected if not s[1].is_file()]
    if missing:
        print(f'Missing test files: {", ".join(missing)}')
        return 2

    results = []
    total = time.time()
    for name, path, _slow, description in selected:
        results.append((name, *run(name, path, description)))

    print(f'\n{"=" * 72}\nSUMMARY\n{"=" * 72}')
    for name, ok, elapsed in results:
        print(f'  {"PASS" if ok else "FAIL"}  {name:16} {elapsed:6.0f}s')

    failed = [name for name, ok, _ in results if not ok]
    print(f'\n{len(results) - len(failed)}/{len(results)} suites passed '
          f'in {time.time() - total:.0f}s')

    if not args.full:
        print('\n(the end-to-end training run was skipped -- add --full to include it)')
    if failed:
        print(f'failed: {", ".join(failed)}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

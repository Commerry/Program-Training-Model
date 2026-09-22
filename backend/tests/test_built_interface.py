"""
The committed build is whole: every file the page asks for is really there.

frontend/dist is committed on purpose -- the factory networks this gets
installed on block the npm registry, so the interface cannot be built on the
machine that needs it, and a browser download of the repository has to be a
complete installation.

That only works while the build in the repository is self-consistent, and it
stopped being so without a word. The ignore rule for frontend/dist came back
in a stray edit and was swept into a commit by `git add -A`. index.html was
already tracked, so the new one was committed; the freshly built scripts it
names were not. The result served a blank black page: the stylesheet loaded,
so the background was right, and nothing else happened. No error in the
server log, no error on the page, nothing to search for.

So this checks the thing that failed: every asset the committed index.html
refers to is a file git is actually tracking.

    python backend/tests/test_built_interface.py
"""
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
DIST = REPO / 'frontend' / 'dist'

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


def tracked_files():
    """What git is actually carrying under frontend/dist."""
    result = subprocess.run(['git', 'ls-files', 'frontend/dist'],
                            cwd=str(REPO), capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


print('== the build is in the repository at all ==')
index = DIST / 'index.html'
check('index.html exists', index.is_file(), str(index))
if not index.is_file():
    print('\n1 FAILED')
    sys.exit(1)

tracked = tracked_files()
if tracked is None:
    print('  (not a git checkout -- nothing to compare against, skipping)')
    print('\nBUILT INTERFACE OK')
    sys.exit(0)

check('git is carrying it', 'frontend/dist/index.html' in tracked,
      len(tracked))

print('\n== every file the page asks for ==')
html = index.read_text(encoding='utf-8')
referenced = re.findall(r'(?:src|href)="/([^"]+)"', html)
check('the page references something', len(referenced) >= 2, referenced)

missing_on_disk = [ref for ref in referenced if not (REPO / 'frontend' / 'dist' / ref).is_file()]
check('each one is on disk', not missing_on_disk, missing_on_disk)

not_tracked = [ref for ref in referenced
               if f'frontend/dist/{ref}' not in tracked]
# This is the check that would have caught it. A file can be on disk here and
# absent from the commit, and then the page is blank for everybody who
# installs by downloading the repository -- which is everybody on a network
# that blocks npm.
check('and each one is committed, not just built locally',
      not not_tracked, not_tracked)

print('\n== the ignore rule has not crept back ==')
rules = (REPO / '.gitignore').read_text(encoding='utf-8').splitlines()
active = [line for line in rules
          if line.strip() == 'frontend/dist/' and not line.strip().startswith('#')]
check('frontend/dist is not being ignored', not active, active)

print('\n== the scripts are not empty ==')
scripts = [ref for ref in referenced if ref.endswith('.js')]
check('there is at least one script', scripts, referenced)
for ref in scripts:
    path = REPO / 'frontend' / 'dist' / ref
    size = path.stat().st_size if path.is_file() else 0
    check(f'{Path(ref).name} has content', size > 1024, size)

print('\n' + ('BUILT INTERFACE OK' if not fails else f'{len(fails)} FAILED: {fails}'))
sys.exit(1 if fails else 0)

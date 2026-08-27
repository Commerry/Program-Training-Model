"""
Find var(--x) references that nothing defines.

An undefined custom property with no fallback invalidates the whole
declaration at compute time, so a single typo inside a linear-gradient wipes
out the entire background and leaves no error anywhere.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(r'd:\Vision-PSE\Train-Model-Webapp-main\frontend\src')

defined = set(re.findall(r'^\s*(--[a-z0-9-]+)\s*:',
                         (ROOT / 'assets/styles/main.css').read_text(encoding='utf-8'),
                         re.M))

# Locally declared properties inside a component count as defined there.
STYLE_RE = re.compile(r'<style[^>]*>(.*?)</style>', re.S)
USE_RE = re.compile(r'var\(\s*(--[a-z0-9-]+)\s*(,)?')

missing = {}
for path in sorted(ROOT.rglob('*.vue')):
    text = path.read_text(encoding='utf-8')
    for css in STYLE_RE.findall(text):
        local = set(re.findall(r'(--[a-z0-9-]+)\s*:', css))
        for match in USE_RE.finditer(css):
            name, has_fallback = match.group(1), match.group(2)
            if has_fallback:
                continue  # a fallback keeps the declaration valid
            if name not in defined and name not in local:
                line = css[:match.start()].count('\n') + 1
                missing.setdefault(path.name, set()).add(name)

for css_path in [ROOT / 'assets/styles/main.css']:
    css = css_path.read_text(encoding='utf-8')
    for match in USE_RE.finditer(css):
        name, has_fallback = match.group(1), match.group(2)
        if not has_fallback and name not in defined:
            missing.setdefault(css_path.name, set()).add(name)

if missing:
    total = sum(len(v) for v in missing.values())
    print(f'{total} undefined custom propert(ies):\n')
    for filename, names in sorted(missing.items()):
        print(f'  {filename}')
        for name in sorted(names):
            print(f'      {name}')
    sys.exit(1)

print('every var() reference resolves')

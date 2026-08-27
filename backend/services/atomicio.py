"""
Atomic JSON writes that survive Windows file sharing.

Two separate problems have to be solved here.

1. A reader must never see a half-written document, which is why every write
   goes to a temp file and is renamed into place.

2. On Windows, `os.replace` over a file that any process currently has open
   fails with PermissionError (WinError 5). Python's `open()` does not request
   FILE_SHARE_DELETE, so an ordinary reader — the gallery loading annotations,
   a training worker polling its config between batches — blocks the rename.
   Verified on this machine: replacing a target held by a plain `open()` for
   reading raises immediately.

The rename is therefore retried briefly, and a write that still cannot land
reports failure so the caller can tell the user instead of claiming success.
That last part matters: an annotator whose save silently did nothing loses
work and has no way to know.
"""

import itertools
import json
import os
import random
import threading
import time
from pathlib import Path

_counter = itertools.count()

# Readers hold a file for well under a millisecond, so a handful of short
# retries covers any realistic collision without stalling the request.
_RETRY_DELAYS = (0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.25)


class AtomicWriteError(OSError):
    """A write could not be committed to disk."""


def _temp_path(target: Path) -> Path:
    """A temp name unique to this process, thread and call."""
    token = f'{os.getpid()}.{threading.get_ident()}.{next(_counter)}'
    return target.with_name(f'{target.name}.{token}.tmp')


def _replace_with_retry(tmp: Path, target: Path):
    """Rename tmp over target, retrying while another process holds it open."""
    last = None
    for delay in (0.0,) + _RETRY_DELAYS:
        if delay:
            # Jitter so several writers colliding on the same file do not keep
            # retrying in lockstep.
            time.sleep(delay * (0.5 + random.random()))
        try:
            os.replace(tmp, target)
            return
        except PermissionError as exc:     # target is open elsewhere
            last = exc
        except FileNotFoundError as exc:   # our temp file vanished
            raise AtomicWriteError(f'temp file disappeared: {tmp}') from exc
    raise AtomicWriteError(
        f'could not replace {target.name} after {len(_RETRY_DELAYS)} retries; '
        f'another process is holding it open'
    ) from last


def write_json(path, data, indent=2):
    """
    Write JSON atomically.

    Raises AtomicWriteError if the data could not be committed. Callers that
    report success to a user must let this propagate rather than swallow it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path)
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp, path)
    except AtomicWriteError:
        tmp.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise AtomicWriteError(f'could not write {path.name}: {exc}') from exc
    return True


def write_json_best_effort(path, data, indent=2):
    """
    Write JSON, returning False instead of raising.

    For caches and other data that can be regenerated, where a failed write is
    not worth failing the request over.
    """
    try:
        write_json(path, data, indent=indent)
        return True
    except (AtomicWriteError, OSError):
        return False


def read_json(path, default=None):
    """
    Read JSON, tolerating a BOM and returning `default` on any failure.

    Reads are kept as short as possible: the file is opened, slurped and
    closed, because every millisecond a reader holds a handle is a millisecond
    a writer's rename can fail.
    """
    try:
        with open(path, 'rb') as f:
            raw = f.read()
    except OSError:
        return default
    if not raw:
        return default
    try:
        return json.loads(raw.decode('utf-8-sig'))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return default

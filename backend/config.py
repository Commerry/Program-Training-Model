"""
Central configuration and path resolution.

Every path used anywhere in the backend is derived from here, so nothing
depends on the current working directory. That was the root cause of a whole
class of "works when started from folder X, breaks from folder Y" bugs.
"""

import os
import secrets
from pathlib import Path

# <repo>/backend/config.py  ->  <repo>
REPO_ROOT = Path(__file__).resolve().parent.parent

# .env is read before anything below looks at os.environ. Without this every
# setting documented in .env.example was silently ignored — including
# ADMIN_PASSWORD and SECRET_KEY, which is a security problem, not a
# convenience one. Values already in the real environment win.
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / '.env', override=False)
except ImportError:  # python-dotenv is optional; env vars still work
    pass
BACKEND_ROOT = REPO_ROOT / 'backend'

# Directory holding every training project (images/, annotations/, training/).
# Resolution order:
#   1. PROJECTS_ROOT environment variable
#   2. <repo>/data/projects              (current layout)
#   3. <repo>/training_module/projects   (legacy layout, kept working)
_LEGACY_PROJECTS_ROOT = REPO_ROOT / 'training_module' / 'projects'
_DEFAULT_PROJECTS_ROOT = REPO_ROOT / 'data' / 'projects'


def _resolve_projects_root() -> Path:
    env_value = os.environ.get('PROJECTS_ROOT')
    if env_value:
        return Path(env_value).expanduser().resolve()
    if _DEFAULT_PROJECTS_ROOT.exists():
        return _DEFAULT_PROJECTS_ROOT
    if _LEGACY_PROJECTS_ROOT.exists():
        return _LEGACY_PROJECTS_ROOT
    return _DEFAULT_PROJECTS_ROOT


PROJECTS_ROOT = _resolve_projects_root()
INSTANCE_DIR = REPO_ROOT / 'data' / 'instance'

# Pretrained checkpoints (yolo11s.pt and friends) are downloaded on first use.
# Giving them an explicit home keeps them out of whatever directory a worker
# happened to start in.
WEIGHTS_CACHE_DIR = REPO_ROOT / 'data' / 'weights'

# Models brought in from elsewhere -- an export from Custom Vision, a detector
# from a previous system -- so they can pre-label a new project before this
# installation has trained anything of its own. Each gets its own folder so the
# files that belong with it travel with it: an ONNX carries no class names, and
# nothing in it records how it wants to be fed.
IMPORTED_MODELS_DIR = REPO_ROOT / 'data' / 'imported'

# Ports. Frontend dev server proxies /api to the backend port.
BACKEND_PORT = int(os.environ.get('BACKEND_PORT', 64031))
FRONTEND_PORT = int(os.environ.get('FRONTEND_PORT', 64030))
# 127.0.0.1 keeps the API off the network. Set 0.0.0.0 to serve other machines.
BACKEND_HOST = os.environ.get('BACKEND_HOST', '127.0.0.1')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _secret_key():
    """
    The session signing key.

    A key hardcoded in the source is a key everyone who has seen the source
    can forge session cookies with, which stops being theoretical the moment
    the app is reachable from another machine. If nothing is configured, one
    is generated and stored in data/instance/secret_key so sessions still
    survive a restart.
    """
    from_env = os.environ.get('SECRET_KEY')
    if from_env:
        return from_env

    key_file = INSTANCE_DIR / 'secret_key'
    try:
        if key_file.exists():
            existing = key_file.read_text(encoding='utf-8').strip()
            if existing:
                return existing
        INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
        generated = secrets.token_urlsafe(48)
        key_file.write_text(generated, encoding='utf-8')
        return generated
    except OSError:
        # Read-only install: fall back to a per-process key. Sessions will not
        # survive a restart, which is inconvenient but not insecure.
        return secrets.token_urlsafe(48)


class Config:
    SECRET_KEY = _secret_key()
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', f'sqlite:///{(INSTANCE_DIR / "vision_training.db").as_posix()}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB — bulk image uploads
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_HTTPONLY = True

    # Set REQUIRE_AUTH=0 to expose the training API without a login (local use).
    REQUIRE_AUTH = os.environ.get('REQUIRE_AUTH', '1') not in ('0', 'false', 'False')

    # Only consulted when the frontend is served from a different origin than
    # the API. In the normal setup the dev server proxies /api, and the built
    # frontend is served by this app, so both are same-origin and CORS never
    # comes into play.
    CORS_ORIGINS = [
        o.strip() for o in os.environ.get(
            'CORS_ORIGINS',
            f'http://localhost:{FRONTEND_PORT},http://127.0.0.1:{FRONTEND_PORT}'
        ).split(',') if o.strip()
    ]

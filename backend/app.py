"""
Application entry point.

Run with:  python backend/app.py       (or `npm run dev` from frontend/)
"""

import os
import sys
from pathlib import Path

# Allow "from services import ..." whichever directory the process starts in.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify  # noqa: E402
from flask_cors import CORS  # noqa: E402

from config import (  # noqa: E402
    BACKEND_HOST, BACKEND_PORT, INSTANCE_DIR, PROJECTS_ROOT, REPO_ROOT,
    WEIGHTS_CACHE_DIR, Config,
)
from extensions import bcrypt, db, login_manager  # noqa: E402


def create_app(config_object=Config):
    app = Flask(__name__)
    app.config.from_object(config_object)

    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    WEIGHTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # In development the Vite dev server proxies /api, so requests are
    # same-origin and CORS is only needed if the frontend is served elsewhere.
    CORS(app, supports_credentials=True, origins=app.config['CORS_ORIGINS'])

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)

    from models import User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    @login_manager.unauthorized_handler
    def unauthorized():
        return jsonify({'success': False, 'message': 'Authentication required'}), 401

    from api import register_blueprints, register_error_handlers
    register_blueprints(app)
    _serve_built_frontend(app)
    register_error_handlers(app)

    @app.get('/api/health')
    def health():
        return jsonify({
            'success': True,
            'service': 'Vision Training Platform API',
            'version': '3.0.0',
            'projects_root': str(PROJECTS_ROOT),
            'auth_required': app.config['REQUIRE_AUTH'],
        })

    with app.app_context():
        db.create_all()
        _ensure_default_admin(app)

    _warm_project_indexes()
    return app


def _warm_project_indexes():
    """
    Build each project's gallery index in the background at startup.

    Building it reads every annotation file, which takes a few seconds on a
    large project. Doing it here means the cost lands before anyone opens a
    page rather than on the first gallery request.
    """
    import threading

    def warm():
        try:
            from services import projects
            for project in projects.list_projects():
                try:
                    projects.load_index(project['name'])
                except Exception:  # noqa: BLE001 - warming is best-effort
                    pass
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=warm, name='index-warmup', daemon=True).start()


DIST_DIR = REPO_ROOT / 'frontend' / 'dist'


def _serve_built_frontend(app):
    """
    Serve frontend/dist from this app when it has been built.

    This is what makes the tool reachable on one port from another machine:
    no Node process, no proxy, and the API is same-origin with the pages so
    the session cookie works with no CORS configuration at all. During
    development the Vite dev server is used instead and this does nothing.
    """
    from flask import send_from_directory

    # Registered for every method so an unmatched /api/... path is a JSON 404
    # rather than a 405 from the SPA rule (which is GET-only) or an HTML page.
    # Werkzeug ranks a <path:> converter below concrete rules, so every real
    # endpoint still wins this match.
    @app.route('/api/<path:_unmatched>',
               methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
    def api_not_found(_unmatched):
        return jsonify({'success': False, 'message': 'No such API endpoint'}), 404

    @app.get('/')
    @app.get('/<path:requested>')
    def spa(requested=''):
        # Never answer for the API namespace. Without this the catch-all
        # swallows a mistyped or wrong-method /api/... request and hands back
        # an HTML page (or a 405), instead of the JSON error every client here
        # is written to expect.
        if requested == 'api' or requested.startswith('api/'):
            return jsonify({'success': False, 'message': 'Not found'}), 404

        if not DIST_DIR.is_dir():
            return jsonify({
                'success': True,
                'service': 'Vision Training Platform API',
                'message': 'The API is running. Build the frontend with '
                           '"npm run build" in frontend/ to serve the UI from '
                           'this port, or start the Vite dev server.',
            })

        candidate = (DIST_DIR / requested).resolve() if requested else None
        if (candidate and candidate.is_file()
                and candidate.is_relative_to(DIST_DIR.resolve())):
            return send_from_directory(DIST_DIR, requested)

        # Anything else is a client-side route; the SPA resolves it itself.
        return send_from_directory(DIST_DIR, 'index.html')


def _ensure_default_admin(app):
    """
    Create the first administrator so a fresh install can be signed into.

    The password is only defaulted when nothing is configured, and the app
    says so loudly in that case.
    """
    from models import User

    if User.query.first():
        return

    username = os.environ.get('ADMIN_USERNAME', 'admin')
    password = os.environ.get('ADMIN_PASSWORD')
    generated = password is None
    if generated:
        password = 'admin123'

    admin = User(
        username=username,
        email=os.environ.get('ADMIN_EMAIL', 'admin@vision-training.local'),
        full_name='Administrator',
        is_admin=True,
    )
    admin.set_password(password)
    db.session.add(admin)
    db.session.commit()

    print(f'Created administrator account "{username}".')
    if generated:
        print('  Password: admin123  <-- change this from Settings, or set '
              'ADMIN_PASSWORD before first run.')


def _local_addresses():
    """Best-effort list of this machine's IPv4 addresses, for the banner."""
    import socket
    addresses = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if not address.startswith('127.') and address not in addresses:
                addresses.append(address)
    except OSError:
        pass
    return addresses or ['<this machine\'s IP>']


def _using_default_admin_password():
    from models import User
    with app.app_context():
        admin = User.query.filter_by(username=os.environ.get('ADMIN_USERNAME', 'admin')).first()
        return bool(admin and admin.check_password('admin123'))


app = create_app()


if __name__ == '__main__':
    # Defaults are the safe ones. The Werkzeug debugger executes arbitrary code
    # from the browser, so it is opt-in rather than opt-out, and the server
    # binds to loopback unless the machine is deliberately asked to serve the
    # network.
    debug = os.environ.get('FLASK_DEBUG', '0') in ('1', 'true', 'True')
    host = BACKEND_HOST

    print(f'Projects directory: {PROJECTS_ROOT}')
    if DIST_DIR.is_dir():
        print(f'Serving the built UI from {DIST_DIR}')
    else:
        print('No frontend build found — run "npm run build" in frontend/ to '
              'serve the UI from this port.')

    if host in ('0.0.0.0', '::'):
        for address in _local_addresses():
            print(f'  http://{address}:{BACKEND_PORT}')
        print()
        print('  Reachable from other machines on this network.')
        if _using_default_admin_password():
            print('  WARNING: the admin account still uses the default '
                  'password. Change it in Settings before leaving this open.')
    else:
        print(f'  http://127.0.0.1:{BACKEND_PORT}  (this machine only — set '
              'BACKEND_HOST=0.0.0.0 to allow others)')

    if debug:
        print('  WARNING: debug mode is on. The Werkzeug debugger runs '
              'arbitrary code; never expose it to a network.')

    # The reloader spawns a second process, which would double-launch training
    # subprocess bookkeeping, so it stays off even in debug mode.
    app.run(host=host, port=BACKEND_PORT, debug=debug, use_reloader=False,
            threaded=True)

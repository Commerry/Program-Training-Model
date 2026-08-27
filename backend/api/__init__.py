"""
HTTP layer.

Route functions stay thin: they parse the request, call a service, and return
the result. Every service raises ProjectError for anything the client did
wrong, and the handler registered here turns that into the right status code,
so no route needs its own try/except wrapper.
"""

from functools import wraps

from flask import Blueprint, current_app, jsonify
from flask_login import current_user

from services.projects import ProjectError


def ok(payload=None, **extra):
    """Standard success envelope: {'success': True, ...}."""
    body = {'success': True}
    if payload:
        body.update(payload)
    body.update(extra)
    return jsonify(body)


def login_required_api(view):
    """
    Reject unauthenticated calls with JSON rather than a login redirect.

    Flask-Login's own decorator answers with a 302 to the login page, which an
    XHR client cannot act on; a 401 lets the frontend clear its session and
    send the user to /login.
    """
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_app.config.get('REQUIRE_AUTH', True):
            return view(*args, **kwargs)
        if not current_user.is_authenticated:
            return jsonify({'success': False, 'message': 'Authentication required'}), 401
        return view(*args, **kwargs)
    return wrapper


def register_error_handlers(app):
    @app.errorhandler(ProjectError)
    def _handle_project_error(error):
        return jsonify({'success': False, 'message': error.message}), error.status

    @app.errorhandler(404)
    def _handle_not_found(_error):
        return jsonify({'success': False, 'message': 'Not found'}), 404

    @app.errorhandler(413)
    def _handle_too_large(_error):
        return jsonify({
            'success': False,
            'message': 'Upload is too large. Try importing fewer images at a time.',
        }), 413

    @app.errorhandler(Exception)
    def _handle_unexpected(error):
        # Werkzeug's HTTP exceptions carry their own status and message.
        from werkzeug.exceptions import HTTPException
        if isinstance(error, HTTPException):
            return jsonify({'success': False, 'message': error.description}), error.code

        # Logged in full for the operator; the client gets a generic message.
        # str(error) routinely contains absolute filesystem paths and library
        # internals, which do not belong in a browser response.
        app.logger.exception('Unhandled error')
        payload = {'success': False,
                   'message': 'Internal server error. Check the server log for details.'}
        if app.debug:
            import traceback
            payload['message'] = str(error) or 'Internal server error'
            payload['trace'] = traceback.format_exc()
        return jsonify(payload), 500


def register_blueprints(app):
    from api import auth, inference, projects, training

    # Built per call rather than at module scope: a Flask blueprint object can
    # only be registered once, and building it here keeps create_app() usable
    # more than once in the same process (tests, scripts).
    api_bp = Blueprint('api', __name__, url_prefix='/api')
    api_bp.register_blueprint(projects.projects_bp)
    api_bp.register_blueprint(training.training_bp)
    api_bp.register_blueprint(inference.inference_bp)

    app.register_blueprint(auth.auth_bp)
    app.register_blueprint(api_bp)

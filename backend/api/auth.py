"""Registration, login and profile endpoints."""

import re
from datetime import datetime

from flask import Blueprint, request
from flask_login import current_user, login_user, logout_user

from api import login_required_api, ok
from extensions import db
from models import User
from services.projects import ProjectError

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

EMAIL_PATTERN = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
MIN_PASSWORD_LENGTH = 8


def _validate_credentials(username, email, password):
    if not username or not 3 <= len(username) <= 80:
        raise ProjectError('Username must be between 3 and 80 characters')
    if not re.match(r'^[A-Za-z0-9._-]+$', username):
        raise ProjectError('Username may contain letters, digits, dots, underscores and hyphens')
    if not email or not EMAIL_PATTERN.match(email):
        raise ProjectError('Enter a valid email address')
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise ProjectError(f'Password must be at least {MIN_PASSWORD_LENGTH} characters')


@auth_bp.post('/register')
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    _validate_credentials(username, email, password)

    if User.query.filter_by(username=username).first():
        raise ProjectError('That username is already taken')
    if User.query.filter_by(email=email).first():
        raise ProjectError('That email address is already registered')

    user = User(username=username, email=email,
                full_name=(data.get('full_name') or '').strip())
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    login_user(user)
    return ok({'message': 'Account created', 'user': user.to_dict()})


@auth_bp.post('/login')
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    if not username or not password:
        raise ProjectError('Username and password are required')

    user = User.query.filter_by(username=username).first()
    # The same message for both cases so the endpoint cannot be used to
    # enumerate which usernames exist.
    if not user or not user.check_password(password):
        raise ProjectError('Incorrect username or password', status=401)
    if not user.is_active:
        raise ProjectError('This account has been disabled', status=403)

    login_user(user, remember=bool(data.get('remember')))
    user.last_login = datetime.utcnow()
    db.session.commit()

    return ok({'message': 'Signed in', 'user': user.to_dict()})


@auth_bp.post('/logout')
def logout():
    logout_user()
    return ok({'message': 'Signed out'})


@auth_bp.get('/me')
def me():
    """
    Who is signed in, or nobody.

    Deliberately not behind login_required_api: "nobody is signed in" is a
    valid answer to this question, not an error. Returning 401 made the app's
    own startup probe log a console error on every visit to the login page,
    and made a normal signed-out state indistinguishable from a real failure.
    """
    if not current_user.is_authenticated:
        return ok({'user': None})
    return ok({'user': current_user.to_dict()})


@auth_bp.put('/profile')
@login_required_api
def update_profile():
    data = request.get_json(silent=True) or {}

    if 'full_name' in data:
        current_user.full_name = (data.get('full_name') or '').strip()

    email = (data.get('email') or '').strip().lower()
    if email and email != current_user.email:
        if not EMAIL_PATTERN.match(email):
            raise ProjectError('Enter a valid email address')
        existing = User.query.filter_by(email=email).first()
        if existing and existing.id != current_user.id:
            raise ProjectError('That email address is already in use')
        current_user.email = email

    new_password = data.get('new_password')
    if new_password:
        if not current_user.check_password(data.get('current_password') or ''):
            raise ProjectError('Current password is incorrect')
        if len(new_password) < MIN_PASSWORD_LENGTH:
            raise ProjectError(f'Password must be at least {MIN_PASSWORD_LENGTH} characters')
        current_user.set_password(new_password)

    db.session.commit()
    return ok({'message': 'Profile updated', 'user': current_user.to_dict()})

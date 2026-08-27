#!/usr/bin/env bash
#
# Vision Training Platform launcher (Linux / macOS).
#
#   ./start.sh              local only, backend + Vite dev server
#   ./start.sh --network    reachable from other machines on the LAN
#   ./start.sh --backend    backend only
#   ./start.sh --frontend   frontend only
#   ./start.sh --install    install dependencies first
#   ./start.sh --update     pull the latest code, rebuild, then start
#   ./start.sh --doctor     check the environment and exit
#
# --network builds the frontend and serves it from the backend on a single
# port. That is deliberately different from the local dev setup: one port is
# far easier to open through a firewall, needs no Node process at run time, and
# keeps the UI and the API same-origin so the login cookie works with no CORS
# configuration at all.
#
# The PowerShell equivalent for Windows is start.ps1, with the same switches.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-64031}"
FRONTEND_PORT="${FRONTEND_PORT:-64030}"

DO_BACKEND=0
DO_FRONTEND=0
DO_INSTALL=0
DO_NETWORK=0
DO_DOCTOR=0
DO_UPDATE=0

for arg in "$@"; do
    case "$arg" in
        --backend)  DO_BACKEND=1 ;;
        --frontend) DO_FRONTEND=1 ;;
        --install)  DO_INSTALL=1 ;;
        --update)   DO_UPDATE=1 ;;
        --network)  DO_NETWORK=1 ;;
        --doctor)   DO_DOCTOR=1 ;;
        -h|--help)  sed -n '3,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

have() { command -v "$1" >/dev/null 2>&1; }

# A virtualenv's python if one is present, otherwise the first system one that
# actually runs. Being on PATH is not enough to go on: Windows ships a python3
# stub that exists, resolves, and then refuses to do anything.
PYTHON=""
for candidate in "$ROOT/.venv/bin/python" python3 python; do
    if "$candidate" --version >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "No working python was found on PATH. Install it, e.g." >&2
    echo "    sudo apt install python3 python3-venv python3-pip     # Debian/Ubuntu" >&2
    echo "    sudo dnf install python3 python3-pip                  # Fedora/RHEL" >&2
    exit 1
fi

local_addresses() {
    # Every non-loopback IPv4 this machine answers on.
    if have ip; then
        ip -4 -o addr show scope global 2>/dev/null | awk '{split($4,a,"/"); print a[1]}'
    elif have ifconfig; then
        ifconfig 2>/dev/null | awk '/inet /{print $2}' | grep -v '^127\.'
    fi
}

if [[ $DO_DOCTOR -eq 1 ]]; then
    exec "$PYTHON" "$ROOT/backend/scripts/doctor.py"
fi

stop_running_server() {
    # A server already bound to the port keeps serving the old code: the new
    # process cannot bind, and requests for endpoints added by the update come
    # back as 404 from the still-running old one. This is the single easiest
    # way to conclude an update did not work when it did.
    local port="$1"
    local pids=""
    if have lsof; then
        pids=$(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)
    elif have fuser; then
        pids=$(fuser -n tcp "$port" 2>/dev/null | tr -d ' ' || true)
    elif have ss; then
        pids=$(ss -lptn "sport = :${port}" 2>/dev/null |
               grep -oP 'pid=\K[0-9]+' | sort -u || true)
    fi
    for pid in $pids; do
        echo "  stopping the server already on port ${port} (pid ${pid})"
        kill "$pid" 2>/dev/null || true
    done
    [[ -n "$pids" ]] && sleep 2
    return 0
}

if [[ $DO_UPDATE -eq 1 ]]; then
    if ! have git || [[ ! -d "$ROOT/.git" ]]; then
        echo "This is not a git checkout, so there is nothing to pull." >&2
        echo "Download the latest copy instead, or clone the repository." >&2
        exit 1
    fi

    cd "$ROOT"
    stop_running_server "$BACKEND_PORT"
    stop_running_server "$FRONTEND_PORT"

    before=$(git rev-parse HEAD)

    # Annotations are tracked on purpose -- they are work that cannot be
    # regenerated -- which means boxes drawn since the last update show up as
    # local changes and would block the pull. They are set aside and put back
    # afterwards rather than being lost or being allowed to stop the update.
    stashed=0
    if [[ -n "$(git status --porcelain -- data 2>/dev/null)" ]]; then
        echo "  setting your data aside while the code updates"
        git stash push --quiet --include-untracked --message 'start.sh --update' -- data \
            && stashed=1
    fi

    echo "Fetching..."
    # git reports "Already up to date." itself, so nothing here repeats it.
    # Its output is captured so a failure can be shown in full.
    if ! pull_output=$(git pull --ff-only 2>&1); then
        echo "$pull_output" >&2
        echo >&2
        echo "The pull did not go through cleanly." >&2
        echo "Usually that means this checkout has local commits. Look at:" >&2
        echo "    git status" >&2
        echo "    git log --oneline -3" >&2
        [[ $stashed -eq 1 ]] && git stash pop --quiet && echo "Your data was put back." >&2
        exit 1
    fi

    if [[ $stashed -eq 1 ]]; then
        if git stash pop --quiet; then
            echo "  your data is back"
        else
            echo >&2
            echo "Your data could not be put back automatically; it is safe in" >&2
            echo "the stash. Recover it with:  git stash pop" >&2
            exit 1
        fi
    fi

    after=$(git rev-parse HEAD)
    if [[ "$before" != "$after" ]]; then
        echo "Updated:"
        git --no-pager log --oneline "${before}..${after}" | sed 's/^/  /'
    fi

    # Only reinstall when the lists actually changed; pip and npm are slow
    # enough that doing it every time would discourage updating at all.
    if [[ "$before" != "$after" ]]; then
        changed=$(git diff --name-only "$before" "$after")
        if grep -q 'backend/requirements.txt' <<<"$changed"; then
            echo "Python dependencies changed; installing..."
            "$PYTHON" -m pip install -q -r "$ROOT/backend/requirements.txt"
        fi
        if grep -q 'frontend/package.json' <<<"$changed" && have npm; then
            echo "Frontend dependencies changed; installing..."
            (cd "$ROOT/frontend" && npm install --silent)
        fi
    fi

    # The backend serves frontend/dist, not frontend/src, so without this the
    # browser keeps showing the previous version of the interface.
    if have npm; then
        echo "Building the interface..."
        (cd "$ROOT/frontend" && npm run build)
    else
        echo "npm is not installed, so the interface was not rebuilt." >&2
        echo "The API is up to date; the pages are not." >&2
    fi

    echo
    echo "Update finished. Starting..."
    echo
fi

if [[ $DO_INSTALL -eq 1 ]]; then
    echo "Installing Python dependencies..."
    "$PYTHON" -m pip install -r "$ROOT/backend/requirements.txt"

    if have npm; then
        echo "Installing frontend dependencies..."
        (cd "$ROOT/frontend" && npm install)
    else
        echo "npm was not found — skipping the frontend install." >&2
        echo "Install Node 18+ if you want the dev server; --network only needs it once, to build." >&2
    fi
fi

# ── Network mode: one port, built UI served by the backend ──────────────────
if [[ $DO_NETWORK -eq 1 ]]; then
    if [[ ! -d "$ROOT/frontend/dist" ]] || [[ -n "$(find "$ROOT/frontend/src" -newer "$ROOT/frontend/dist" -print -quit 2>/dev/null)" ]]; then
        if ! have npm; then
            echo "The UI has not been built and npm is not installed." >&2
            echo "Install Node 18+ and run:  cd frontend && npm install && npm run build" >&2
            exit 1
        fi
        echo "Building the frontend..."
        (cd "$ROOT/frontend" && npm run build)
    else
        echo "Using the existing build in frontend/dist."
    fi

    export BACKEND_HOST=0.0.0.0
    export BACKEND_PORT

    echo
    echo "========================================================"
    echo " This will be reachable from other machines"
    echo "========================================================"
    echo " Change the admin password in Settings before leaving it open."
    echo " The default admin/admin123 lets anyone on this network in."
    echo
    echo " Open from another machine:"
    while read -r ip; do
        [[ -n "$ip" ]] && echo "     http://${ip}:${BACKEND_PORT}"
    done < <(local_addresses)
    echo "     http://localhost:${BACKEND_PORT}  (this machine)"
    echo

    # Most desktop distributions ship firewalld or ufw with inbound traffic
    # denied, which looks exactly like the server not running.
    if have ufw && ufw status 2>/dev/null | grep -q '^Status: active'; then
        echo " ufw is active. To let other machines in:"
        echo "     sudo ufw allow ${BACKEND_PORT}/tcp"
        echo
    elif have firewall-cmd && firewall-cmd --state >/dev/null 2>&1; then
        echo " firewalld is active. To let other machines in:"
        echo "     sudo firewall-cmd --add-port=${BACKEND_PORT}/tcp --permanent && sudo firewall-cmd --reload"
        echo
    fi

    cd "$ROOT"
    exec "$PYTHON" backend/app.py
fi

# ── Local development: backend on loopback, Vite in front ───────────────────
if [[ $DO_BACKEND -eq 0 && $DO_FRONTEND -eq 0 ]]; then
    DO_BACKEND=1
    DO_FRONTEND=1
fi

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

if [[ $DO_BACKEND -eq 1 ]]; then
    echo "Starting backend on http://127.0.0.1:${BACKEND_PORT}"
    cd "$ROOT"
    BACKEND_PORT="$BACKEND_PORT" "$PYTHON" backend/app.py &
    pids+=($!)
fi

if [[ $DO_FRONTEND -eq 1 ]]; then
    if ! have npm; then
        echo "npm was not found — cannot start the dev server." >&2
        echo "Use ./start.sh --network instead, which serves a build from the backend." >&2
        [[ $DO_BACKEND -eq 1 ]] && wait
        exit 1
    fi
    [[ $DO_BACKEND -eq 1 ]] && sleep 2
    echo "Starting frontend on http://localhost:${FRONTEND_PORT}"
    (cd "$ROOT/frontend" && npm run dev) &
    pids+=($!)
fi

echo
echo "Open http://localhost:${FRONTEND_PORT}"
echo "Press Ctrl+C to stop."
wait

#!/usr/bin/env bash
#
# Vision Training Platform launcher (Linux / macOS).
#
#   ./start.sh              local only, backend + Vite dev server
#   ./start.sh --network    reachable from other machines on the LAN
#   ./start.sh --backend    backend only
#   ./start.sh --frontend   frontend only
#   ./start.sh --install    install dependencies first
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

for arg in "$@"; do
    case "$arg" in
        --backend)  DO_BACKEND=1 ;;
        --frontend) DO_FRONTEND=1 ;;
        --install)  DO_INSTALL=1 ;;
        --network)  DO_NETWORK=1 ;;
        --doctor)   DO_DOCTOR=1 ;;
        -h|--help)  sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# A virtualenv's python, if one is active or present, otherwise the system one.
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

have() { command -v "$1" >/dev/null 2>&1; }

if ! "$PYTHON" --version >/dev/null 2>&1; then
    echo "python3 was not found. Install it with your package manager, e.g." >&2
    echo "    sudo apt install python3 python3-venv python3-pip" >&2
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

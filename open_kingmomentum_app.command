#!/bin/zsh
set -e
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"
if [[ ! -x "$APP_DIR/.venv/bin/streamlit" ]]; then
  python3 -m venv "$APP_DIR/.venv"
  "$APP_DIR/.venv/bin/python" -m pip install -r "$APP_DIR/requirements.txt"
fi
if ! "$APP_DIR/.venv/bin/python" -c "import streamlit, pandas, numpy, plotly, pyarrow" >/dev/null 2>&1; then
  "$APP_DIR/.venv/bin/python" -m pip install -r "$APP_DIR/requirements.txt"
fi
PORT="${KINGMOMENTUM_PORT:-8501}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
"$APP_DIR/.venv/bin/streamlit" run "$APP_DIR/app.py" --server.headless true --server.port "$PORT" >"$APP_DIR/streamlit.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM
sleep 2
open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true
wait "$SERVER_PID"

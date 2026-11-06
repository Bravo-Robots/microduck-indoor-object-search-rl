#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  task_python=.venv/bin/python
else
  echo 'Environment missing. Run: uv sync --locked' >&2
  exit 1
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$task_python" -m duck_search "$@"

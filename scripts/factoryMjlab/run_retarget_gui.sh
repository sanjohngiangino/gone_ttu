#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export DISPLAY="${DISPLAY:-:1}"
export MUJOCO_GL=egl
echo "DISPLAY=$DISPLAY"
# Live EGL+OpenCV viewer (mouse orbit)
exec python -u scripts/factoryMjlab/view_retarget_motions.py --start 1 "$@"

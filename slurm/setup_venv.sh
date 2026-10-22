#!/usr/bin/env bash
# Install requirements.txt into .venv inside the container. The container's Python has no
# ensurepip, so the (empty) venv is created on the login node with the same interpreter
# (/usr/bin/python3, 3.10.12 on both):
#   rm -rf .venv && /usr/bin/python3 -m venv .venv
#   slurm/submit.sh -J setup_venv -t 00:30:00 -- bash slurm/setup_venv.sh
set -euo pipefail
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

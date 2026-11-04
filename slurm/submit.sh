#!/usr/bin/env bash
# Submit a command to Slurm; it runs inside the project container with .venv first on PATH.
#
#   slurm/submit.sh [sbatch options] -- <command ...>
#
# Defaults: partition frida, 1 GPU (A100/H100/B200/B300, the GPUs this container supports),
# 16 CPUs, 128G RAM, 1 hour. Any sbatch option given here overrides the default, e.g.
#   slurm/submit.sh -J t1 -t 02:00:00 --constraint GPU_BRD:B300 -- python src/experiments/pca_analysis/hidden_state.py --out ...
# The command runs from the project root; the log goes to logs/<job name>_<job id>.out.
# Prints the job id.
# Hugging Face caches go to the group's shared cache, never to ~/.cache/huggingface; HF_HOME
# stays at home so the token stays private. pip runs without a cache.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh

opts=()
while [[ $# -gt 0 && $1 != -- ]]; do opts+=("$1"); shift; done
[[ $# -gt 1 ]] || { echo "usage: $0 [sbatch options] -- <command ...>" >&2; exit 1; }
shift
cmd=$(printf '%q ' "$@")
mkdir -p "$ROOT/logs"

sbatch --parsable -p frida -A povejmo -c 16 --mem 128G -t 01:00:00 --gres gpu:1 \
    --constraint 'GPU_BRD:B300|GPU_BRD:B200|GPU_BRD:H100|GPU_BRD:A100' \
    -J mllp -o "$ROOT/logs/%x_%j.out" "${opts[@]}" \
    --wrap "srun --export=ALL,MELLANOX_VISIBLE_DEVICES=none --container-image=$CONTAINER --container-mounts=/shared:/shared \
        env HOME=$HOME PYTHONUNBUFFERED=1 \
        HF_HUB_CACHE=/shared/workspace/povejmo/huggingface_cache/hub HF_DATASETS_CACHE=/shared/workspace/povejmo/huggingface_cache/datasets HF_XET_CACHE=/shared/workspace/povejmo/huggingface_cache/xet HF_ASSETS_CACHE=/shared/workspace/povejmo/huggingface_cache/assets PIP_NO_CACHE_DIR=1 \
        bash -c 'cd $ROOT && export PATH=$ROOT/.venv/bin:\$PATH && echo [\$(hostname)] && $cmd'"

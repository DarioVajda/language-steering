#!/bin/bash
# Teacher-forced KL to the native question (kl.py) inside one Slurm job: one worker per GPU, then the merge.
# Usage: kl.sh <model> <gpus> [main job id]
# With a main job id, the job is a helper: its workers take batches from the main job's claim queue
# (results/<model>/generations/claims/<main job id>/kl/); the main job's merge waits for them.
set -u
cd "$(dirname "$0")/../../.."
G=src/experiments/mgsm_crosslingual
export XL_MODEL=$1
N=$2 MAIN=${3:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
[ -n "$MAIN" ] && export SLURM_JOB_ID=$MAIN
for i in $(seq 0 $((N - 1))); do CUDA_VISIBLE_DEVICES=$i python $G/kl.py run & done
wait
if [ -z "$MAIN" ]; then python $G/kl.py merge; fi

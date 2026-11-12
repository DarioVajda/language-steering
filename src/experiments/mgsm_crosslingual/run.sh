#!/bin/bash
# Run every stage of the experiment inside one Slurm job: vectors, dev, selection, test, tables.
# Usage: run.sh <gpus> <workers per gpu> [main job id]
# With a main job id, the job is a helper: it waits for the main job's vectors and selection and takes units from
# the same claim queue (results/generations/claims/<main job id>/); the main job writes the tables.
set -u
cd "$(dirname "$0")/../../.."
G=src/experiments/mgsm_crosslingual
D=$G/results/generations
N=$1 W=$2 MAIN=${3:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
[ -n "$MAIN" ] && export SLURM_JOB_ID=$MAIN

workers() {
    for i in $(seq 0 $((N * W - 1))); do CUDA_VISIBLE_DEVICES=$((i / W)) python $G/generate_pairs.py "$1" & done
    wait
}
await() { until [ -e "$1" ]; do sleep 30; done; }

if [ -z "$MAIN" ]; then python $G/generate_pairs.py vectors || exit 1; else await $D/vectors.pt; fi
workers dev
if [ -z "$MAIN" ]; then python $G/analyse_pairs.py --select || exit 1; else await $D/chosen.json; fi
workers test
if [ -z "$MAIN" ]; then python $G/analyse_pairs.py; fi

#!/bin/bash
# The MMLU-ProX pilot (pilot.py) inside one Slurm job: one worker per GPU sharing the tasks, then the analysis.
# With HELPER=1 the workers leave when nothing is left to take and there is no analysis: a helper job that joins
# the tasks of a main one (the same options).
# Usage: [HELPER=1] pilot.sh <model> [<HF_HUB_CACHE>] [pilot.py options]
#   e.g. pilot.sh llama-3.1-8b $PWD/logs/hf_cache --max-new 8192 --first-cap 2048 --pairs all
set -u
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 XL_MODEL=$1
[ -n "${2:-}" ] && export HF_HUB_CACHE=$2
shift $(($# < 2 ? $# : 2))
P=src/experiments/mmlu_prox/pilot.py
helper=${HELPER:+--helper}
for g in $(seq 0 $(($(nvidia-smi -L | wc -l) - 1))); do
    CUDA_VISIBLE_DEVICES=$g python $P run "$@" $helper >> "logs/mmlu_prox_pilot_${SLURM_JOB_ID:-local}_$g.out" 2>&1 &
done
rc=0
for p in $(jobs -p); do wait $p || rc=1; done
[ $rc = 0 ] || exit 1
[ -n "$helper" ] || python $P analyse "$@"

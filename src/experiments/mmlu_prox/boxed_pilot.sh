#!/bin/bash
# The pilot of the \boxed{X} answer format (pilot.py --format boxed): all four models on the first 3 questions of each
# category, cot in every language and the other arms on the 11 tuning pairs, each arm's units pooled into one call.
# The models are split over the job's GPUs (one worker per GPU, its models one after another), each analysed when done.
# Usage (in a Slurm job): boxed_pilot.sh
set -u
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
P=src/experiments/mmlu_prox/pilot.py
OPTS="--first 3 --format boxed --pool --max-new 4096 --first-cap 2048 --pairs tune"
SHARED=/shared/workspace/povejmo/huggingface_cache/hub
declare -A HUB=([llama-3.1-8b]=$PWD/logs/hf_cache [qwen3-8b]=$SHARED [gemma3-12b]=$SHARED [qwen3-32b]=$SHARED)
GROUPS_=("llama-3.1-8b qwen3-8b" "gemma3-12b qwen3-32b")
n=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((n - 1))); do
    (for k in $(seq $g $n $((${#GROUPS_[@]} - 1))); do for m in ${GROUPS_[$k]}; do
        log=logs/mmlu_prox_boxed_${SLURM_JOB_ID:-local}_$m.out
        echo "[$(date '+%F %T')] $m on GPU $g" >> $log
        CUDA_VISIBLE_DEVICES=$g XL_MODEL=$m HF_HUB_CACHE=${HUB[$m]} python $P run $OPTS >> $log 2>&1 \
            && CUDA_VISIBLE_DEVICES=$g XL_MODEL=$m HF_HUB_CACHE=${HUB[$m]} python $P analyse $OPTS >> $log 2>&1 \
            || echo "[$(date '+%F %T')] $m failed" >> $log
    done; done) &
done
wait

#!/bin/bash
# The ReCoVeR baseline (recover.py) for several models in one job: one worker per GPU. Each worker goes through the
# models in the order given, runs every task that is free (recover.py work), and starts again from the first model
# after doing any work, so the models listed first get the GPUs first. Jobs on other nodes share the tasks through
# the claim files. A worker stops once none of its models has work left for it; with --main, only once every
# model's analysis exists (so the main job takes over the tasks of a helper that dies).
# Usage: recover.sh [--main] <model>[:<HF_HUB_CACHE>] ...
# e.g. recover.sh --main qwen3-32b:$PWD/logs/hf_cache gemma3-12b qwen3-8b llama-3.1-8b:$PWD/logs/hf_cache
set -u
cd "$(dirname "$0")/../../.."
G=src/experiments/mgsm_crosslingual
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
main=()
[ "${1:-}" = --main ] && { main=(--main); shift; }
specs=("$@")

worker() {  # worker <gpu>
    local -A finished=()
    while [ ${#finished[@]} -lt ${#specs[@]} ]; do
        local progress=0 spec model cache rc
        for spec in "${specs[@]}"; do
            [ -n "${finished[$spec]:-}" ] && continue
            IFS=: read -r model cache <<< "$spec"
            env CUDA_VISIBLE_DEVICES=$1 XL_MODEL=$model HF_HUB_CACHE=${cache:-$HF_HUB_CACHE} \
                python $G/recover.py work "${main[@]}"
            rc=$?
            case $rc in
                0) finished[$spec]=1 ;;
                2) progress=1; break ;;  # back to the first model
                3) ;;
                *) echo "recover.py work failed for $model (exit $rc)"; sleep 60 ;;
            esac
        done
        [ $progress = 0 ] && [ ${#finished[@]} -lt ${#specs[@]} ] && sleep 60
    done
    echo "worker on GPU $1: done"
}

n=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((n - 1))); do
    worker $g > "logs/recover_${SLURM_JOB_ID:-local}_gpu$g.out" 2>&1 &
done
wait

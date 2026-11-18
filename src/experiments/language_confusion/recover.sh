#!/bin/bash
# The ReCoVeR baseline on LCB (recover.py) for several models in one job, all processes on the job's GPU(s): model i
# runs on GPU i % (number of GPUs), so with one large GPU (B300, 288 GB) every model shares it. Each process logs
# to logs/lcb_recover_<job id>_<model>.out. recover.py skips the stages whose output exists, so a rerun resumes.
# Usage: recover.sh [--reverse | --test-helper] <model>[:<HF_HUB_CACHE>] ...   (helpers next to a running job, see
# recover.py; a model may be given several times, e.g. one --test-helper per GPU)
# e.g. recover.sh llama-3.1-8b:$PWD/logs/hf_cache qwen3-8b qwen3-32b:$PWD/logs/hf_cache gemma3-12b
set -u
extra=()
case "${1:-}" in --reverse|--test-helper) extra=("$1"); shift ;; esac
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
n=$(nvidia-smi -L | wc -l)
i=0 pids=() names=()
for spec in "$@"; do
    IFS=: read -r model cache <<< "$spec"
    env CUDA_VISIBLE_DEVICES=$((i % n)) LCB_MODEL=$model HF_HUB_CACHE=${cache:-$HF_HUB_CACHE} \
        python src/experiments/language_confusion/recover.py "${extra[@]}" > "logs/lcb_recover_${SLURM_JOB_ID:-local}_${model}_$i.out" 2>&1 &
    pids+=($!) names+=("$model")
    i=$((i + 1))
done
rc=0
for k in "${!pids[@]}"; do
    wait "${pids[$k]}" || { echo "${names[$k]} failed (exit $?)"; rc=1; }
done
exit $rc

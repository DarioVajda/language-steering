#!/bin/bash
# The FLORES-subset experiment (flores_subsets.py) inside one Slurm job: one GPU per model, each with its own
# number of workers, then each model's tables.
# Usage: flores_subsets.sh <model>:<workers per gpu>[:<HF_HUB_CACHE>[:<gpus>]] ...   (GPUs given out in order;
# one GPU per model unless <gpus> says more)
# e.g. flores_subsets.sh llama-3.1-8b:2:$PWD/logs/hf_cache qwen3-8b:2 qwen3-32b:1:$PWD/logs/hf_cache gemma3-12b:1
#      flores_subsets.sh qwen3-32b:1:$PWD/logs/hf_cache:2
# Options before the specs: --defer-to <job id>,... (passed to every worker; see flores_subsets.py) and --no-analyse
# (for a job that only helps another, which writes the tables).
set -u
cd "$(dirname "$0")/../../.."
G=src/experiments/mgsm_crosslingual
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
analyse=1 opts=()
while [[ $# -gt 0 && $1 == --* ]]; do
    case $1 in --no-analyse) analyse=0; shift ;; *) opts+=("$1" "$2"); shift 2 ;; esac
done
gpu=0
for spec in "$@"; do
    IFS=: read -r model workers cache gpus <<< "$spec"
    gpus=${gpus:-1}
    (
        export XL_MODEL=$model
        [ -n "$cache" ] && export HF_HUB_CACHE=$cache
        for g in $(seq $gpu $((gpu + gpus - 1))); do
            for w in $(seq "$workers"); do CUDA_VISIBLE_DEVICES=$g python $G/flores_subsets.py run "${opts[@]}" & done
        done
        wait
        [ $analyse = 1 ] && python $G/flores_subsets.py analyse
    ) > "logs/flores_subsets_${SLURM_JOB_ID:-local}_$model.out" 2>&1 &
    gpu=$((gpu + gpus))
done
wait

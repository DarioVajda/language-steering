#!/bin/bash
# The every-layer cells of the layers x positions ablation (recover.py with XL_ARM / LCB_ARM = every_steer and
# every_steer_prompt) on cross-lingual MGSM and LCB, for several models in one job over one or more nodes: run once
# per node (srun --ntasks-per-node 1); one worker per GPU. The first worker of the job runs every_layer_check.py and
# the others wait for it to pass. Each worker then takes one free LCB run (one model and arm; one process, claimed by
# a lock directory) and afterwards joins the MGSM workers, which share the dev and test units through recover.py's
# claim files, the prompt-only arm first. A worker stops once every MGSM analysis exists.
# Usage: every_layer.sh <model>[:<HF_HUB_CACHE>] ...
set -u
cd "$(dirname "$0")/../../.."
G=src/experiments/mgsm_crosslingual
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
LOCK=logs/every_layer_${SLURM_JOB_ID:-local}
mkdir -p "$LOCK"
host=$(hostname -s)
ARMS=(every_steer_prompt every_steer)
models=("$@")

if mkdir "$LOCK/check" 2>/dev/null; then
    IFS=: read -r m c <<< "${models[0]}"
    if env CUDA_VISIBLE_DEVICES=0 XL_MODEL=$m HF_HUB_CACHE=${c:-$HF_HUB_CACHE} python $G/every_layer_check.py \
            > "$LOCK/check.out" 2>&1; then touch "$LOCK/check.ok"; else touch "$LOCK/check.failed"; fi
fi
while [ ! -e "$LOCK/check.ok" ]; do
    [ -e "$LOCK/check.failed" ] && { echo "$host: every_layer_check.py failed, see $LOCK/check.out"; exit 1; }
    sleep 20
done

worker() {  # worker <gpu>
    local spec m c arm rc
    for spec in "${models[@]}"; do  # one LCB run, if any is free
        IFS=: read -r m c <<< "$spec"
        for arm in "${ARMS[@]}"; do
            if mkdir "$LOCK/lcb_${m}_$arm" 2>/dev/null; then
                env CUDA_VISIBLE_DEVICES=$1 LCB_MODEL=$m LCB_ARM=$arm HF_HUB_CACHE=${c:-$HF_HUB_CACHE} \
                    python src/experiments/language_confusion/recover.py > "$LOCK/lcb_${m}_$arm.out" 2>&1
                echo "LCB $m $arm: exit $?" >> "$LOCK/lcb_${m}_$arm.out"
                break 2
            fi
        done
    done
    local -A finished=()
    local n=$((${#models[@]} * ${#ARMS[@]}))
    while [ ${#finished[@]} -lt $n ]; do
        local progress=0
        for arm in "${ARMS[@]}"; do
            for spec in "${models[@]}"; do
                [ -n "${finished[$spec:$arm]:-}" ] && continue
                IFS=: read -r m c <<< "$spec"
                env CUDA_VISIBLE_DEVICES=$1 XL_MODEL=$m XL_ARM=$arm HF_HUB_CACHE=${c:-$HF_HUB_CACHE} \
                    python $G/recover.py work --main
                rc=$?
                case $rc in
                    0) finished[$spec:$arm]=1 ;;
                    2) progress=1; break 2 ;;  # back to the first arm and model
                    3) ;;
                    *) echo "recover.py work failed for $m $arm (exit $rc)"; sleep 60 ;;
                esac
            done
        done
        [ $progress = 0 ] && [ ${#finished[@]} -lt $n ] && sleep 60
    done
    echo "worker on $host GPU $1: done"
}

n=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((n - 1))); do
    worker $g > "$LOCK/${host}_gpu$g.out" 2>&1 &
done
wait

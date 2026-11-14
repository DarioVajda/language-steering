#!/bin/bash
# Run the whole experiment for one model inside one Slurm job: the pilot, every language with all-token and
# prompt-only steering, the universal runs of both, and the tables.
# Usage: run.sh <model> <gpus> <workers per gpu> [main job id]   (model: a key of MODELS in analyse.py)
# Workers take tasks (one generate.py run) from a claim queue, results/<model>/generations/claims/<main job id>/;
# a task whose output exists is skipped. The universal tasks of a mode wait until the dev grids of all six
# languages in that mode exist. With a main job id, the job is a helper on the same queue; the main job waits
# for every output and writes the tables. A job stopped by Slurm (time limit, cancel) releases the tasks it had
# claimed but not finished, so that another job takes them; a short job can therefore fill a gap in the queue.
set -u
cd "$(dirname "$0")/../../.."
G=src/experiments/language_confusion
export LCB_MODEL=$1
D=$G/results/$LCB_MODEL/generations
N=$2 W=$3 MAIN=${4:-$SLURM_JOB_ID}
C=$D/claims/$MAIN
LANGS="ja zh ru hi tr ar"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
mkdir -p "$C"

# Task list: "<name> <output> <generate.py arguments>"; the independent tasks come first.
tasks() {
    echo "pilot pilot_dev.jsonl --pilot"
    for L in $LANGS; do echo "$L ${L}_test.jsonl --lang $L"; echo "${L}_prompt ${L}_test_prompt.jsonl --lang $L --prompt-only"; done
    for L in $LANGS; do
        echo "${L}_universal ${L}_test_universal.jsonl --lang $L --universal"
        echo "${L}_universal_prompt ${L}_test_universal_prompt.jsonl --lang $L --universal --prompt-only"
    done
}
failed() { compgen -G "$C/*.failed" > /dev/null; }
await() {  # await <file>...: until every file exists; fails if any task failed
    for f in "$@"; do until [ -e "$f" ]; do failed && return 1; sleep 30; done; done
}

worker() {
    tasks | while read -r name out args; do
        [ -e "$D/$out" ] && continue
        mkdir "$C/$name" 2> /dev/null || continue  # claimed by another worker
        echo "$SLURM_JOB_ID $out" > "$C/$name/owner"
        if [[ $args == *--universal* ]]; then
            sfx=$([[ $args == *--prompt-only* ]] && echo _prompt)
            await $(for L in $LANGS; do echo "$D/${L}_dev$sfx.jsonl"; done) || return 1
        fi
        echo "[$(hostname) gpu $CUDA_VISIBLE_DEVICES] $name"
        python $G/generate.py $args > "$C/$name/log" 2>&1
        rc=$?
        if [ $rc -gt 128 ]; then rm -rf "$C/$name"; return 1; fi  # killed by a signal: released, not failed
        [ $rc -eq 0 ] || { touch "$C/$name.failed"; echo "$name failed, see $C/$name/log"; return 1; }
    done
}
release() {
    for o in "$C"/*/owner; do
        read -r job out < "$o"
        [ "$job" = "$SLURM_JOB_ID" ] && [ ! -e "$D/$out" ] && rm -rf "${o%/owner}" && echo "released ${o%/owner}"
    done
    exit 1
}
trap release TERM INT
for i in $(seq 0 $((N * W - 1))); do CUDA_VISIBLE_DEVICES=$((i / W)) worker & done
wait

[ "$MAIN" = "$SLURM_JOB_ID" ] || exit 0
await $(tasks | while read -r name out args; do echo "$D/$out"; done) || { echo "a task failed"; exit 1; }
python $G/analyse.py --pilot && python $G/analyse.py && python $G/analyse.py --prompt-only

#!/bin/bash
# The MMLU-ProX runs of the models after Llama-3.1-8B, one after another, inside one Slurm job: pilot.sh per model
# with the \boxed{X} answer format (--format boxed; see README, "Answer format"), one worker per GPU, the workers of
# every job running it sharing the tasks; then the analysis and the tests (tests.py), or with HELPER=1 neither, the
# helper moving on to the next model when nothing of this one is left to take. The settings of the Llama run: all
# 110 pairs, stage 1 to 2048 new tokens, the capped responses continued to 4096.
# (The runs reported were first made free-form and then rerun boxed, in jobs and a queue with these same options.)
# Usage: [HELPER=1] models.sh
set -u
cd "$(dirname "$0")"
OPTS="--format boxed --max-new 4096 --first-cap 2048"
fail=0
for m in qwen3-8b gemma3-12b qwen3-32b; do
    echo "[$(date '+%F %T')] $m"
    bash pilot.sh $m /shared/workspace/povejmo/huggingface_cache/hub $OPTS --pairs all \
        && { [ -n "${HELPER:-}" ] || (cd ../../.. && XL_MODEL=$m python src/experiments/mmlu_prox/tests.py $OPTS); } \
        || { echo "[$(date '+%F %T')] $m failed"; fail=1; }
done
exit $fail

#!/usr/bin/env bash
# A snapshot of the code, logs/code/<time>-<pid>/: src/ and slurm/ copied without outputs, results, data, logs and
# .venv linked to the real ones, so that editing the code does not change jobs that are queued or running.
# Prints its path. Used by submit.sh and enqueue.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/logs/code"
SNAP=$ROOT/logs/code/$(date +%Y%m%d-%H%M%S)-$$
mkdir "$SNAP"
OUT=(results states scratch)  # output directories inside src/ (written by jobs, shared with the real tree)
rsync -a --exclude __pycache__ "${OUT[@]/#/--exclude=}" "$ROOT/src" "$ROOT/slurm" "$SNAP/"
for d in $(cd "$ROOT/src" && find . -type d \( -name results -o -name states -o -name scratch \) -prune); do
    ln -s "$ROOT/src/${d#./}" "$SNAP/src/${d#./}"
done
for f in "$ROOT"/* "$ROOT"/.venv; do
    case ${f##*/} in src|slurm) ;; *) ln -s "$f" "$SNAP/${f##*/}" ;; esac
done
echo "$SNAP"

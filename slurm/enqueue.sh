#!/usr/bin/env bash
# Add a task to a queue job (queue.sh): <command> runs on every GPU of the job (CUDA_VISIBLE_DEVICES set to it), and
# <after>, if given, once when every GPU has finished <command>. Both run in a snapshot of the code taken now
# (snapshot.sh), with .venv first on PATH, inside the job's container and environment (submit.sh).
# Usage: slurm/enqueue.sh <queue> <name> <command> [<after>]
#   e.g. slurm/enqueue.sh logs/queue/b200 qwen3-32b-boxed 'python ... run ...' 'python ... analyse ...'
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[[ $# -ge 3 ]] || { echo "usage: $0 <queue> <name> <command> [<after>]" >&2; exit 1; }
Q=$1 name=$2
mkdir -p "$Q/tasks"
SNAP=$("$ROOT/slurm/snapshot.sh")
last=$(ls "$Q/tasks" | grep -oE '^[0-9]+' | sort -n | tail -1 || true)
n=$(printf '%03d-%s' $((10#${last:-0} + 1)) "$name")
write() {  # atomically, so that a worker never runs a part of it
    printf '#!/bin/bash\n# %s, enqueued %s; code: %s\nset -u\ncd %q && export PATH=%q:$PATH || exit 1\n%s\n' \
        "$n" "$(date '+%F %T')" "$SNAP" "$SNAP" "$ROOT/.venv/bin" "$2" > "$Q/tasks/.$1"
    mv "$Q/tasks/.$1" "$Q/tasks/$1"
}
[[ $# -ge 4 ]] && write "$n.after.sh" "$4"
write "$n.sh" "$3"
echo "$Q/tasks/$n.sh"

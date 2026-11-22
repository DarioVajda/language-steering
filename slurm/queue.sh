#!/bin/bash
# A job that runs a queue of tasks which can still be added to while it runs (enqueue.sh): it keeps its GPUs until
# the work is done, so that later tasks run in the same allocation. Every GPU of the job runs every task, in the
# order of their numbers, with CUDA_VISIBLE_DEVICES set to that GPU (the workers of a task share its work themselves,
# e.g. through pilot.py's claims); the after part of a task runs once, by the last worker to finish the task.
# The job ends when every worker has been idle for QUEUE_LINGER_MIN minutes (default 10).
#
#   <queue>/tasks/NNN-<name>.sh        a task (written by enqueue.sh; removing it before it starts cancels it)
#   <queue>/tasks/NNN-<name>.after.sh  its after part, if any
#   <queue>/logs/NNN-<name>_<host>-<gpu>.out, NNN-<name>_after.out
#   <queue>/state/<job id>/            exit codes and idle marks
#
# Usage (in a Slurm job, once per node, e.g. slurm/submit.sh --ntasks-per-node 1 ... -- bash slurm/queue.sh <queue>)
set -u
Q=$(realpath "$1")
S=$Q/state/${SLURM_JOB_ID:-local}
mkdir -p "$Q/tasks" "$Q/logs" "$S/idle"
LINGER=${QUEUE_LINGER_MIN:-10}
N=${QUEUE_WORKERS:-${SLURM_GPUS:-$(nvidia-smi -L | wc -l)}}  # the workers of the whole job, over all its nodes
host=$(hostname -s)
say() { echo "[$(date '+%F %T')] $*"; }

pending() {  # the tasks worker $1 has not run, in order
    local t n
    for t in "$Q"/tasks/[0-9]*.sh; do
        [[ -e $t && $t != *.after.sh ]] || continue
        n=$(basename "$t" .sh)
        [ -e "$S/$n/$1.rc" ] || echo "$n"
    done
}

worker() {
    local g=$1 id=$host-$1 n rc
    while true; do
        if [ -e "$S/closed" ]; then say "$id: leaving"; return; fi
        n=$(pending "$id" | head -1)
        if [ -n "$n" ]; then
            rm -f "$S/idle/$id"
            mkdir -p "$S/$n"
            say "$id: $n"
            CUDA_VISIBLE_DEVICES=$g QUEUE_WORKER=$id bash "$Q/tasks/$n.sh" >> "$Q/logs/${n}_$id.out" 2>&1
            rc=$?
            echo $rc > "$S/$n/$id.rc"
            say "$id: $n done (exit $rc)"
            if [ -e "$Q/tasks/$n.after.sh" ] && [ "$(ls "$S/$n" | grep -c '\.rc$')" -ge "$N" ] \
                    && mkdir "$S/$n/after" 2>/dev/null; then
                say "$id: $n after (exit codes: $(cat "$S/$n"/*.rc | xargs))"
                CUDA_VISIBLE_DEVICES=$g QUEUE_WORKER=$id bash "$Q/tasks/$n.after.sh" >> "$Q/logs/${n}_after.out" 2>&1
                rc=$?
                echo $rc > "$S/$n/after/rc"
                say "$id: $n after done (exit $rc)"
            fi
            continue
        fi
        [ -e "$S/idle/$id" ] || { touch "$S/idle/$id"; say "$id: idle"; }
        # every worker idle for LINGER minutes: the work is done, close the queue
        if [ "$(find "$S/idle" -type f -mmin +"$LINGER" | wc -l)" -ge "$N" ] && mkdir "$S/closed" 2>/dev/null; then
            say "$id: all $N workers idle for $LINGER min, closing the queue"
        fi
        sleep 30
    done
}

say "$host: queue $Q, $N workers in the job"
for g in $(seq 0 $(($(nvidia-smi -L | wc -l) - 1))); do worker "$g" & done
wait

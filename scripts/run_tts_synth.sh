#!/bin/bash
# Plan 14 M2 — launch the full Chatterbox synthesis across all 8 GPUs of vllab6.
#
# Runs N shards of `src.data.build_tts_audio synth` (one per GPU) detached via
# nohup. Each shard is resumable — re-running this script skips wavs that
# already exist, so an interrupted run just continues. Per-shard logs + pids
# go to $PLAN14_AUDIO/synth_logs/.
#
# After all shards finish:
#   <cbx_python> src/data/build_tts_audio.py manifest
set -euo pipefail

CBX_PY=${CBX_PY:-/tmp3/zhuoyuan/tts_pilot/envs/cbx/bin/python}
NSHARDS=${NSHARDS:-8}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOGDIR=${PLAN14_AUDIO:-/tmp3/zhuoyuan/plan14_audio}/synth_logs

mkdir -p "$LOGDIR"
: > "$LOGDIR/pids.txt"
cd "$REPO"

for i in $(seq 0 $((NSHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i nohup "$CBX_PY" src/data/build_tts_audio.py \
        synth --shard "$i" --num-shards "$NSHARDS" \
        > "$LOGDIR/shard${i}.log" 2>&1 &
    echo "$!" >> "$LOGDIR/pids.txt"
    echo "launched shard $i on GPU $i (pid $!)"
done

echo "all $NSHARDS shards launched; logs: $LOGDIR/shard*.log"

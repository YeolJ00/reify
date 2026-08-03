#!/bin/bash
# Keep the expand_neg regeneration going on a shared box.
#
# gen_expand.py skips clips that already exist, so a restart resumes rather than redoing
# work. This waits for a GPU with enough headroom before each attempt: the 14,850-char
# negative prompt needs materially more memory for text conditioning than the old stub,
# and the first launch OOM'd on a GPU another user was already on.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
NEED_MIB=${NEED_MIB:-36000}
LAB=outputs/scene/expand_neg
TARGET=168

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
    | awk -F'[ ,]+' -v need="$NEED_MIB" '{ if ($4-$2 >= need) { print $1; exit } }'
}

while true; do
  n=$(ls $LAB/vid_*.npz 2>/dev/null | wc -l)
  if [ "$n" -ge "$TARGET" ]; then echo "DONE: $n/$TARGET clips"; break; fi
  if pgrep -f 'gen_[e]xpand.py' >/dev/null; then sleep 120; continue; fi

  g=$(free_gpu)
  if [ -z "$g" ]; then
    echo "waiting for a GPU with ${NEED_MIB} MiB free (at $n/$TARGET)"
    sleep 300
    continue
  fi
  echo "starting on GPU $g at $n/$TARGET clips"
  CUDA_VISIBLE_DEVICES=$g HF_HOME=/home/nas5/jooyeolyun/hf_cache SEEDS=0,1,2,3,4,5 \
    LAB=$LAB PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/jooyeolyun/anaconda3/envs/cosmos/bin/python -u scripts/gen_expand.py \
    >> /tmp/expand_neg.log 2>&1
  sleep 30
done

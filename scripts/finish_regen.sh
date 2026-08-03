#!/bin/bash
# Runs the whole downstream pipeline once the expand_neg regeneration completes.
#
# Order matters: the fit must run with the corrected contact bound (cd search now capped
# at 200, the stable monotonic region), so any parameter it returns is reachable by a
# contact that does not create energy.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
PY=/home/jooyeolyun/anaconda3/envs/warp/bin/python
LABN=outputs/scene/expand_neg

while [ "$(ls $LABN/vid_*.npz 2>/dev/null | wc -l)" -lt 168 ]; do
  pgrep -f 'gen_[s]upervisor.sh' >/dev/null || { echo "SUPERVISOR GONE"; break; }
  sleep 180
done
echo "=== regen complete: $(ls $LABN/vid_*.npz | wc -l)/168 ==="

echo "=== fit: OLD lab (no negative prompt) ==="
$PY scripts/fit_expand.py 2>&1 | tail -18

echo "=== fit: NEW lab (with negative prompt), corrected contact bound ==="
LAB=$LABN $PY scripts/fit_expand.py 2>&1 | tail -18

echo "ALL DONE"

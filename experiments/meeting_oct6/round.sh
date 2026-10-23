#!/bin/bash
# One ~160 s round of every pending local experiment on 4 cores.
# Repeat until progress.py shows everything finished.
cd "$(dirname "$0")"
export RISLEY_SIGN_SWEEP=0  # keep local campaigns comparable to the 669k campaign
export LC_WORK=${LC_WORK:-$HOME/work/lc}
END=$(( $(date +%s) + ${ROUND_SECS:-160} ))
for k in 0 1 2 3; do (

  for job in "local_campaign.py run --exp ben --n 160" \
             "local_campaign.py run --exp ben066 --n 60" \
             "local_campaign.py run --exp box10 --n 150" \
             "local_campaign.py run --exp calib --n 150" \
             "nonunique.py run" \
             "sign_sweep_local.py run --work $HOME/work/ss"; do
    rem=$(( END - $(date +%s) ))
    [ $rem -gt 8 ] && timeout $rem python3 $job --shard $k --nshard 4 --budget $rem
  done ) &
done
wait

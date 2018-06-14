#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Expecting at least one argument (model path)."
fi

if [ $# -le 1 ]; then
  seed=0
else
  seed=$2
fi

if [ $# -le 2 ]; then
  traj_limitation=-1
else
  traj_limitation=$3
fi

python3 -m baselines.gail.vis_model \
  --env humanoid_CMU_run \
  --seed $seed \
  --expert_path data/cmu_mocap.npz \
  --load_model_path $1 \
  --traj_limitation $traj_limitation

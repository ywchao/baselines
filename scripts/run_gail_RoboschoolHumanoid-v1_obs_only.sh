#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Expecting at least one argument (expert path)."
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

time python3 -m baselines.gail.run_mujoco \
  --env RoboschoolHumanoid-v1 \
  --seed $seed \
  --expert_path $1 \
  --traj_limitation $traj_limitation \
  --num_timesteps 15000000 \
  --obs_only

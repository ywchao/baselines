#!/bin/bash

if [ $# -eq 0 ]; then
  seed=0
else
  seed=$1
fi

time python3 -m baselines.trpo_mpi.run_mujoco \
  --env RoboschoolHumanoid-v1 \
  --seed $seed \
  --num-timesteps 50000000

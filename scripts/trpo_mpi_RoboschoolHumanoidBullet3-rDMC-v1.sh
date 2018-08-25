#!/bin/bash

set -e

if [ $# -eq 0 ]; then
  seed=0
else
  seed=$1
fi

time python -m baselines.trpo_mpi.run_mujoco \
  --env=RoboschoolHumanoidBullet3-rDMC-v1 \
  --seed=$seed \
  --num-timesteps=60000000

OUT_DIR="output/trpo_mpi/RoboschoolHumanoidBullet3-rDMC-v1.seed_$seed.num-timesteps_6.00e+07"

time python -m baselines.trpo_mpi.vis_model \
  --env=RoboschoolHumanoidBullet3-rDMC-v1 \
  --seed=0 \
  --load-model-path=$OUT_DIR/checkpoints/model.ckpt-58500 \
  --render-mode=array \
  --vis-path=$OUT_DIR/vis_model.ckpt-58500.mp4

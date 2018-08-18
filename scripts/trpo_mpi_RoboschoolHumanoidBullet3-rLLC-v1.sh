#!/bin/bash

set -e

if [ $# -eq 0 ]; then
  seed=0
else
  seed=$1
fi

time python -m baselines.trpo_mpi.run_mujoco \
  --env RoboschoolHumanoidBullet3-rLLC-v1 \
  --seed $seed \
  --num-timesteps 100000000

OUT_DIR="output/trpo_mpi/RoboschoolHumanoidBullet3-rLLC-v1.seed_$seed.num-timesteps_1.00e+08"

time python -m baselines.trpo_mpi.vis_model \
  --env RoboschoolHumanoidBullet3-rLLC-v1 \
  --seed 0 \
  --load-model-path $OUT_DIR/checkpoints/model.ckpt-97600 \
  --render-mode array \
  --vis-file $OUT_DIR/vis_model.ckpt-97600.mp4

#!/bin/bash

set -e

if [ $# -eq 0 ]; then
  seed=0
else
  seed=$1
fi

time python -m baselines.trpo_mpi.run_mujoco \
  --env RoboschoolHumanoid-v1 \
  --seed $seed \
  --num-timesteps 40000000

OUT_DIR="output/trpo_mpi/RoboschoolHumanoid-v1.seed_$seed.num-timesteps_4.00e+07"

time python -m baselines.trpo_mpi.vis_model \
  --env RoboschoolHumanoid-v1 \
  --seed 0 \
  --load-model-path $OUT_DIR/checkpoints/model.ckpt-39000 \
  --render-mode array \
  --vis-file $OUT_DIR/vis_model.ckpt-39000.mp4

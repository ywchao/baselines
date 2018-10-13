#!/bin/bash

set -e

if [ $# -eq 0 ]; then
  seed=0
else
  seed=$1
fi

time python -m baselines.ppo2.run_roboschool \
  --env=RoboschoolHumanoidBullet3-rTurn-train-v1 \
  --seed=$seed \
  --num-timesteps=30000000 \
  --nsteps=8192 \
  --nminibatches=32 \
  --noptepochs=4 \
  --lr=1e-4

OUT_DIR="output/ppo2/RoboschoolHumanoidBullet3-rTurn-train-v1.seed_$seed.num-timesteps_3.00e+07"

time python -m baselines.ppo2.vis_model \
  --env=RoboschoolHumanoidBullet3-rTurn-v1 \
  --seed=0 \
  --load-model-path=$OUT_DIR/checkpoints/03600 \
  --render-mode=array \
  --vis-path=$OUT_DIR/vis_03600.mp4

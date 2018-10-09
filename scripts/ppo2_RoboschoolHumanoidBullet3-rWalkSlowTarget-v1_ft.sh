#!/bin/bash

set -e

if [ $# -eq 0 ]; then
  seed=0
else
  seed=$1
fi

time python -m baselines.ppo2.run_roboschool \
  --env=RoboschoolHumanoidBullet3-rWalkSlowTarget-train-v1 \
  --seed=$seed \
  --num-timesteps=120000000 \
  --nsteps=8192 \
  --nminibatches=32 \
  --noptepochs=4 \
  --lr=1e-4 \
  --load-model-path=output/ppo2/RoboschoolHumanoidBullet3-rWalkSlow-train-v1.seed_0.num-timesteps_8.00e+07/checkpoints/09700

OUT_DIR="output/ppo2/RoboschoolHumanoidBullet3-rWalkSlowTarget-train-v1-ft.seed_$seed.num-timesteps_1.20e+08"

time python -m baselines.ppo2.vis_model \
  --env=RoboschoolHumanoidBullet3-rWalkSlowTarget-v1 \
  --seed=0 \
  --load-model-path=$OUT_DIR/checkpoints/14600 \
  --render-mode=array \
  --vis-path=$OUT_DIR/vis_14600.mp4

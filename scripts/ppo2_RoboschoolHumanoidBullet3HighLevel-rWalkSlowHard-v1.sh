#!/bin/bash

set -e

if [ $# -eq 0 ]; then
  seed=0
else
  seed=$1
fi

time python -m baselines.ppo2.run_roboschool \
  --env=RoboschoolHumanoidBullet3HighLevel-rWalkSlowHard-train-v1 \
  --seed=$seed \
  --num-timesteps=1200000 \
  --nsteps=64 \
  --nminibatches=8 \
  --noptepochs=2 \
  --lr=1e-4 \
  --load_sub_paths="{
      'walk': 'output/ppo2/RoboschoolHumanoidBullet3-rWalkSlowTarget-train-v1-ft.seed_0.num-timesteps_1.20e+08/checkpoints/14600',
      }"

OUT_DIR="output/ppo2/RoboschoolHumanoidBullet3HighLevel-rWalkSlowHard-train-v1.seed_$seed.num-timesteps_1.20e+06"

time python -m baselines.ppo2.vis_model \
  --env=RoboschoolHumanoidBullet3HighLevel-rWalkSlowHard-v1 \
  --seed=0 \
  --load-model-path=$OUT_DIR/checkpoints/18700 \
  --render-mode=array \
  --vis-path=$OUT_DIR/vis_18700.mp4

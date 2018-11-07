#!/bin/bash

set -e

if [ $# -eq 0 ]; then
  seed=0
else
  seed=$1
fi

time python -m baselines.ppo2.run_roboschool \
  --env=RoboschoolHumanoidBullet3-rSitFromTurn-train-v1 \
  --seed=$seed \
  --num-timesteps=25000000 \
  --nsteps=8192 \
  --nminibatches=32 \
  --noptepochs=4 \
  --lr=1e-4 \
  --load-model-path=output/ppo2/RoboschoolHumanoidBullet3-rSit-train-v1.seed_$seed.num-timesteps_2.50e+07/checkpoints/03000

OUT_DIR="output/ppo2/RoboschoolHumanoidBullet3-rSitFromTurn-train-v1.seed_$seed.num-timesteps_2.50e+07"

time python -m baselines.ppo2.vis_model \
  --env=RoboschoolHumanoidBullet3-rSitFromTurn-v1 \
  --seed=0 \
  --load-model-path=$OUT_DIR/checkpoints/03000 \
  --render-mode=array \
  --vis-path=$OUT_DIR/vis_03000.mp4

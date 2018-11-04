#!/bin/bash

set -e

if [ $# -eq 0 ]; then
  seed=0
else
  seed=$1
fi

python -m baselines.ppo2.sample_roboschool \
  --env=RoboschoolHumanoidBullet3-rWalkSlowTarget-v1 \
  --seed=$seed \
  --name=walk_slow_target \
  --ntrajs=1000 \
  --nsteps=60 \
  --nlast=1 \
  --nsurv=60 \
  --load-model-path=output/ppo2/RoboschoolHumanoidBullet3-rWalkSlowTarget-train-v1-ft.seed_0.num-timesteps_1.20e+08/checkpoints/14600

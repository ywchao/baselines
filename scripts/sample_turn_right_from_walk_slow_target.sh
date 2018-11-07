#!/bin/bash

set -e

if [ $# -eq 0 ]; then
  seed=0
else
  seed=$1
fi

python -m baselines.ppo2.sample_roboschool \
  --env=RoboschoolHumanoidBullet3-rTurnRightFromWalkSlowTarget-v1 \
  --seed=$seed \
  --name=turn_right_from_walk_slow_target \
  --ntrajs=1000 \
  --nsteps=80 \
  --nlast=1 \
  --nsurv=60 \
  --load-model-path=output/ppo2/RoboschoolHumanoidBullet3-rTurnRightFromWalkSlowTarget-train-v1-ft.seed_0.num-timesteps_4.00e+07/checkpoints/04800

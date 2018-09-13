#!/bin/bash

set -e

if [ $# -eq 0 ]; then
  seed=0
else
  seed=$1
fi

time python -m baselines.ppo2.run_roboschool \
  --env=RoboschoolHumanoid-v1 \
  --seed=$seed \
  --num-timesteps=60000000 \
  --nsteps=4096 \
  --nminibatches=32 \
  --noptepochs=4 \
  --lr=1e-4

OUT_DIR="output/ppo2/RoboschoolHumanoid-v1.seed_$seed.num-timesteps_6.00e+07"

time python -m baselines.ppo2.vis_model \
  --env=RoboschoolHumanoid-v1 \
  --seed=0 \
  --load-model-path=$OUT_DIR/checkpoints/model.ckpt-14600 \
  --render-mode=array \
  --vis-path=$OUT_DIR/vis_model.ckpt-14600.mp4

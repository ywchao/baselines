#!/bin/bash

time python3 -m baselines.trpo_mpi.run_mujoco \
  --env RoboschoolHumanoid-v1 \
  --task sample \
  --load-model-path $1

#!/bin/bash

# walk
for i in $(seq -f "%02g" 1 11); do
  echo "processing 08/08_$i.bvh ... "

  python motion_retarget/retarget_cmu.py \
    --src_bvh data/cmu_mocap_bvh/08/08_$i.bvh \
    --out_bvh data/cmu_mocap_bvh_retarget/08/08_$i.bvh
done

# turn
echo "processing 69/69_13.bvh ... "

python motion_retarget/retarget_cmu.py \
  --src_bvh data/cmu_mocap_bvh/69/69_13.bvh \
  --out_bvh data/cmu_mocap_bvh_retarget/69/69_13.bvh

# sit
echo "processing 143/143_18.bvh ... "

python motion_retarget/retarget_cmu.py \
  --src_bvh data/cmu_mocap_bvh/143/143_18.bvh \
  --out_bvh data/cmu_mocap_bvh_retarget/143/143_18.bvh


python motion_retarget/collect_cmu.py

python motion_retarget/vis_cmu.py

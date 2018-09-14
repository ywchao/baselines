import os
import numpy as np

from dm_control.rl.control import flatten_observation


def get_humanoid_cmu_obs(env):
    # Assumes env is an instance of `HumanoidCMU`.
    obs = env.task.get_observation(env.physics)
    # Add head to extremities
    torso_frame = env.physics.named.data.xmat['thorax'].reshape(3, 3)
    torso_pos = env.physics.named.data.xpos['thorax']
    torso_to_head = env.physics.named.data.xpos['head'] - torso_pos
    positions = torso_to_head.dot(torso_frame)
    obs['extremities'] = np.hstack((obs['extremities'], positions))
    # Remove joint_angles, head_height, velocity
    del obs['joint_angles']
    del obs['head_height']
    del obs['velocity']
    # Flatten observation
    return flatten_observation(obs)['observations']

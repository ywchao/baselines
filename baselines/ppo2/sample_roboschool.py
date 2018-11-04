#!/usr/bin/env python3
import numpy as np
import os
from baselines.common.cmd_util import mujoco_arg_parser, parse_unknown_args
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines import bench
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
from baselines.ppo2.run_roboschool import parse
import gym
import tensorflow as tf
import roboschool


def main():
    args, unknown_args = mujoco_arg_parser().parse_known_args()
    extra_args = {k: parse(v) for k,v in parse_unknown_args(unknown_args).items()}

    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env():
        env = gym.make(args.env)
        env = bench.Monitor(env, None, allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(args.seed)

    ob_space = env.observation_space
    ac_space = env.action_space

    policy = MlpPolicy
    make_model = lambda : ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
    	            nbatch_act=1, nbatch_train=64,
                    nsteps=2048, ent_coef=0.0, vf_coef=0.5,
                    max_grad_norm=0.5, ob_rms=env.ob_rms, ret_rms=env.ret_rms)
    model = make_model()
    model.load(args.load_model_path)

    print('sampling model ... ')

    def get_qpos(env):
        env_ = env.venv.envs[0].env.env
        return np.concatenate(
            [np.array([j.current_position() for j in env_.ordered_joints], dtype=np.float).flatten()] + 
            [np.array(env_.robot_body.pose().xyz(), dtype=np.float)] + 
            [np.array(env_.robot_body.pose().rpy(), dtype=np.float)] +
            [np.array(env_.robot_body.speed(), dtype=np.float)]
        )

    def traj_segment_generator(model, env, name, ntrajs, nsteps, nlast, nsurv):
        all_qpos = []
        obs = env.reset()
        qpos = []
        while True:
            actions = model.step(obs)[0]
            obs, _, dones, _ = env.step(actions)
            if dones[0]:
                qpos = []
            else:
	            qpos.append(get_qpos(env))
            if len(qpos) == nsteps + nsurv:
                all_qpos.append(qpos[:-nsurv])
                obs = env.reset()
                qpos = []
            if len(all_qpos) == ntrajs:
                break
        
        all_qpos = np.array(all_qpos)
        all_qpos = all_qpos[:,-nlast:,:]

        save_path = 'data/sample_{}'.format(name)
        if not os.path.isfile(save_path):
            np.savez(save_path, qpos=all_qpos)

    seg_gen = traj_segment_generator(model, env, **extra_args)

    print('done.')


if __name__ == '__main__':
    main()

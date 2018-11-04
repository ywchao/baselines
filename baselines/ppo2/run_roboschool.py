#!/usr/bin/env python3
import numpy as np
from baselines.common.cmd_util import mujoco_arg_parser, parse_unknown_args
from baselines import bench, logger
import os
import roboschool


def train(env_id, num_timesteps, seed, save_per_iter, load_model_path,
          nsteps, nminibatches, noptepochs, lr, load_sub_paths=None):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy, HierarchicalMlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env])
    if isinstance(env.action_space, gym.spaces.Dict):
        env = VecNormalize(env, skip_rms=('switch',))
    else:
        env = VecNormalize(env)

    set_global_seeds(seed)
    if isinstance(env.action_space, gym.spaces.Dict):
        policy = HierarchicalMlpPolicy
    else:
        policy = MlpPolicy

    model = ppo2.learn(policy=policy, env=env, nsteps=nsteps, nminibatches=nminibatches,
                       lam=0.95, gamma=0.99, noptepochs=noptepochs, log_interval=1,
                       ent_coef=0.0,
                       lr=lr,
                       cliprange=0.2,
                       save_interval=100,
                       total_timesteps=num_timesteps,
                       load_path=load_model_path,
                       load_sub_paths=load_sub_paths)

    return model, env


def get_out_dir(args):
    dir_name = args.env
    if args.load_model_path is not None:
        dir_name += "-ft"
    dir_name += ".seed_" + str(args.seed)
    dir_name += ".num-timesteps_" + "{0:.2e}".format(args.num_timesteps)
    return os.path.join(args.out_base, 'ppo2', dir_name)


def parse(v):
    assert isinstance(v, str)
    try:
        return eval(v)
    except (NameError, SyntaxError):
        return v


def main():
    args, unknown_args = mujoco_arg_parser().parse_known_args()
    extra_args = {k: parse(v) for k,v in parse_unknown_args(unknown_args).items()}
    logger.configure(dir=get_out_dir(args))
    model, env = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        save_per_iter=args.save_per_iter, load_model_path=args.load_model_path, **extra_args)


if __name__ == '__main__':
    main()

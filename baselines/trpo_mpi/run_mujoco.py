#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi

import os

try:
    import roboschool
except ImportError as e:
    print("{}. You will not be able to run the experiments that require Roboschool envs.".format(e))

def train(env_id, task, num_timesteps, seed, out_dir, save_per_iter,
          load_model_path, sample_stochastic):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure(dir=out_dir)
    else:
        logger.configure(dir=out_dir, format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)
    env = make_mujoco_env(env_id, workerseed)
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    log_dir = logger.get_dir()
    if task == 'train':
        trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
            max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3,
            save_per_iter=save_per_iter, ckpt_dir=checkpoint_dir, log_dir=log_dir)
    elif task == 'sample':
        trpo_mpi.sample(env, policy_fn, timesteps_per_batch=1024, load_model_path=load_model_path,
            sample_stochastic=sample_stochastic, max_sample_traj=1500, log_dir=log_dir)
    else:
        raise NotImplementedError
    env.close()

def get_out_dir(args):
    if args.task == 'train':
        dir_name = "trpo." + args.env
        dir_name += ".seed_" + str(args.seed)
        dir_name += ".num-timesteps_" + "{0:.2e}".format(args.num_timesteps)
        dir_name = os.path.join('trpo_mpi', dir_name)
    elif args.task == 'sample':
        dir_name = [s for s in args.load_model_path.split('/') if args.env in s][0]
        dir_name = os.path.join('trpo_mpi_sample', dir_name)
    else:
        raise NotImplementedError

    return os.path.join(args.out_base, dir_name)

def main():
    args = mujoco_arg_parser().parse_args()
    train(args.env, task=args.task, num_timesteps=args.num_timesteps, seed=args.seed,
        out_dir=get_out_dir(args), save_per_iter=args.save_per_iter, load_model_path=args.load_model_path,
        sample_stochastic=args.sample_stochastic)


if __name__ == '__main__':
    main()


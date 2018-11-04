#!/usr/bin/env python3
from baselines.common.cmd_util import mujoco_arg_parser
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines import bench
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy, HierarchicalMlpPolicy
import gym
import tensorflow as tf
import roboschool


def main():
    args = mujoco_arg_parser().parse_args()

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
    if isinstance(env.action_space, gym.spaces.Dict):
        env = VecNormalize(env, skip_rms=('switch',))
    else:
        env = VecNormalize(env)

    set_global_seeds(args.seed)

    ob_space = env.observation_space
    ac_space = env.action_space

    if isinstance(env.action_space, gym.spaces.Dict):
        policy = HierarchicalMlpPolicy
    else:
        policy = MlpPolicy
    make_model = lambda : ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
    	            nbatch_act=1, nbatch_train=64,
                    nsteps=2048, ent_coef=0.0, vf_coef=0.5,
                    max_grad_norm=0.5, ob_rms=env.ob_rms, ret_rms=env.ret_rms)
    model = make_model()
    model.load(args.load_model_path)

    print('visualizing model ... ')

    def traj_segment_generator_human(model, env):
        obs = env.reset()
        while True:
            if isinstance(env.action_space, gym.spaces.Dict):
                actions = model.step(obs)[4]
                obs = env.step([s for s in zip(*actions)])[0]
            else:
                actions = model.step(obs)[0]
                obs = env.step(actions)[0]
            env.render()

    def traj_segment_generator_array(model, env, save_file):
        import matplotlib.pyplot as plt
        import matplotlib.animation as manimation
        import os

        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=60)
        fig = plt.figure(
            figsize=(env.venv.envs[0].env.env.VIDEO_W/100, env.venv.envs[0].env.env.VIDEO_H/100))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if os.path.isfile(save_file):
            os.remove(save_file)

        with writer.saving(fig, save_file, dpi=100):
            # Run 10 episodes
            video = []
            for i in range(10):
                print('episode {:02d}/{:02d}'.format(i+1,10))
                obs = env.reset()
                while True:
                    video.append(env.venv.envs[0].render("rgb_array"))
                    if isinstance(env.action_space, gym.spaces.Dict):
                        actions = model.step(obs)[4]
                        obs, _, dones, _ = env.step([s for s in zip(*actions)])
                    else:
                        actions = model.step(obs)[0]
                        obs, _, dones, _  = env.step(actions)
                    # Cannot run extra steps after done due to auto reset in VecNormalize
                    if dones[0]:
                        break
            print('saving to file: ' + save_file)
            for i in range(len(video)):
                if i == 0:
                  img = plt.imshow(video[0])
                  plt.axis('tight')
                else:
                  img.set_data(video[i])
                writer.grab_frame()

    if args.render_mode == 'human':
        seg_gen = traj_segment_generator_human(model, env)
    if args.render_mode == 'array':
        seg_gen = traj_segment_generator_array(model, env, save_file=args.vis_path)


if __name__ == '__main__':
    main()

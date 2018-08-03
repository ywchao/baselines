#!/usr/bin/env python3
from mpi4py import MPI
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.ppo1.mlp_policy import MlpPolicy
import baselines.common.tf_util as U
import roboschool

def main():
    args = mujoco_arg_parser().parse_args()

    sess = U.single_threaded_session()
    sess.__enter__()

    workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
    env = make_mujoco_env(args.env, workerseed)

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)

    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)

    U.initialize()
    U.load_state(args.load_model_path)

    print('visualizing model ... ')

    def traj_segment_generator_human(pi, env, stochastic):
        ob = env.reset()
        while True:
            ac, vpred = pi.act(stochastic, ob)
            ob, _, new, _ = env.step(ac)
            env.render()
            if new:
                ob = env.reset()

    def traj_segment_generator_array(pi, env, stochastic, save_file):
        import matplotlib.pyplot as plt
        import matplotlib.animation as manimation
        import os

        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=60)
        fig = plt.figure(
            figsize=(env.env.env.VIDEO_W/100, env.env.env.VIDEO_H/100))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if os.path.isfile(save_file):
            os.remove(save_file)

        with writer.saving(fig, save_file, dpi=100):
            # Run 10 episodes
            video = []
            for i in range(10):
                print('episode {:02d}/{:02d}'.format(i+1,10))
                ob = env.reset()
                cnt_done = None
                while True:
                    video.append(env.render("rgb_array"))
                    ac, vpred = pi.act(stochastic, ob)
                    ob, _, new, _ = env.step(ac)
                    # Run 50 more steps after done
                    if cnt_done is None and new:
                        cnt_done = 0
                    if cnt_done is not None:
                        cnt_done += 1
                    if cnt_done == 50 or cnt_done == env.spec.timestep_limit:
                        cnt_done = None
                        break
                    if new:
                        env.needs_reset = False
            print('saving to file: ' + save_file)
            for i in range(len(video)):
                if i == 0:
                  img = plt.imshow(video[0])
                  plt.axis('tight')
                else:
                  img.set_data(video[i])
                writer.grab_frame()

    if args.render_mode == 'human':
        seg_gen = traj_segment_generator_human(pi, env, stochastic=True)
    if args.render_mode == 'array':
        seg_gen = traj_segment_generator_array(
            pi, env, stochastic=True, save_file=args.vis_path)

if __name__ == '__main__':
    main()

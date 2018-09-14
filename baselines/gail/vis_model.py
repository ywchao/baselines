import os
import types
import numpy as np
import gym

from baselines.gail.run_mujoco import argsparser
from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
from baselines import logger

from dm_control.suite import humanoid_CMU
from baselines.common.dm_control_util import get_humanoid_cmu_obs


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    if args.env_id == 'humanoid_CMU_run':
        env = humanoid_CMU.run()
        env.task.random.seed(args.seed)
        # Load data for initialization
        print('loading data for initialization ...')
        from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
        dataset = Mujoco_Dset(expert_path=args.expert_path,
                              traj_limitation=args.traj_limitation,
                              obs_only=True)
        _, qpos = dataset.get_next_batch(dataset.num_transition)
    else:
        env = gym.make(args.env_id)
        env = bench.Monitor(env, logger.get_dir() and
                            os.path.join(logger.get_dir(), "monitor.json"))
        env.seed(args.seed)

    if args.env_id != 'humanoid_CMU_run' and hasattr(env.env.env, 'viewer_setup'):
        def viewer_setup(self):
            from mujoco_py.generated import const
            self.viewer.cam.type = const.CAMERA_TRACKING
            self.viewer.cam.trackbodyid = 0
            self.viewer.cam.distance = self.model.stat.extent * 2.0
            self.viewer.cam.elevation = -20

        env.env.env.viewer_setup = types.MethodType(viewer_setup, env.env.env)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)

    if args.env_id == 'humanoid_CMU_run':
        from gym import spaces
        obs_dim = get_humanoid_cmu_obs(env).shape
        ob_space = spaces.Box(np.inf*np.ones(obs_dim)*(-1),
                              np.inf*np.ones(obs_dim))
        ac_space = spaces.Box(env.action_spec().minimum,
                              env.action_spec().maximum)
    else:
        ob_space = env.observation_space
        ac_space = env.action_space

    pi = policy_fn("pi", ob_space, ac_space)

    U.initialize()
    U.load_state(args.load_model_path)

    print('visualizing model ... ')

    def traj_segment_generator(pi, env, stochastic):
        ob = env.reset()
        while True:
            ac, vpred = pi.act(stochastic, ob)
            ob, _, new, _ = env.step(ac)
            env.render()
            if new:
                ob = env.reset()

    def traj_segment_generator_cmu(pi, env, stochastic, save_file):
        import matplotlib.pyplot as plt
        import matplotlib.animation as manimation

        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=50)
        fig = plt.figure(figsize=(9.6, 4.8))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if os.path.isfile(save_file):
            os.remove(save_file)

        with writer.saving(fig, save_file, dpi=100):
            # Run 10 episodes
            video = []
            for i in range(10):
                print('episode {:02d}/{:02d}'.format(i+1,10))
                env.reset()
                t = np.random.randint(qpos.shape[0])
                p_t = qpos[t, :63]
                v_t = qpos[t, 63:]
                with env.physics.reset_context():
                    env.physics.data.qpos[:] = p_t
                    env.physics.data.qvel[:] = v_t
                ob = get_humanoid_cmu_obs(env)
                cnt_done = None
                while True:
                    video.append(np.hstack([env.physics.render(480, 480, camera_id=0),
                                            env.physics.render(480, 480, camera_id=1)]))
                    ac, vpred = pi.act(stochastic, ob)
                    step_type, _, _, ob = env.step(ac)
                    # Run 50 more steps after done
                    if cnt_done is None and ob['head_height'] < 1.0:
                        cnt_done = 0
                    if cnt_done is not None:
                        cnt_done += 1
                    if cnt_done == 50 or step_type == 2:
                        cnt_done = None
                        break
                    ob = get_humanoid_cmu_obs(env)
            print('saving to file: ' + save_file)
            for i in range(len(video)):
                if i == 0:
                  img = plt.imshow(video[0])
                  plt.axis('tight')
                else:
                  img.set_data(video[i])
                writer.grab_frame()

    if args.env_id == 'humanoid_CMU_run':
        seg_gen = traj_segment_generator_cmu(pi, env, stochastic=True, save_file=args.vis_path)
    else:
        seg_gen = traj_segment_generator(pi, env, stochastic=True)

    print('done.')


if __name__ == '__main__':
    args = argsparser()
    main(args)

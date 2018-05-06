import gym
import types

from baselines.gail.run_mujoco import argsparser
from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
from baselines import logger

try:
    import roboschool
except ImportError as e:
    print("{}. You will not be able to run the experiments that require Roboschool envs.".format(e))

def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)

    if hasattr(env.env.env, 'viewer_setup'):
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

    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)

    U.initialize()
    U.load_state(args.load_model_path)

    def traj_segment_generator(pi, env, stochastic):
        # Initialize state variables
        new = True
        ob = env.reset()
        while True:
            ac, vpred = pi.act(stochastic, ob)
            ob, _, new, _ = env.step(ac)
            env.render()
            if new:
                ob = env.reset()

    seg_gen = traj_segment_generator(pi, env, stochastic=True)


if __name__ == '__main__':
    args = argsparser()
    main(args)

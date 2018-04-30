#!/usr/bin/env python3
from mpi4py import MPI
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.ppo1.mlp_policy import MlpPolicy
import baselines.common.tf_util as U
import types

try:
    import roboschool
except ImportError as e:
    print("{}. You will not be able to run the experiments that require Roboschool envs.".format(e))

def main():
    args = mujoco_arg_parser().parse_args()

    sess = U.single_threaded_session()
    sess.__enter__()

    workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
    env = make_mujoco_env(args.env, workerseed)

    if hasattr(env.env.env, 'viewer_setup'):
        def viewer_setup(self):
            from mujoco_py.generated import const
            self.viewer.cam.type = const.CAMERA_TRACKING
            self.viewer.cam.trackbodyid = 0
            self.viewer.cam.distance = self.model.stat.extent * 2.0
            self.viewer.cam.elevation = -20

        env.env.env.viewer_setup = types.MethodType(viewer_setup, env.env.env)

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)

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
    main()

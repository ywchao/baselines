import argparse
import gym
import types
import numpy as np
import time


def argsparser():
    parser = argparse.ArgumentParser("Visualize expert data")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    return parser.parse_args()

def main(args):
    env = gym.make(args.env_id)

    def viewer_setup(self):
        from mujoco_py.generated import const
        self.viewer.cam.type = const.CAMERA_TRACKING
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 2.0
        self.viewer.cam.elevation = -20
        
    env.env.viewer_setup = types.MethodType(viewer_setup, env.env)

    if args.env_id == 'Hopper-v2':
        data = np.load('data/deterministic.trpo.Hopper.0.00.npz')
        tidx = 0
        def set_qpos(env, d):
            env.env.sim.data.qpos[0] = i * 0.01
            env.env.sim.data.qpos[1:] = d[0:5]
    elif args.env_id == 'Humanoid-v2':
        data = np.load('data/deterministic.trpo.Humanoid.0.00.npz')
        tidx = 1168
        def set_qpos(env, d):
            env.env.sim.data.qpos[0] = i * 0.01
            env.env.sim.data.qpos[1] = 0
            env.env.sim.data.qpos[2:] = d[0:22]
    else:
        raise NotImplementedError

    d = data['obs'][tidx].copy()
    for i in range(d.shape[0]):
        set_qpos(env, d[i])
        env.env.sim.forward()
        env.render("human")
        time.sleep(0.01)


if __name__ == '__main__':
    args = argsparser()
    main(args)

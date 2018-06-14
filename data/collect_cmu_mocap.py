import argparse
import os
import numpy as np

try:
    os.environ['DISABLE_MUJOCO_RENDERING'] = '1'
    from dm_control.suite import humanoid_CMU
    from dm_control.suite.utils import parse_amc
    from baselines.common.dm_control_util import get_humanoid_cmu_obs
except ImportError as e:
    print("{}. To run collect_cmu_mocap.py, please first install dm_control.".format(e))
    raise

_SUBJECT_ID = 8
_SUBJECT_AMC_ID = [1,2,3,4,5,6,7,8,9,10,11]


def argsparser():
    parser = argparse.ArgumentParser("Collect CMU MoCap data")
    parser.add_argument('--data_path', help='path to the `all_asfamc` folder', default=None)
    parser.add_argument('--save_file', help='file to save the output', default='data/cmu_mocap.npz')
    return parser.parse_args()

def main(args):
    env = humanoid_CMU.run()

    print('collecting cmu mocap data ... ')

    assert args.data_path is not None
    amc_qpos = []
    amc_obs = []
    for i in _SUBJECT_AMC_ID:
        file_path = os.path.join(args.data_path,
                                 'subjects','{:02d}'.format(_SUBJECT_ID),
                                 '{:02d}_{:02d}.amc'.format(_SUBJECT_ID,i))
        converted = parse_amc.convert(
            file_path, env.physics, env.control_timestep())

        qpos_seq = []
        obs_seq = []
        for t in range(converted.qpos.shape[-1] - 1):
            p_t = converted.qpos[:, t + 1]
            v_t = converted.qvel[:, t]
            qpos_seq.append(np.concatenate((p_t, v_t))[None])
            with env.physics.reset_context():
                env.physics.data.qpos[:] = p_t
                env.physics.data.qvel[:] = v_t
            obs = get_humanoid_cmu_obs(env)
            obs_seq.append(obs[None])

        amc_qpos.append(np.concatenate(qpos_seq))
        amc_obs.append(np.concatenate(obs_seq))

    if not os.path.isfile(args.save_file):
        np.savez(args.save_file, obs=amc_obs, qpos=amc_qpos)

    print('done.')


if __name__ == '__main__':
    args = argsparser()
    main(args)

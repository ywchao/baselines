import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

try:
    from dm_control.suite import humanoid_CMU
    from dm_control.suite.utils import parse_amc
except ImportError as e:
    print("{}. To run vis_cmu_mocap.py, please first install dm_control.".format(e))
    raise

_SUBJECT_ID = 8
_SUBJECT_AMC_ID = [1,2,3,4,5,6,7,8,9,10,11]


def argsparser():
    parser = argparse.ArgumentParser("Visualize CMU MoCap data")
    parser.add_argument('--data_path', help='path to the `all_asfamc` folder', default=None)
    parser.add_argument('--save_path', help='path to save the output', default='data/cmu_mocap_vis')
    return parser.parse_args()

def main(args):
    env = humanoid_CMU.run()

    print('visualizing cmu mocap data ... ')

    assert args.data_path is not None

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=50)

    fig = plt.figure()

    for i in _SUBJECT_AMC_ID:
        file_path = os.path.join(args.data_path,
                                 'subjects','{:02d}'.format(_SUBJECT_ID),
                                 '{:02d}_{:02d}.amc'.format(_SUBJECT_ID,i))
        converted = parse_amc.convert(
            file_path, env.physics, env.control_timestep())


        save_file = os.path.join(args.save_path,'{:02d}_{:02d}.mp4'.format(_SUBJECT_ID,i))
        if os.path.exists(save_file):
            continue

        with writer.saving(fig, save_file, dpi=100):
            video = []
            for i in range(converted.qpos.shape[1] - 1):
                p_i = converted.qpos[:, i + 1]
                v_i = converted.qvel[:, i]
                with env.physics.reset_context():
                    env.physics.data.qpos[:] = p_i
                    env.physics.data.qvel[:] = v_i
                video.append(np.hstack([env.physics.render(480, 480, camera_id=0),
                                        env.physics.render(480, 480, camera_id=1)]))
            for i in range(len(video)):
                if i == 0:
                    img = plt.imshow(video[0])
                else:
                    img.set_data(video[i])
                writer.grab_frame()

    print('done.')


if __name__ == '__main__':
    args = argsparser()
    main(args)

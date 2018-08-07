import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

import gym, roboschool

_SUBJECT_ID = 8
_SUBJECT_AMC_ID = [1,2,3,4,5,6,7,8,9,10,11]


def argsparser():
    parser = argparse.ArgumentParser("Visualize CMU MoCap data")
    parser.add_argument('--data_path', help='path to the npz file', default='data/cmu_mocap.npz')
    parser.add_argument('--save_path', help='path to save the output', default='data/cmu_mocap_vis')
    return parser.parse_args()

def main(args):
    env = gym.make("RoboschoolHumanoidBullet3-v1")
    env.reset()

    print('visualizing cmu mocap data ... ')

    assert args.data_path is not None

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=60)
    fig = plt.figure(
        figsize=(env.env.VIDEO_W/100, env.env.VIDEO_H/100))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    data = np.load('data/cmu_mocap.npz')
    assert len(data['qpos']) == len(_SUBJECT_AMC_ID)

    for i in range(len(_SUBJECT_AMC_ID)):
        qpos = data['qpos'][i]

        save_file = os.path.join(args.save_path,'{:02d}_{:02d}.mp4'.format(_SUBJECT_ID,_SUBJECT_AMC_ID[i]))
        if os.path.exists(save_file):
            continue

        with writer.saving(fig, save_file, dpi=100):
            video = []
            for t in range(len(qpos)):
                for j, joint in enumerate(env.env.ordered_joints):
                    joint.reset_current_position(qpos[t, 2*j], qpos[t, 2*j+1])
                cpose = roboschool.scene_abstract.cpp_household.Pose()
                cpose.set_xyz(*qpos[t, -9:-6])
                cpose.set_rpy(*qpos[t, -6:-3])
                env.env.cpp_robot.set_pose_and_speed(cpose, *qpos[t, -3:])
                for r in env.env.mjcf:
                    r.query_position()
                video.append(env.render("rgb_array"))
            for t in range(len(video)):
                if t == 0:
                    img = plt.imshow(video[0])
                    plt.axis('tight')
                else:
                    img.set_data(video[t])
                writer.grab_frame()

    print('done.')


if __name__ == '__main__':
    args = argsparser()
    main(args)

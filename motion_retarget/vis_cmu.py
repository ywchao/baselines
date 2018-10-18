import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

import gym, roboschool

_SUBJECT_ID = {
    'walk': 8,
    'turn': 69,
    'sit': 143,
}
_SUBJECT_AMC_ID = {
    'walk': [1,2,3,4,5,6,7,8,9,10,11],
    'turn': [13],
    'sit': [18],
}

_SEG_KEYS = {
    'walk': ("rstep", "lstep"),
    'turn': ("rturn", "lturn"),
    'sit': ("sitd",),
}


def argsparser():
    parser = argparse.ArgumentParser("Visualize CMU MoCap data")
    parser.add_argument('--save_path', help='path to save the output', default='data/cmu_mocap_vis')
    return parser.parse_args()

def main(args):
    env = gym.make("RoboschoolHumanoidBullet3-v1")
    env.reset()

    print('visualizing cmu mocap data ... ')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=60)
    fig = plt.figure(
        figsize=(env.env.VIDEO_W/100, env.env.VIDEO_H/100))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    for task in ("walk", "turn", "sit"):
        data = np.load('data/cmu_mocap_{}.npz'.format(task))
        assert len(data['qpos']) == len(_SUBJECT_AMC_ID[task])

        for i in range(len(_SUBJECT_AMC_ID[task])):
            qpos = data['qpos'][i]

            save_file = os.path.join(
                args.save_path,'{}_{:02d}_{:02d}.mp4'.format(task,_SUBJECT_ID[task],_SUBJECT_AMC_ID[task][i]))
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

        for k in _SEG_KEYS[task]:
            save_file = os.path.join(args.save_path,'{}_{:02d}_{}.mp4'.format(task,_SUBJECT_ID[task],k))
            if os.path.exists(save_file):
                continue

            with writer.saving(fig, save_file, dpi=100):
                video = []
                for i, s in enumerate(data[k]):
                    qpos = data['qpos'][s[0]].copy()
                    for t in range(s[1], s[1] + s[2] + 1):
                        if task == "sit":
                            qpos[t, -4] += np.pi/2
                            qpos[t, -9:-7] = qpos[t, -9:-7].dot(np.array([[0.0, 1.0],[-1.0, 0.0]], dtype=np.float32))
                            qpos[t, -3:-1] = qpos[t, -3:-1].dot(np.array([[0.0, 1.0],[-1.0, 0.0]], dtype=np.float32))
                            qpos[t, -9] -= data['offsetx'][0][i][t-s[1]]
                            qpos[t, -7] -= data['offsetz'][0][i][t-s[1]]
                            if t > s[1]:
                                qpos[t, -3] = (qpos[t, -9] - qpos[t-1, -9]) / 0.0165
                                qpos[t, -1] = (qpos[t, -7] - qpos[t-1, -7]) / 0.0165
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

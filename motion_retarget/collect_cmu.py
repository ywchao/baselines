import argparse
import os
import sys
import numpy as np
import gym
import roboschool

from scipy import interpolate

from roboschool.scene_abstract import cpp_household

sys.path.append('motion_retarget/motionsynth')
import BVH as BVH

timestep = 0.0165  # timestep of RoboschoolForwardWalker

_SUBJECT_ID = {
    'walk': 8,
    'turn': 69,
    'sit': 143,
    'holistic': 13,
}
_SUBJECT_AMC_ID = {
    'walk': [1,2,3,4,5,6,7,8,9,10,11],
    'turn': [13],
    'sit': [18],
    'holistic': [2],
}

# Since this is a heuristic algorithm, we need to manually clean up the output
# by adding and removing indices. The keys here are indices, not amc ids.
_ADD_LIST = {
    'walk': {},
    'sit': {},
}
_DEL_LIST = {
    'walk': {},
    'sit': {},
}

# Manual segmentations
_L_TURN = np.array([[0,  213,  88], [0,  726,  79], [0, 1234,  87], [0, 1784,  96]], dtype=np.int64)
_R_TURN = np.array([[0,  444, 101], [0,  948, 109], [0, 1504,  97], [0, 2055, 110]], dtype=np.int64)
_HOLIST = np.array([[0,  0,  354]], dtype=np.int64)


def argsparser():
    parser = argparse.ArgumentParser("Collect CMU MoCap data")
    parser.add_argument('--retarget_path', help='path to the retarget data folder', default='data/cmu_mocap_bvh_retarget')
    return parser.parse_args()

def collect_qpos(file_path):
    anim, _, ftime = BVH.load(file_path)

    xyz = anim.positions[:,0]
    rot = anim.euler_rotations.reshape(len(anim), -1)
    rot = rot[:,[0,1,2,6,7,14,18,19,20,25,28,35,45,46,47,52,55,62,73,74,83,94,95,104]]
    rot = np.deg2rad(rot)
    qpos_values = np.hstack((xyz,rot))

    # Apply gaussian smoothing
    from scipy import signal
    sigma = 10  # try 10 and 15
    w = signal.gaussian(6*sigma+1, sigma)
    w = w / np.sum(w)
    qpos_values_padded = np.pad(qpos_values, ((3*sigma, 3*sigma), (0, 0)), 'edge')
    for j in range(qpos_values.shape[1]):
        qpos_values[:, j] = np.convolve(qpos_values_padded[:,j], w, 'valid')

    # Resample qpos
    time_vals = np.arange(0, len(anim)*ftime - 1e-8, ftime)
    time_vals_new = np.arange(0, len(anim)*ftime, timestep)
    while time_vals_new[-1] > time_vals[-1]:
        time_vals_new = time_vals_new[:-1]

    qpos_values_resampled = []
    for d in range(qpos_values.shape[1]):
        f = interpolate.splrep(time_vals, qpos_values[:, d])
        qpos_values_resampled.append(interpolate.splev(time_vals_new, f))

    qpos_values_resampled = np.stack(qpos_values_resampled, axis=1)  # ntime by nq

    # Check and clip angles
    rot = np.rad2deg(qpos_values_resampled[:,3:])

    _ASSERT_TOL = 0.5
    assert np.all(np.logical_and(rot[:, 3] >  -45-_ASSERT_TOL, rot[:, 3] <  45+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:, 4] >  -75-_ASSERT_TOL, rot[:, 4] <  30+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:, 5] >  -35-_ASSERT_TOL, rot[:, 5] <  35+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:, 6] >  -25-_ASSERT_TOL, rot[:, 6] <   5+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:, 7] >  -60-_ASSERT_TOL, rot[:, 7] <  35+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:, 8] > -110-_ASSERT_TOL, rot[:, 8] <  20+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:, 9] >    2-_ASSERT_TOL, rot[:, 9] < 160+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,10] >  -50-_ASSERT_TOL, rot[:,10] <  50+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,11] >  -50-_ASSERT_TOL, rot[:,11] <  50+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,12] >  -25-_ASSERT_TOL, rot[:,12] <   5+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,13] >  -60-_ASSERT_TOL, rot[:,13] <  35+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,14] > -120-_ASSERT_TOL, rot[:,14] <  20+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,15] >    2-_ASSERT_TOL, rot[:,15] < 160+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,16] >  -50-_ASSERT_TOL, rot[:,16] <  50+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,17] >  -50-_ASSERT_TOL, rot[:,17] <  50+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,18] >  -85-_ASSERT_TOL, rot[:,18] <  60+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,19] >  -85-_ASSERT_TOL, rot[:,19] <  60+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,20] >  -90-_ASSERT_TOL, rot[:,20] <  50+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,21] >  -60-_ASSERT_TOL, rot[:,21] <  85+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,22] >  -60-_ASSERT_TOL, rot[:,22] <  85+_ASSERT_TOL))
    assert np.all(np.logical_and(rot[:,23] >  -90-_ASSERT_TOL, rot[:,23] <  50+_ASSERT_TOL))

    rot[:, 3] = rot[:, 3].clip( -45,  45)  # abdomen_z
    rot[:, 4] = rot[:, 4].clip( -75,  30)  # abdomen_y
    rot[:, 5] = rot[:, 5].clip( -35,  35)  # abdomen_x
    rot[:, 6] = rot[:, 6].clip( -25,   5)  # right_hip1
    rot[:, 7] = rot[:, 7].clip( -60,  35)  # right_hip2
    rot[:, 8] = rot[:, 8].clip(-110,  20)  # right_hip3
    rot[:, 9] = rot[:, 9].clip(   2, 160)  # right_knee (-y axis)
    rot[:,10] = rot[:,10].clip( -50,  50)  # right_ankle_y
    rot[:,11] = rot[:,11].clip( -50,  50)  # right_ankle_x
    rot[:,12] = rot[:,12].clip( -25,   5)  # left_hip1
    rot[:,13] = rot[:,13].clip( -60,  35)  # left_hip2
    rot[:,14] = rot[:,14].clip(-120,  20)  # left_hip3
    rot[:,15] = rot[:,15].clip(   2, 160)  # left_knee (-y axis)
    rot[:,16] = rot[:,16].clip( -50,  50)  # left_ankle_y
    rot[:,17] = rot[:,17].clip( -50,  50)  # left_ankle_x
    rot[:,18] = rot[:,18].clip( -85,  60)  # right_shoulder1
    rot[:,19] = rot[:,19].clip( -85,  60)  # right_shoulder2
    rot[:,20] = rot[:,20].clip( -90,  50)  # right_elbow
    rot[:,21] = rot[:,21].clip( -60,  85)  # left_shoulder1
    rot[:,22] = rot[:,22].clip( -60,  85)  # left_shoulder2
    rot[:,23] = rot[:,23].clip( -90,  50)  # left_elbow

    qpos_values_resampled[:,3:] = np.deg2rad(rot)

    qvel_list = []
    for t in range(qpos_values_resampled.shape[0]-1):
        p_tp1 = qpos_values_resampled[t + 1, :]
        p_t = qpos_values_resampled[t, :]
        qvel = (p_tp1 - p_t) / timestep
        qvel_list.append(qvel)

    qvel_values_resampled = np.vstack(qvel_list)

    qpos_values_resampled = qpos_values_resampled[1:]

    qpos_values_resampled[:,[12,18]] = -qpos_values_resampled[:,[12,18]]
    qvel_values_resampled[:,[12,18]] = -qvel_values_resampled[:,[12,18]]

    jnt_pos = qpos_values_resampled[:,6:]
    jnt_vel = qvel_values_resampled[:,6:]

    jnt = np.empty((jnt_pos.shape[0], jnt_pos.shape[1]*2), dtype=jnt_pos.dtype)
    jnt[:,0::2] = jnt_pos
    jnt[:,1::2] = jnt_vel

    # Note that the yaw values can be out of the range (-np.pi, np.pi] due
    # to the current retargt implementation.
    xyz = qpos_values_resampled[:,:3]
    rpy = qpos_values_resampled[:,3:6][:,::-1]
    vel = qvel_values_resampled[:,:3]

    return np.hstack((jnt, xyz, rpy, vel))

def collect_obs(env, qpos, task='walk'):
    # `feet_contact` is not included as it is unset without running `step`.
    obs = []
    if task == "walk":
        aux = {'rfoot_z': [], 'lfoot_z': []}
    if task in ("turn", "holistic"):
        aux = {}
    if task == "sit":
        aux = {'pelvis_z': [], 'foot_z': [], 'foot_x': []}
    for t in range(len(qpos)):
        for j, joint in enumerate(env.env.ordered_joints):
            joint.reset_current_position(qpos[t, 2*j], qpos[t, 2*j+1])
        cpose = cpp_household.Pose()
        cpose.set_xyz(*qpos[t, -9:-6])
        cpose.set_rpy(*qpos[t, -6:-3])
        env.env.cpp_robot.set_pose_and_speed(cpose, *qpos[t, -3:])
        for r in env.env.mjcf:
            r.query_position()
        state = env.env.calc_state()
        obs.append(state[None])
        if task == "walk":
            aux['rfoot_z'].append(env.env.parts['right_foot'].pose().xyz()[2])
            aux['lfoot_z'].append(env.env.parts['left_foot'].pose().xyz()[2])
        if task == "sit":
            aux['pelvis_z'].append(env.env.parts['pelvis'].pose().xyz()[2])
            aux['foot_z'].append(np.mean([env.env.parts['right_foot'].pose().xyz()[2],
                                          env.env.parts['left_foot'].pose().xyz()[2]]))
            aux['foot_x'].append(np.mean([-1.0 * env.env.parts['right_foot'].pose().xyz()[1],
                                          -1.0 * env.env.parts['left_foot'].pose().xyz()[1]]))
    obs = np.vstack(obs)
    aux = {k: np.array(v, dtype=np.float32) for k, v in aux.items()}
    return obs, aux

def find_zero_crossings(x):
    assert np.all(np.sign(x) != 0)
    ind = np.where(np.diff(np.sign(x)))[0]
    out = []
    for i in ind:
        if abs(x[i]) <= abs(x[i+1]):
            out.append(i)
        else:
            out.append(i+1)
    return np.asarray(out)

def segment_walk(rfoot_z, lfoot_z, ind, task):
    rfoot_g1 = np.gradient(rfoot_z)
    lfoot_g1 = np.gradient(lfoot_z)

    zr = find_zero_crossings(rfoot_g1)
    zl = find_zero_crossings(lfoot_g1)

    rfoot_g2 = np.gradient(rfoot_g1)
    lfoot_g2 = np.gradient(lfoot_g1)

    # Thresholds are set by heuristics (inspecting the curves).
    start_r = zl[np.logical_and(lfoot_g2[zl] > 0, rfoot_g2[zl] < -1e-4)]
    start_l = zr[np.logical_and(rfoot_g2[zr] > 0, lfoot_g2[zr] < -1e-4)]

    if ind in _ADD_LIST[task]:
        if 'r' in _ADD_LIST[task][ind]:
            start_r = np.sort(np.concatenate((start_r, _ADD_LIST[task][ind]['r'])))
        if 'l' in _ADD_LIST[task][ind]:
            start_l = np.sort(np.concatenate((start_l, _ADD_LIST[task][ind]['l'])))
    if ind in _DEL_LIST:
        if 'r' in _DEL_LIST[task][ind]:
            start_r = np.delete(start_r, [np.where(start_r == x) for x in _DEL_LIST[task][ind]['r']])
        if 'l' in _DEL_LIST[task][ind]:
            start_l = np.delete(start_l, [np.where(start_l == x) for x in _DEL_LIST[task][ind]['l']])

    assert abs(len(start_r) - len(start_l)) <= 1
    assert np.intersect1d(start_r, start_l).size == 0
    if start_r[0] < start_l[0]:
        assert len(start_r) >= len(start_l)
        rs = start_r[:len(start_l)]
        re = start_l
        ls = start_l[:len(start_r)-1]
        le = start_r[1:]
    else:
        assert len(start_l) >= len(start_r)
        rs = start_r[:len(start_l)-1]
        re = start_l[1:]
        ls = start_l[:len(start_r)]
        le = start_r
    assert np.all(np.less(rs, re))
    assert np.all(np.less(ls, le))

    rstep = np.stack((np.asarray([ind] * len(rs)), rs, re - rs), axis=-1)
    lstep = np.stack((np.asarray([ind] * len(ls)), ls, le - ls), axis=-1)
    return rstep, lstep

def segment_sit(pelvis_z, ind, task):
    pelvis_g1 = np.gradient(pelvis_z)
    pelvis_g2 = np.gradient(pelvis_g1)

    z = find_zero_crossings(pelvis_g1)

    # Thresholds are set by heuristics (inspecting the curves).
    start_d = z[np.logical_and(np.abs(pelvis_g2[z]) < 1e-4, pelvis_z[z] > 0.8)]
    start_u = z[np.logical_and(np.abs(pelvis_g2[z]) < 1e-4, pelvis_z[z] < 0.8)]

    if ind in _ADD_LIST:
        if 'd' in _ADD_LIST[ind]:
            start_d = np.sort(np.concatenate((start_d, _ADD_LIST[task][ind]['d'])))
        if 'u' in _ADD_LIST[ind]:
            start_u = np.sort(np.concatenate((start_u, _ADD_LIST[task][ind]['u'])))
    if ind in _DEL_LIST:
        if 'd' in _DEL_LIST[ind]:
            start_d = np.delete(start_d, [np.where(start_d == x) for x in _DEL_LIST[task][ind]['d']])
        if 'u' in _DEL_LIST[ind]:
            start_u = np.delete(start_u, [np.where(start_u == x) for x in _DEL_LIST[task][ind]['u']])

    assert abs(len(start_d) - len(start_u)) <= 1
    assert np.intersect1d(start_d, start_u).size == 0
    if start_d[0] < start_u[0]:
        assert len(start_d) >= len(start_u)
        ds = start_d[:len(start_u)]
        de = start_u
        us = start_u[:len(start_d)-1]
        ue = start_d[1:]
    else:
        assert len(start_u) >= len(start_d)
        ds = start_d[:len(start_u)-1]
        de = start_u[1:]
        us = start_u[:len(start_d)]
        ue = start_d
    assert np.all(np.less(ds, de))
    assert np.all(np.less(us, ue))

    sitd = np.stack((np.asarray([ind] * len(ds)), ds, de - ds), axis=-1)
    situ = np.stack((np.asarray([ind] * len(us)), us, ue - us), axis=-1)

    return sitd, situ

def align_foot_sit(segments, foot_z, foot_x):
    offset_z, offset_x = [], []
    for s in segments:
        offset_z.append(foot_z[s[1]:s[1] + s[2] + 1] - 0.07)  # approximating touching floor
        offset_x.append(foot_x[s[1]:s[1] + s[2] + 1] - foot_x[s[1]])
    return offset_z, offset_x

def main(args):
    env = gym.make('RoboschoolHumanoidBullet3-v1')
    env.reset()

    print('collecting cmu mocap data ... ')

    assert args.retarget_path is not None

    for task in ("walk", "turn", "sit", "holistic"):
        data = {'qpos': [], 'obs': []}
        if task == "walk":
            data['rstep'] = np.empty([0, 3], dtype=np.int64)
            data['lstep'] = np.empty([0, 3], dtype=np.int64)
        if task == "turn":
            data['rturn'] = _R_TURN
            data['lturn'] = _L_TURN
        if task == "sit":
            data['sitd'] = np.empty([0, 3], dtype=np.int64)
            data['offsetz'] = []
            data['offsetx'] = []
        if task == "holistic":
            data['holist'] = _HOLIST

        for ind, i in enumerate(_SUBJECT_AMC_ID[task]):
            file_path = os.path.join(
                args.retarget_path,'{:02d}'.format(_SUBJECT_ID[task]),'{:02d}_{:02d}.bvh'.format(_SUBJECT_ID[task],i))
            qpos = collect_qpos(file_path)
            obs, aux = collect_obs(env, qpos, task=task)
            data['qpos'].append(qpos)
            data['obs'].append(obs)
            if task == "walk":
                rstep, lstep = segment_walk(aux['rfoot_z'], aux['lfoot_z'], ind, task)
                data['rstep'] = np.vstack((data['rstep'], rstep))
                data['lstep'] = np.vstack((data['lstep'], lstep))
            if task == "sit":
                sitd, _ = segment_sit(aux['pelvis_z'], ind, task)
                data['sitd'] = np.vstack((data['sitd'], sitd))
                offset_z, offset_x = align_foot_sit(sitd, aux['foot_z'], aux['foot_x'])
                data['offsetz'].append(offset_z)
                data['offsetx'].append(offset_x)

        save_path = 'data/cmu_mocap_{}.npz'.format(task)
        if not os.path.isfile(save_path):
            np.savez(save_path, **data)


if __name__ == '__main__':
    args = argsparser()
    main(args)

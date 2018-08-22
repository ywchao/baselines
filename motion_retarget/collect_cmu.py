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

_SUBJECT_ID = 8
_SUBJECT_AMC_ID = [1,2,3,4,5,6,7,8,9,10,11]

# Since this is a heuristic algorithm, we need to manually clean up the output
# by adding and removing indices. The keys here are indices, not amc ids.
# TODO: add _DEL_LIST.
_ADD_LIST = {}


def argsparser():
    parser = argparse.ArgumentParser("Collect CMU MoCap data")
    parser.add_argument('--retarget_path', help='path to the retarget data folder', default='data/cmu_mocap_bvh_retarget')
    return parser.parse_args()

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

def segment_steps(rfoot_z, lfoot_z, ind):
    rfoot_g1 = np.gradient(np.array(rfoot_z))
    lfoot_g1 = np.gradient(np.array(lfoot_z))

    zr = find_zero_crossings(rfoot_g1)
    zl = find_zero_crossings(lfoot_g1)

    rfoot_g2 = np.gradient(rfoot_g1)
    lfoot_g2 = np.gradient(lfoot_g1)

    # Thresholds are set by heuristics (inspecting the curves).
    start_r = zl[np.where(np.logical_and(lfoot_g2[zl] > 0, rfoot_g2[zl] < -1e-4))[0]]
    start_l = zr[np.where(np.logical_and(rfoot_g2[zr] > 0, lfoot_g2[zr] < -1e-4))[0]]

    # Add indices from _ADD_LIST.
    if ind in _ADD_LIST:
        if 'r' in _ADD_LIST[ind]:
            start_r = np.sort(np.concatenate((start_r, _ADD_LIST[ind]['r'])))
        if 'l' in _ADD_LIST[ind]:
            start_l = np.sort(np.concatenate((start_l, _ADD_LIST[ind]['l'])))
    # TODO: add _DEL_LIST.

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

def main(args):
    env = gym.make('RoboschoolHumanoidBullet3-v1')
    env.reset()

    print('collecting cmu mocap data ... ')

    assert args.retarget_path is not None
    all_qpos = []
    all_obs = []
    all_rstep = np.empty([0, 3], dtype=np.int64)
    all_lstep = np.empty([0, 3], dtype=np.int64)
    for ind, i in enumerate(_SUBJECT_AMC_ID):
        file_path = os.path.join(args.retarget_path,'{:02d}'.format(_SUBJECT_ID),'{:02d}_{:02d}.bvh'.format(_SUBJECT_ID,i))
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

        # Construct qpos and obs
        qpos_values_resampled = qpos_values_resampled[1:]

        qpos_values_resampled[:,[12,18]] = -qpos_values_resampled[:,[12,18]]
        qvel_values_resampled[:,[12,18]] = -qvel_values_resampled[:,[12,18]]

        jnt_pos = qpos_values_resampled[:,6:]
        jnt_vel = qvel_values_resampled[:,6:]

        jnt = np.empty((jnt_pos.shape[0], jnt_pos.shape[1]*2), dtype=jnt_pos.dtype)
        jnt[:,0::2] = jnt_pos
        jnt[:,1::2] = jnt_vel

        xyz = qpos_values_resampled[:,:3]
        rpy = qpos_values_resampled[:,3:6][:,::-1]
        vel = qvel_values_resampled[:,:3]

        obs = []
        rfoot_z = []
        lfoot_z = []
        for t in range(len(jnt)):
            for j, joint in enumerate(env.env.ordered_joints):
                joint.reset_current_position(jnt[t, 2*j], jnt[t, 2*j+1])
            cpose = cpp_household.Pose()
            cpose.set_xyz(*xyz[t])
            cpose.set_rpy(*rpy[t])
            env.env.cpp_robot.set_pose_and_speed(cpose, *vel[t])
            for r in env.env.mjcf:
                r.query_position()
            state = env.env.calc_state()
            obs.append(state[None])
            rfoot_z.append(env.env.parts['right_foot'].pose().xyz()[2])
            lfoot_z.append(env.env.parts['left_foot'].pose().xyz()[2])

        # `feet_contact` is not included as it is unset without running `step`.
        # This does not matter if using new state vector.
        qpos = np.hstack((jnt, xyz, rpy, vel))
        obs = np.vstack(obs)

        all_qpos.append(qpos)
        all_obs.append(obs)

        rstep, lstep = segment_steps(rfoot_z, lfoot_z, ind)

        all_rstep = np.vstack((all_rstep, rstep))
        all_lstep = np.vstack((all_lstep, lstep))

    save_path = 'data/cmu_mocap.npz'
    if not os.path.isfile(save_path):
        np.savez(save_path, obs=all_obs, qpos=all_qpos, rstep=all_rstep, lstep=all_lstep)

    print('done.')


if __name__ == '__main__':
    args = argsparser()
    main(args)

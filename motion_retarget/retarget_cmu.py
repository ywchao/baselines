import argparse
import numpy as np
import scipy
import os
import sys

sys.path.append('motion_retarget/motionsynth')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from InverseKinematics import BasicInverseKinematics


def argsparser():
    parser = argparse.ArgumentParser("Motion retargeting for CMU MoCap")
    parser.add_argument('--src_bvh', help='source BVH file', default='data/cmu_mocap_bvh/08/08_01.bvh')
    parser.add_argument('--skel_bvh', help='target skeleton BVH file', default='motion_retarget/humanoid.bvh')
    parser.add_argument('--out_bvh',  help='output BVH file', default='data/cmu_mocap_bvh_retarget/08/08_01.bvh')
    return parser.parse_args()


def main(args):
    if os.path.exists(args.out_bvh) and os.path.isfile(args.out_bvh):
        print('output file already exists.')
        sys.exit(0)

    anim, names, ftime, targetmap = basicIK(args.skel_bvh, args.src_bvh)
    anim = simplifiedIK(anim, targetmap)

    path = os.path.dirname(args.out_bvh)
    if path is not '' and not os.path.exists(path):
        os.makedirs(path)
    BVH.save(args.out_bvh, anim, names, ftime)
    print('done.')


def basicIK(skel_bvh, src_bvh):
    rest, names, _ = BVH.load(skel_bvh)
    rest_targets = Animation.positions_global(rest)
    rest_height = 1.063  # abdomen_zy to right/left foot
    rest.positions = rest.offsets[np.newaxis]
    rest.rotations.qs = rest.orients.qs[np.newaxis]

    anim, _, ftime = BVH.load(src_bvh)
    anim_targets = Animation.positions_global(anim)
    # The first frame is in the upright pose already.
    anim_height = anim_targets[0,12,1] - anim_targets[0,9,1]  # 12: Spline, 9: RightFoot

    targets = (rest_height / anim_height) * anim_targets

    # Remove the first frame and convert to roboschool coordinates.
    targets = targets[1:]
    targets = targets[...,[2,0,1]]

    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)
    # Initialize translations by displacement of abdomen_zy (Spline) from torso
    anim.positions[:,0] = targets[:,12] - (rest_targets[0,2] - rest_targets[0,0])

    # TODO: try different mapping for torso, lwaist, abdomen_zy, pelvis, and
    # abdomen_x. Note that only angle matters here.
    mapping = {
         0: 13,  # torso -> Spine1
         1: 12,  # lwaist -> Spine
         2: 12,  # abdomen_zy -> Spine
         3:  0,  # pelvis -> Hips
         4:  0,  # adbomen_x -> Hips
         5:  7,  # right_hip_forward -> RighUpLeg
         6:  7,  # right_hip -> RighUpLeg
         7:  7,  # right_hip_backward -> RighUpLeg
         8:  8,  # right_knee -> RightLeg
         9:  9,  # right_ankle_y -> RightFoot
        10:  9,  # right_ankle_x_forward -> RightFoot
        11:  9,  # right_ankle_x -> RightFoot
        12:  9,  # right_ankle_x_backward -> RightFoot
        13: 10,  # right_toe_base -> RightToeBase
        14:  2,  # left_hip_forward -> LeftUpLeg
        15:  2,  # left_hip -> LeftUpLeg
        16:  2,  # left_hip_backward -> LeftUpLeg
        17:  3,  # left_knee -> LeftLeg
        18:  4,  # left_ankle_y -> LeftFoot
        19:  4,  # left_ankle_x_forward -> LeftFoot
        20:  4,  # left_ankle_x -> LeftFoot
        21:  4,  # left_ankle_x_backward -> LeftFoot
        22:  5,  # left_toe_base -> LeftToeBase
        23: 25,  # right_shoulder_forward -> RightArm
        24: 25,  # right_shoulder -> RightArm
        25: 25,  # right_shoulder_backward -> RightArm
        26: 26,  # right_elbow_forward -> RightForeArm
        27: 26,  # right_elbow -> RightForeArm
        28: 26,  # right_elbow_backward -> RightForeArm
        29: 27,  # right_hand -> RightHand
        30: 18,  # left_shoulder_forward -> LeftArm
        31: 18,  # left_shoulder -> LeftArm
        32: 18,  # left_shoulder_backward -> LeftArm
        33: 19,  # left_elbow_forward -> LeftForeArm
        34: 19,  # left_elbow -> LeftForeArm
        35: 19,  # left_elbow_backward -> LeftForeArm
        36: 20,  # left_hand -> LeftHand
    }
    mapping = [mapping[i] for i in range(rest_targets.shape[1])]

    targetmap = targets[:,mapping]

    ik = BasicInverseKinematics(anim, targetmap, iterations=10, silent=True)
    ik()

    anim.euler_rotations = np.degrees(anim.rotations.euler(order='zyx'[::-1])[...,::-1])

    return anim, names, ftime, targetmap


def simplifiedIK(anim, targetmap):
    modify_list = [  0,   1,   2,  # torso
                     6,   7,       # adbomen_zy
                              14,  # abdomen_x
                    18,  19,  20,  # right_hip
                         25,       # right_knee
                         28,       # right_ankle_y
                              35,  # right_ankle_x
                    45,  46,  47,  # left_hip
                         52,       # left_knee
                         55,       # left_ankle_y
                              62,  # left_ankle_x
                         73,  74,  # right_shoulder
                              83,  # right_elbow
                         94,  95,  # left shoulder
                             104]  # left elbow

    end_effector_list = [ 9,  # right_ankle_y
                         11,  # right_ankle_x
                         13,  # right_toe_base
                         18,  # left_ankle_y
                         20,  # left_ankle_x
                         22]  # left_toe_base

    accum_err = [0.0, 0.0]

    for frame_idx in range(anim.rotations.shape[0]):
        # print(frame_idx, end=', ')

        anim_i = anim.copy()
        anim_i.positions = anim.positions[frame_idx][None]
        anim_i.rotations = anim.rotations[frame_idx][None]
        anim_i.euler_rotations = anim.euler_rotations[frame_idx][None]

        if frame_idx == 0:
            euler_rotations_1 = anim.euler_rotations[frame_idx].copy()
            translations_1 = anim.positions[frame_idx,0].copy()

        target_pos = targetmap[frame_idx]

        # Phase 1: all joints
        euler_rotations_1, translations_1, it_1, error_1 = jacobianIK(
            anim_i, euler_rotations_1, translations_1, modify_list, target_pos, opt_thres=1e-5)
        accum_err[0] += error_1

        # Phase 2: ankles only, with criterion
        euler_rotations_2, translations_2, it_2, error_2 = jacobianIK(
            anim_i, euler_rotations_1, translations_1, modify_list, target_pos, opt_thres=1e-5,
            end_effector_list=end_effector_list, criterion=True)
        accum_err[1] += error_2

        print('frame: {:04d} | iter_1: {:03d}, error_1: {:.6f} | iter_2: {:03d}, error_2: {:.6f}'.format(
            frame_idx, it_1, error_1, it_2, error_2))

        anim.euler_rotations[frame_idx] = euler_rotations_2.copy()
        anim.rotations[frame_idx] = Quaternions.from_euler(np.radians(euler_rotations_2), order=anim.order, world=False)
        anim.positions[frame_idx,0] = translations_2[None]

    print('avg error_1: {:.6f}, avg error_2: {:.6f}'.format(
        accum_err[0] / anim.rotations.shape[0], accum_err[1] / anim.rotations.shape[0]))
    return anim


def jacobianIK(anim, euler_rotations, translations, modify_list, target_pos,
        opt_thres=1e-5, end_effector_list=None, criterion=False):
    """
    Args:
        euler_rotations: A [num_joints, 3] numpy array
        translations: A [3] numpy array
        target_pos: A [num_joints, 3] numpy array
    """
    def prefix_angles(euler_rotations):
        # Blender has issues visualizaing the output bvh. To fix this just
        # change all the `90.` to `89.99` for entry 5, 7, 10, and 12.
        euler_rotations[ 1] = np.array([    0.    ,   -0.2290,    0.    ])
        euler_rotations[ 3] = np.array([    0.    ,   -0.2290,    0.    ])
        euler_rotations[ 5] = np.array([   90.    ,    0.    ,   90.    ])
        euler_rotations[ 7] = np.array([  -90.    ,  -90.    ,    0.    ])
        euler_rotations[10] = np.array([    0.    ,   63.4349,    0.    ])
        euler_rotations[12] = np.array([    0.    ,  -63.4349,    0.    ])
        euler_rotations[14] = np.array([   90.    ,    0.    ,  -90.    ])
        euler_rotations[16] = np.array([  -90.    ,   90.    ,    0.    ])
        euler_rotations[19] = np.array([    0.    ,   63.4349,    0.    ])
        euler_rotations[21] = np.array([    0.    ,  -63.4349,    0.    ])
        euler_rotations[23] = np.array([  -90.    ,  -45.    ,   35.2644])
        euler_rotations[25] = np.array([   90.    ,   35.2644,   45.    ])
        euler_rotations[26] = np.array([  -90.    ,  -45.    ,   35.2644])
        euler_rotations[28] = np.array([   90.    ,   35.2644,   45.    ])
        euler_rotations[30] = np.array([   90.    ,  -45.    ,  144.7356])
        euler_rotations[32] = np.array([   90.    ,  -35.2644,  135.    ])
        euler_rotations[33] = np.array([  -90.    ,   45.    ,   35.2644])
        euler_rotations[35] = np.array([   90.    ,   35.2644,  -45.    ])
        return euler_rotations

    def clip_angles(euler_rotations):
        euler_rotations[ 2,0] = euler_rotations[ 2,0].clip( -45,  45)  # abdomen_z
        euler_rotations[ 2,1] = euler_rotations[ 2,1].clip( -75,  30)  # abdomen_y
        euler_rotations[ 4,2] = euler_rotations[ 4,2].clip( -35,  35)  # abdomen_x
        euler_rotations[ 6,0] = euler_rotations[ 6,0].clip( -25,   5)  # right_hip1
        euler_rotations[ 6,1] = euler_rotations[ 6,1].clip( -60,  35)  # right_hip2
        euler_rotations[ 6,2] = euler_rotations[ 6,2].clip(-110,  20)  # right_hip3
        euler_rotations[ 8,1] = euler_rotations[ 8,1].clip(   2, 160)  # right_knee (-y axis)
        euler_rotations[ 9,1] = euler_rotations[ 9,1].clip( -50,  50)  # right_ankle_y
        euler_rotations[11,2] = euler_rotations[11,2].clip( -50,  50)  # right_ankle_x
        euler_rotations[15,0] = euler_rotations[15,0].clip( -25,   5)  # left_hip1
        euler_rotations[15,1] = euler_rotations[15,1].clip( -60,  35)  # left_hip2
        euler_rotations[15,2] = euler_rotations[15,2].clip(-120,  20)  # left_hip3
        euler_rotations[17,1] = euler_rotations[17,1].clip(   2, 160)  # left_knee (-y axis)
        euler_rotations[18,1] = euler_rotations[18,1].clip( -50,  50)  # left_ankle_y
        euler_rotations[20,2] = euler_rotations[20,2].clip( -50,  50)  # left_ankle_x
        euler_rotations[24,1] = euler_rotations[24,1].clip( -85,  60)  # right_shoulder1
        euler_rotations[24,2] = euler_rotations[24,2].clip( -85,  60)  # right_shoulder2
        euler_rotations[27,2] = euler_rotations[27,2].clip( -90,  50)  # right_elbow
        euler_rotations[31,1] = euler_rotations[31,1].clip( -60,  85)  # left_shoulder1
        euler_rotations[31,2] = euler_rotations[31,2].clip( -60,  85)  # left_shoulder2
        euler_rotations[34,2] = euler_rotations[34,2].clip( -90,  50)  # left_elbow
        return euler_rotations

    # Initialize Jacobian
    if end_effector_list is None:
        J = np.zeros((target_pos.shape[0] * target_pos.shape[1], len(modify_list) + 3))
    else:
        J = np.zeros((len(end_effector_list) * target_pos.shape[1], len(modify_list) + 3))

    euler_rotations = euler_rotations.copy()
    translations = translations.copy()

    fixed_list = list(set(range(anim.euler_rotations.shape[1]*3)) - set(modify_list))
    for idx in fixed_list:
        euler_rotations[idx // 3, idx % 3] = 0.0

    # Prefix angles
    euler_rotations = prefix_angles(euler_rotations)

    anim.update_rotations(euler_rotations[None])
    anim.positions[:,0] = translations[None]

    if criterion:
        theta_0 = np.hstack((euler_rotations.flatten()[modify_list], translations))

    eps = 1e-9
    prev_error = 1e10
    grad_scaling_factor = 100

    it = 0
    while True:
        init_coord = Animation.positions_global(anim)[0]
        dists = (target_pos - init_coord) * grad_scaling_factor
        if end_effector_list is not None:
            dists = dists[end_effector_list]
        dists = dists.flatten()

        # Compute Jacobian
        for col_idx, ang_idx in enumerate(modify_list):
            rot = euler_rotations.copy()

            rot[ang_idx // 3, ang_idx % 3] += eps
            anim.update_rotations(rot[None])
            coord_plus_eps = Animation.positions_global(anim)[0]
            if end_effector_list is not None:
                coord_plus_eps = coord_plus_eps[end_effector_list]

            rot[ang_idx // 3, ang_idx % 3] -= 2.0 * eps
            anim.update_rotations(rot[None])
            coord_minus_eps = Animation.positions_global(anim)[0]
            if end_effector_list is not None:
                coord_minus_eps = coord_minus_eps[end_effector_list]

            grad = (coord_plus_eps - coord_minus_eps) / (2.0 * eps)
            J[:, col_idx] = grad.flatten()

        anim.update_rotations(euler_rotations[None])

        for idx in range(3):
            trans = translations.copy()

            trans[idx] += eps
            anim.positions[:,0] = trans[None]
            coord_plus_eps = Animation.positions_global(anim)[0]
            if end_effector_list is not None:
                coord_plus_eps = coord_plus_eps[end_effector_list]

            trans[idx] -= 2.0 * eps
            anim.positions[:,0] = trans[None]
            coord_minus_eps = Animation.positions_global(anim)[0]
            if end_effector_list is not None:
                coord_minus_eps = coord_minus_eps[end_effector_list]

            grad = (coord_plus_eps - coord_minus_eps) / (2.0 * eps)
            J[:, len(modify_list)+idx] = grad.flatten()

        anim.positions[:,0] = translations[None]

        J *= grad_scaling_factor

        damping = 2.0
        l = damping * (1.0 / (1 + 0.001))
        if not criterion:
            # Assumes J.shape[0] >= J.shape[1]
            d = (l*l) * np.eye(J.shape[1])
            dr = scipy.linalg.lu_solve(scipy.linalg.lu_factor(J.T.dot(J) + d), J.T.dot(dists))
        else:
            # Assumes J.shape[0] < J.shape[1]
            d = (l*l) * np.eye(J.shape[0])
            grad_crit = theta_0 - np.hstack((euler_rotations.flatten()[modify_list], translations))
            dr = scipy.linalg.lu_solve(scipy.linalg.lu_factor(J.dot(J.T) + d), dists - J.dot(grad_crit))
            dr = J.T.dot(dr) + grad_crit

        for col_idx, ang_idx in enumerate(modify_list):
            euler_rotations[ang_idx // 3, ang_idx % 3] += (dr[col_idx])

        for idx in range(3):
            translations[idx] += dr[len(modify_list)+idx]

        # Clip angles to range
        euler_rotations = clip_angles(euler_rotations)

        anim.update_rotations(euler_rotations[None])
        anim.positions[:,0] = translations[None]

        cur_coord = Animation.positions_global(anim)[0]
        error = np.sum(np.square(cur_coord - target_pos))

        it += 1
        # print(it, error)
        if it > 1000 or prev_error - error < opt_thres:
            break

        prev_error = error

    return euler_rotations, translations, it, error


if __name__ == '__main__':
    args = argsparser()
    main(args)


# roboschool humanoid.bvh             | cmu-mocap bvh
#  0: torso                           |  0: Hips
#  1: lwaist                          |  1: LHipJoint
#  2: abdomen_zy                      |  2: LeftUpLeg
#  3: pelvis                          |  3: LeftLeg
#  4: adbomen_x                       |  4: LeftFoot
#  5: right_hip_forward               |  5: LeftToeBase
#  6: right_hip                       |  6: RHipJoint
#  7: right_hip_backward              |  7: RightUpLeg
#  8: right_knee                      |  8: RightLeg
#  9: right_ankle_y                   |  9: RightFoot
# 10: right_ankle_x_forward           | 10: RightToeBase
# 11: right_ankle_x                   | 11: LowerBack
# 12: right_ankle_x_backward          | 12: Spine
# 13: right_toe_base                  | 13: Spint1
# 14: left_hip_forward                | 14: Neck
# 15: left_hip                        | 15: Neck1
# 16: left_hip_backward               | 16: Head
# 17: left_knee                       | 17: LeftShoulder
# 18: left_ankle_y                    | 18: LeftArm
# 19: left_ankle_x_forward            | 19: LeftForeArm
# 20: left_ankle_x                    | 20: LeftHand
# 21: left_ankle_x_backward           | 21: LeftFingerBase
# 22: left_toe_base                   | 22: LeftHandIndex1
# 23: right_shoulder_forward          | 23: LThumb
# 24: right_shoulder                  | 24: RightShoulder
# 25: right_shoulder_backward         | 25: RightArm
# 26: right_elbow_forward             | 26: RightForeArm
# 27: right_elbow                     | 27: RightHand
# 28: right_elbow_backward            | 28: RightFingerBase
# 29: right_hand                      | 29: RightHandIndex1
# 30: left_shoulder_forward           | 30: RThumb
# 31: left_shoulder
# 32: left_shoulder_backward
# 33: left_elbow_forward
# 34: left_elbow
# 35: left_elbow_backward
# 36: left_hand

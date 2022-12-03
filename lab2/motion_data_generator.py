from answer_task1 import *

def get_motion_frame(motion, frame_index):
    trans_y = motion.joint_position[frame_index][0][1]
    trans_xz = np.array([motion.joint_position[frame_index][0][0], motion.joint_position[frame_index][0][2]])
    ori_y, ori_xz = motion.decompose_rotation_with_yaxis(motion.joint_rotation[frame_index][0])
    rots = motion.joint_rotation[frame_index][1:-1]

    return trans_y, trans_xz, ori_y, ori_xz, rots


def load_motion_data(files):
    features = np.zeros((0, 103))
    labels = np.zeros((0, 97))
    for file in files:
        motion = BVHMotion(file)

        motion_input = np.zeros((motion.motion_length - 1, 103))
        motion_output = np.zeros((motion.motion_length - 1, 97))

        l_trans_y, l_trans_xz, l_ori_y, l_ori_xz, l_rots = get_motion_frame(motion, 0)
        n_trans_y, n_trans_xz, n_ori_y, n_ori_xz, n_rots = l_trans_y, l_trans_xz, l_ori_y, l_ori_xz, l_rots
        for f in range(1, motion.motion_length):
            n_trans_y, n_trans_xz, n_ori_y, n_ori_xz, n_rots = get_motion_frame(motion, f)

            delta_trans_y = n_trans_y - l_trans_y
            delta_trans_xz = n_trans_xz - l_trans_xz

            delta_ori_y = (R.from_quat(n_ori_y) * R.inv(R.from_quat(l_ori_y))).as_quat()
            delta_ori_xz = (R.from_quat(n_ori_xz) * R.inv(R.from_quat(l_ori_xz))).as_quat()

            delta_rots = np.copy(l_rots)
            for i in range(23):
                delta_rots[i] = (R.from_quat(n_rots[i]) * R.inv(R.from_quat(l_rots[i]))).as_quat()

            motion_input[f - 1][0] = l_trans_y
            motion_input[f - 1][1:5] = l_ori_xz
            motion_input[f - 1][5:97] = np.resize(l_rots, (1, 92))
            motion_input[f - 1][97:99] = delta_trans_xz
            motion_input[f - 1][99:103] = delta_ori_y

            motion_output[f - 1][0] = delta_trans_y
            motion_output[f - 1][1:5] = delta_ori_xz
            motion_output[f - 1][5:97] = np.resize(delta_rots, (1, 92))

            l_trans_y, l_trans_xz, l_ori_y, l_ori_xz, l_rots = n_trans_y, n_trans_xz, n_ori_y, n_ori_xz, n_rots

        features = np.concatenate((features, motion_input), axis=0)
        labels = np.concatenate((labels, motion_output), axis=0)

    return features, labels



motion_files = [
    'motion_material/idle.bvh',
    'motion_material/run_forward.bvh',
    'motion_material/walk_and_ture_right.bvh',
    'motion_material/walk_and_turn_left.bvh',
    'motion_material/walk_forward.bvh',
    'motion_material/walkF.bvh',

    # 'motion_material/physics_motion/long_run.bvh',
    # 'motion_material/physics_motion/long_run_mirror.bvh',
    # 'motion_material/physics_motion/long_walk.bvh',
    # 'motion_material/physics_motion/long_walk_mirror.bvh',

    # 'motion_material/kinematic_motion/long_run.bvh',
    # 'motion_material/kinematic_motion/long_run_mirror.bvh',
    # 'motion_material/kinematic_motion/long_walk.bvh',
    # 'motion_material/kinematic_motion/long_walk_mirror.bvh',
]

features, labels = load_motion_data(motion_files)

np.savetxt('motion_material/_features.csv', features, fmt='%f', delimiter=',')
np.savetxt('motion_material/_labels.csv', labels, fmt='%f', delimiter=',')
print('features.shape:', features.shape, 'labels.shape:', labels.shape)
# n_features = np.loadtxt('motion_material/_features.csv', dtype=np.float, delimiter=',', unpack=False)
# n_labels = np.loadtxt('motion_material/_labels.csv', dtype=np.float, delimiter=',', unpack=False)
# print('n_features.shape:', n_features.shape, 'n_labels.shape:', n_labels.shape)

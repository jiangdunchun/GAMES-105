from answer_task1 import *

def get_local_pos_in_coord(trans_base, orien_base, trans):
    return R.inv(R.from_quat(orien_base)).as_matrix() @ (trans - trans_base)

def get_local_rot_in_coord(trans_base, orien_base, orien):
    return (R.inv(R.from_quat(orien_base)) * R.from_quat(orien)).as_quat()



def construct_input_data(l_root_trans_ic, l_root_orien_ic, c_key_joints_pos_list, l_key_joints_trans_icj_list, desire_root_trans_ic_list, desire_root_orien_ic_list):
    input = np.array([])

    input = np.append(input, l_root_trans_ic)
    input = np.append(input, l_root_orien_ic)

    for c_key_joints_pos in c_key_joints_pos_list:
        input = np.append(input, c_key_joints_pos)

    for l_key_joints_trans_icj in l_key_joints_trans_icj_list:
        input = np.append(input, l_key_joints_trans_icj)

    for desire_root_trans_ic in desire_root_trans_ic_list:
        input = np.append(input, desire_root_trans_ic)

    for desire_root_orien_ic in desire_root_orien_ic_list:
        input = np.append(input, desire_root_orien_ic) 

    input = input.reshape(1, -1)
    return input  

def deconstruct_input_data(input, key_joints, desire_frames):
    index = 0

    l_root_trans_ic = input[index:index + 3]
    index = index + 3
    l_root_orien_ic = input[index:index + 4]
    index = index + 4

    c_key_joints_pos_list = []
    for _ in key_joints:
        c_key_joints_pos_list.append(input[index:index + 3])
        index = index + 3

    l_key_joints_trans_icj_list = []
    for _ in key_joints:
        l_key_joints_trans_icj_list.append(input[index:index + 3])
        index = index + 3

    desire_root_trans_ic_list = []
    for _ in desire_frames:
        desire_root_trans_ic_list.append(input[index:index + 3])
        index = index + 3
    
    desire_root_orien_ic_list = []
    for _ in desire_frames:
        desire_root_orien_ic_list.append(input[index:index + 4])
        index = index + 4
        
    return l_root_trans_ic, l_root_orien_ic, c_key_joints_pos_list, l_key_joints_trans_icj_list, desire_root_trans_ic_list, desire_root_orien_ic_list      

def construct_output_data(n_root_trans_ic, n_root_orien_ic, n_joints_no_root_rot_list):
    output = np.array([])

    output = np.append(output, n_root_trans_ic)
    output = np.append(output, n_root_orien_ic)

    for n_joints_rot in n_joints_no_root_rot_list:
        output = np.append(output, n_joints_rot)

    output = output.reshape(1, -1)
    return output  

def deconstruct_output_data(output, joints_size):
    index = 0

    n_root_trans_ic = output[index:index + 3]
    index = index + 3
    n_root_orien_ic = output[index:index + 4]
    index = index + 4

    n_joints_no_root_rot_list = []
    for _ in range(1, joints_size):
        n_joints_no_root_rot_list.append(output[index:index + 4])
        index = index + 4
        
    return n_root_trans_ic, n_root_orien_ic, n_joints_no_root_rot_list 

def load_matching_data(files, key_joints, desire_frames_delta, joints_size):
    features = np.zeros((0, 7 + 6 * len(key_joints) + 7 * len(desire_frames_delta)))
    labels = np.zeros((0, 7 + 4 * (joints_size - 1)))

    for file in files:
        motion = BVHMotion(file)

        joint_translation, joint_orientation = motion.batch_forward_kinematics()
        for c_frame in range(1, motion.motion_length):
            l_frame = c_frame - 1

            c_root_trans = motion.joint_position[c_frame][0]
            c_root_orien = motion.joint_rotation[c_frame][0]

            l_root_trans = motion.joint_position[l_frame][0]
            l_root_orien = motion.joint_rotation[l_frame][0]

            l_root_trans_ic = get_local_pos_in_coord(c_root_trans, c_root_orien, l_root_trans)
            l_root_orien_ic = get_local_rot_in_coord(c_root_trans, c_root_orien, l_root_orien)

            c_key_joints_pos_list = []
            l_key_joints_trans_icj_list = []
            for joint in key_joints:
                c_joint_trans = joint_translation[c_frame][joint]
                c_joint_pos = get_local_pos_in_coord(c_root_trans, c_root_orien, c_joint_trans)

                l_joint_trans = joint_translation[l_frame][joint]
                l_key_joints_trans_icj = l_joint_trans - c_joint_trans

                c_key_joints_pos_list.append(c_joint_pos)
                l_key_joints_trans_icj_list.append(l_key_joints_trans_icj)

            desire_frames = []
            for frame_delta in desire_frames_delta:
                desire_frames.append(c_frame + frame_delta)
            if desire_frames[-1] >= motion.motion_length:
                break
            
            desire_root_trans_ic_list = []
            desire_root_orien_ic_list = []
            for desire_frame in desire_frames:
                desire_root_trans = motion.joint_position[desire_frame][0]
                desire_root_orien = motion.joint_rotation[desire_frame][0]

                desire_root_trans_ic = get_local_pos_in_coord(c_root_trans, c_root_orien, desire_root_trans)
                desire_root_orien_ic = get_local_rot_in_coord(c_root_trans, c_root_orien, desire_root_orien)

                desire_root_trans_ic_list.append(desire_root_trans_ic)
                desire_root_orien_ic_list.append(desire_root_orien_ic)

            
            input = construct_input_data(l_root_trans_ic, l_root_orien_ic, c_key_joints_pos_list, l_key_joints_trans_icj_list, desire_root_trans_ic_list, desire_root_orien_ic_list)
            features = np.concatenate((features, input), axis=0)



            n_frame = c_frame + 1

            n_root_trans = motion.joint_position[n_frame][0]
            n_root_orien = motion.joint_rotation[n_frame][0]

            n_root_trans_ic = get_local_pos_in_coord(c_root_trans, c_root_orien, n_root_trans)
            n_root_orien_ic = get_local_rot_in_coord(c_root_trans, c_root_orien, n_root_orien)

            n_joints_no_root_rot_list = []
            for joint in range(1, joints_size):
                n_joints_no_root_rot_list.append(motion.joint_rotation[n_frame][joint])

            n_root_trans_ic[1] = 0
            n_root_orien_ic, _ = motion.decompose_rotation_with_yaxis(n_root_orien_ic)

            output = construct_output_data(n_root_trans_ic, n_root_orien_ic, n_joints_no_root_rot_list)
            labels = np.concatenate((labels, output), axis=0)

    return features, labels



motion_files = [
    'motion_material/idle.bvh',
    # 'motion_material/run_forward.bvh',
    'motion_material/walk_and_ture_right.bvh',
    'motion_material/walk_and_turn_left.bvh',
    'motion_material/walk_forward.bvh',
    # 'motion_material/walkF.bvh',

    # 'motion_material/physics_motion/long_run.bvh',
    # 'motion_material/physics_motion/long_run_mirror.bvh',
    # 'motion_material/physics_motion/long_walk.bvh',
    # 'motion_material/physics_motion/long_walk_mirror.bvh',

    # 'motion_material/kinematic_motion/long_run.bvh',
    # 'motion_material/kinematic_motion/long_run_mirror.bvh',
    # 'motion_material/kinematic_motion/long_walk.bvh',
    # 'motion_material/kinematic_motion/long_walk_mirror.bvh',
]
key_joints = [4,9]
desire_frames_delta = [20, 40, 60, 80, 100]
joints_size = 25
features, labels = load_matching_data(motion_files, key_joints, desire_frames_delta, joints_size)


np.savetxt('motion_material/_features.csv', features, fmt='%f', delimiter=',')
np.savetxt('motion_material/_labels.csv', labels, fmt='%f', delimiter=',')
print('features.shape:', features.shape, 'labels.shape:', labels.shape)
# n_features = np.loadtxt('motion_material/_features.csv', dtype=np.float, delimiter=',', unpack=False)
# n_labels = np.loadtxt('motion_material/_labels.csv', dtype=np.float, delimiter=',', unpack=False)
# print('n_features.shape:', n_features.shape, 'n_labels.shape:', n_labels.shape)

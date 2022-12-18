from answer_task1 import *

def get_local_position_in_coordinate(position, rotation, world_position):
    l_position = position
    l_rotation = R.from_quat(rotation)

    delta_wolrd_position = world_position - l_position
    return R.inv(l_rotation).as_matrix() @ delta_wolrd_position

def get_local_rotation_in_coordinate(position, rotation, world_rotation):
    l_rotation = R.from_quat(rotation)

    return (R.inv(l_rotation) * R.from_quat(world_rotation)).as_quat()

def get_local_position_root_coordinate(motion, frame, world_position):
    root_position = motion.joint_position[frame][0]
    root_rotation = R.from_quat(motion.joint_rotation[frame][0])

    delta_wolrd_position = world_position - root_position
    return R.inv(root_rotation).as_matrix() @ delta_wolrd_position

def get_local_rotation_root_coordinate(motion, frame, world_rotation):
    root_rotation = R.from_quat(motion.joint_rotation[frame][0])
    return (R.inv(root_rotation) * R.from_quat(world_rotation)).as_quat()

def construct_input_data(last_root_local_position, last_root_local_rotation, now_key_joint_local_positions, delta_last_joint_local_positions, desire_root_local_positions, desire_root_local_rotations):
    input = np.array([])

    input = np.append(input, last_root_local_position)
    input = np.append(input, last_root_local_rotation)

    for now_key_joint_local_position in now_key_joint_local_positions:
        input = np.append(input, now_key_joint_local_position)

    for delta_last_joint_local_position in delta_last_joint_local_positions:
        input = np.append(input, delta_last_joint_local_position)

    for desire_root_local_position in desire_root_local_positions:
        input = np.append(input, desire_root_local_position)

    for desire_root_local_rotation in desire_root_local_rotations:
        input = np.append(input, desire_root_local_rotation) 

    return input  

def deconstruct_input_data(input, key_joints, desire_frames):
    index = 0

    last_root_local_position = input[index:index + 3]
    index = index + 3
    last_root_local_rotation = input[index:index + 4]
    index = index + 4

    now_key_joint_local_positions = []
    for _ in key_joints:
        now_key_joint_local_positions.append(input[index:index + 3])
        index = index + 3

    delta_last_joint_local_positions = []
    for _ in key_joints:
        delta_last_joint_local_positions.append(input[index:index + 3])
        index = index + 3

    desire_root_local_positions = []
    for _ in key_joints:
        desire_root_local_positions.append(input[index:index + 3])
        index = index + 3
    
    desire_root_local_rotations = []
    for _ in key_joints:
        desire_root_local_rotations.append(input[index:index + 4])
        index = index + 4
        
    return last_root_local_position, last_root_local_rotation, now_key_joint_local_positions, delta_last_joint_local_positions, desire_root_local_positions, desire_root_local_rotations      

def construct_output_data(next_root_local_position, next_root_local_rotation, delta_next_joint_local_positions, delta_next_joint_local_rotations):
    input = np.array([])

    input = np.append(input, next_root_local_position)
    input = np.append(input, next_root_local_rotation)

    for delta_next_joint_local_position in delta_next_joint_local_positions:
        input = np.append(input, delta_next_joint_local_position)

    for delta_next_joint_local_rotation in delta_next_joint_local_rotations:
        input = np.append(input, delta_next_joint_local_rotation)

    return input  

def deconstruct_output_data(output, joints_size):
    index = 0

    next_root_local_position = input[index:index + 3]
    index = index + 3
    next_root_local_rotation = input[index:index + 4]
    index = index + 4

    delta_next_joint_local_positions = []
    for _ in range(1, joints_size):
        delta_next_joint_local_positions.append(input[index:index + 3])
        index = index + 3

    delta_next_joint_local_rotations = []
    for _ in (1, joints_size):
        delta_next_joint_local_rotations.append(input[index:index + 3])
        index = index + 3
        
    return next_root_local_position, next_root_local_rotation, delta_next_joint_local_positions, delta_next_joint_local_rotations 

def load_matching_data(files, key_joints, desire_frames_delta, joints_size):
    features = np.zeros((0, 7 + 6 * len(key_joints) + 7 * len(desire_frames_delta)))
    labels = np.zeros((0, 7 + 7 * (joints_size - 1)))

    for file in files:
        motion = BVHMotion(file)

        joint_translation, joint_orientation = motion.batch_forward_kinematics()
        for frame in range(1, motion.motion_length):
            last_frame = frame - 1

            last_root_world_position = motion.joint_position[last_frame][0]
            last_root_world_rotation = motion.joint_rotation[last_frame][0]

            last_root_local_position = get_local_position_root_coordinate(motion, frame, last_root_world_position)
            last_root_local_rotation = get_local_rotation_root_coordinate(motion, frame, last_root_world_rotation)

            now_key_joint_local_positions = []
            delta_last_key_joint_local_positions = []
            for joint in key_joints:
                now_joint_world_position = joint_translation[frame][joint]
                now_joint_local_position = get_local_position_root_coordinate(motion, frame, now_joint_world_position)

                now_key_joint_local_positions.append(now_joint_local_position)

                last_joint_world_position = joint_translation[last_frame][joint]
                last_joint_local_position = get_local_position_root_coordinate(motion, last_frame, last_joint_world_position)

                delta_last_joint_local_position = last_joint_local_position - now_joint_local_position

                delta_last_key_joint_local_positions.append(delta_last_joint_local_position)

            desire_frames = []
            for delta in desire_frames_delta:
                desire_frames.append(frame + delta)
            if desire_frames[-1] >= motion.motion_length:
                break
            
            desire_root_local_positions = []
            desire_root_local_rotations = []
            for desire_frame in desire_frames:
                desire_root_world_position = motion.joint_position[desire_frame][0]
                desire_root_world_rotation = motion.joint_rotation[desire_frame][0]

                desire_root_local_position = get_local_position_root_coordinate(motion, frame, desire_root_world_position)
                desire_root_local_rotation = get_local_rotation_root_coordinate(motion, frame, desire_root_world_rotation)

                desire_root_local_positions.append(desire_root_local_position)
                desire_root_local_rotations.append(desire_root_local_rotation)

            
            input = construct_input_data(last_root_local_position, last_root_local_rotation, now_key_joint_local_positions, delta_last_key_joint_local_positions, desire_root_local_positions, desire_root_local_rotations)
            input = input.reshape(1, -1)
            features = np.concatenate((features, input), axis=0)



            next_frame = frame + 1

            next_root_world_position = motion.joint_position[next_frame][0]
            next_root_world_rotation = motion.joint_rotation[next_frame][0]

            next_root_local_position = get_local_position_root_coordinate(motion, frame, next_root_world_position)
            next_root_local_rotation = get_local_rotation_root_coordinate(motion, frame, next_root_world_rotation)

            delta_next_joint_local_positions = []
            delta_next_joint_local_rotations = []
            for joint in range(1, joints_size):
                delta_next_joint_local_position = motion.joint_position[next_frame][joint] - motion.joint_position[frame][joint]
                delta_next_joint_local_rotation = (R.inv(R.from_quat(motion.joint_rotation[frame][joint])) * R.from_quat(motion.joint_rotation[next_frame][joint])).as_quat()

                delta_next_joint_local_positions.append(delta_next_joint_local_position)
                delta_next_joint_local_rotations.append(delta_next_joint_local_rotation)

            output = construct_output_data(next_root_local_position, next_root_local_rotation, delta_next_joint_local_positions, delta_next_joint_local_rotations)
            output = output.reshape(1, -1)
            labels = np.concatenate((labels, output), axis=0)

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

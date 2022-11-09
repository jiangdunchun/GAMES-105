import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_stack_list = []
    offset_list = []
    my_joint_dict = {}

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            next_line = lines[i+1]
            current_line_names = [name for name in line.split()]
            next_line_names = [name for name in next_line.split()]
            if next_line_names[0] == "{":
                current_joint_name = current_line_names[1]
                if current_line_names[0] == "End":
                    current_joint_name = joint_name[-1]+"_end"
                joint_stack_list.append(current_joint_name)
            if current_line_names[0] == "ROOT" or current_line_names[0] == "JOINT":
                joint_name.append(current_line_names[1])
            if current_line_names[0] == "End":
                joint_name.append(joint_name[-1]+"_end")
            if current_line_names[0] == "OFFSET":
                offset_list.append([float(current_line_names[1]), float(current_line_names[2]), float(current_line_names[3])])
            if next_line_names[0] == "}":
                pop_joint_name = joint_stack_list.pop()
                if joint_stack_list == []:
                    pop_joint_name_parent = None
                else:
                    pop_joint_name_parent = joint_stack_list[-1]
                my_joint_dict[pop_joint_name] = pop_joint_name_parent
                if pop_joint_name_parent == None:
                    break

    for i in range(len(joint_name)):
        parent = my_joint_dict[joint_name[i]]
        if parent is None:
            parent_id = -1
        else:
            parent_id = joint_name.index(parent)
        joint_parent.append(parent_id)

    joint_offset = np.array(offset_list)

    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """
    joint_positions = np.zeros((len(joint_name), 3), dtype=np.float)
    joint_orientations = np.zeros((len(joint_name), 4), dtype=np.float)

    rot_channel = 1
    for i in range(len(joint_name)):
        parent_pos = np.array([0.0, 0.0, 0.0])
        parent_rot = np.array([0.0, 0.0, 0.0, 1.0])
        if joint_parent[i] >= 0:
            parent_pos = joint_positions[joint_parent[i]]
            parent_rot = joint_orientations[joint_parent[i]]

        my_offset = joint_offset[i]
        my_rot = np.array([0.0, 0.0, 0.0, 1.0])
        if i == 0:
            my_offset = motion_data[frame_id][0:3]
        if not joint_name[i].endswith("_end"):
            my_rot = R.from_euler('XYZ', motion_data[frame_id][rot_channel*3:(rot_channel+1)*3], degrees=True).as_quat()
            rot_channel += 1

        joint_positions[i] = parent_pos + np.inner(R.from_quat(parent_rot).as_matrix(), my_offset)
        joint_orientations[i] = (R.from_quat(parent_rot) * R.from_quat(my_rot)).as_quat()

    return joint_positions, joint_orientations

def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
    """
    
    print("ATTENTION: this function is really slow, you just wait, wait for sunshine!!!!!!!")

    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    A_motion_data = load_motion_data(A_pose_bvh_path)
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    T_motion_data = np.zeros(A_motion_data.shape, dtype=np.float)

    rot_channel = 1
    T_rot_channels = []
    for i in range(len(T_joint_name)):
        if not T_joint_name[i].endswith("_end"):
            T_rot_channels.append(rot_channel)
            rot_channel += 1
        else:
            T_rot_channels.append(-1)
    
    delta_qot_mats = []
    for T_m_index in range(len(T_joint_name)):
        A_m_index = A_joint_name.index(T_joint_name[T_m_index])

        delta_qot_quat = np.array([0.0, 0.0, 0.0, 1.0])
        T_m_offset = T_joint_offset[T_m_index]
        if np.linalg.norm(T_m_offset) != 0:
            T_m_offset = T_m_offset / np.linalg.norm(T_m_offset)
            A_m_offset = A_joint_offset[A_m_index]
            A_m_offset = A_m_offset / np.linalg.norm(A_m_offset)
            half = T_m_offset + A_m_offset
            half = half / np.linalg.norm(half)

            cos_theta_2 = np.dot(T_m_offset, half)
            normal = np.cross(T_m_offset, half)
            sin_theta_2 = np.linalg.norm(normal)
            if sin_theta_2 != 0:
                normal = normal / sin_theta_2
                delta_qot_quat[0] = normal[0] * sin_theta_2
                delta_qot_quat[1] = normal[1] * sin_theta_2
                delta_qot_quat[2] = normal[2] * sin_theta_2
                delta_qot_quat[3] = cos_theta_2

        delta_qot_mats.append(R.from_quat(delta_qot_quat).as_matrix())

    for f in range(A_motion_data.shape[0]):
        T_motion_data[f][0:3] = A_motion_data[f][0:3]

        _, A_joint_orientations = part2_forward_kinematics(A_joint_name, A_joint_parent, A_joint_offset, A_motion_data, f)
        T_joint_orientations = np.zeros(A_joint_orientations.shape, dtype=np.float)

        for T_m_index in range(len(T_joint_name)):
            T_p_index = T_joint_parent[T_m_index]
            if T_p_index < 0: continue
            A_p_index = A_joint_name.index(T_joint_name[T_p_index])

            T_gp_w_rot_quat = np.array([0.0, 0.0, 0.0, 1.0])
            T_gp_index = T_joint_parent[T_p_index]
            if T_gp_index >= 0:
                T_gp_w_rot_quat = T_joint_orientations[T_gp_index]

            A_p_w_rot_quat = A_joint_orientations[A_p_index]

            T_gp_w_rot_mat = R.from_quat(T_gp_w_rot_quat).as_matrix()
            A_p_w_rot_mat = R.from_quat(A_p_w_rot_quat).as_matrix()
            delta_qot_mat = delta_qot_mats[T_m_index]

            T_p_l_rot_eualr = R.from_matrix(np.linalg.inv(T_gp_w_rot_mat) @ A_p_w_rot_mat @ delta_qot_mat).as_euler('XYZ', degrees=True)

            T_p_rot_channel = T_rot_channels[T_p_index]
            T_motion_data[f][T_p_rot_channel*3:(T_p_rot_channel+1)*3] = T_p_l_rot_eualr
            T_joint_orientations[T_p_index] = (R.from_quat(T_gp_w_rot_quat) * R.from_euler('XYZ', T_p_l_rot_eualr, degrees=True)).as_quat()

    return T_motion_data
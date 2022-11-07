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

        my_pos = joint_offset[i]
        my_rot = np.array([0.0, 0.0, 0.0, 1.0])
        if i == 0:
            my_pos = motion_data[frame_id][0:3]
        if not joint_name[i].endswith("_end"):
            my_rot = R.from_euler('XYZ', motion_data[frame_id][rot_channel*3:(rot_channel+1)*3], degrees=True).as_quat()
            rot_channel += 1

        joint_positions[i] = parent_pos + np.inner(R.from_quat(parent_rot).as_matrix(), my_pos)
        joint_orientations[i] = (R.from_quat(my_rot) * R.from_quat(parent_rot)).as_quat()

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
    
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    A_motion_data = load_motion_data(A_pose_bvh_path)
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    T_motion_data = np.zeros(A_motion_data.shape, dtype=np.float)

    for f in range(A_motion_data.shape[0]):
        A_joint_positions, A_joint_orientations = part2_forward_kinematics(A_joint_name, A_joint_parent, A_joint_offset, A_motion_data, f)
        rot_channel = 1

        for i in range(len(T_joint_name)):
            A_index = A_joint_name.index(T_joint_name[i])
            A_position = A_joint_positions[A_index]
            A_orientation = A_joint_orientations[A_index]

            if T_joint_parent[i] < 0:
                T_motion_data[f][0:3] = A_position

            if not T_joint_name[i].endswith("_end"):
                # @todo     
                T_motion_data[f][rot_channel*3:(rot_channel+1)*3] = (R.from_quat([0.0,0.0,0.0,1.0])).as_euler('XYZ', degrees=True)
                rot_channel += 1

    return T_motion_data
import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, path_name, path_e2r, path_r2r = meta_data.get_path_from_root_to_end()
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position

    local_orientations = np.copy(joint_orientations)
    end_position = joint_positions[path_e2r[0]]

    for iter in range(32):
        for p_index in path_e2r[1:]:
            p_position = joint_positions[p_index]

            from_vec = end_position - p_position
            to_vec = target_pose - p_position
            end_position = p_position + np.linalg.norm(from_vec) * to_vec / np.linalg.norm(to_vec)

            from_vec_norm = from_vec / np.linalg.norm(from_vec)
            to_vec_norm = to_vec / np.linalg.norm(to_vec)
            half = from_vec_norm + to_vec_norm
            half = half / np.linalg.norm(half)
            delta_qot_quat = np.array([0.0, 0.0, 0.0, 1.0])
            cos_theta_2 = np.dot(from_vec_norm, half)
            normal = np.cross(from_vec_norm, half)
            sin_theta_2 = np.linalg.norm(normal)
            if sin_theta_2 != 0:
                normal = normal / sin_theta_2
                delta_qot_quat[0] = normal[0] * sin_theta_2
                delta_qot_quat[1] = normal[1] * sin_theta_2
                delta_qot_quat[2] = normal[2] * sin_theta_2
                delta_qot_quat[3] = cos_theta_2

            gp_orientation = R.from_quat(np.array([0.0, 0.0, 0.0, 1.0]))
            if joint_parent[p_index] >= 0:
                gp_orientation = R.from_quat(joint_orientations[joint_parent[p_index]])
            gp_orientation_inv = R.from_matrix(np.linalg.inv(gp_orientation.as_matrix()))

            local_orientations[p_index] = (gp_orientation_inv * R.from_quat(delta_qot_quat) * gp_orientation * R.from_quat(local_orientations[p_index])).as_quat()

        for p_index in path_r2r[-1:0:-1]:
            print(p_index, end=" ")

        for m_index in range(len(joint_parent)):
            p_index = joint_parent[m_index]
            if p_index >= 0:
                offset = joint_initial_position[m_index] - joint_initial_position[p_index]
                p_position = joint_positions[p_index]
                joint_positions[m_index] = p_position + np.inner(R.from_quat(joint_orientations[p_index]).as_matrix(), offset)
                joint_orientations[m_index] = (R.from_quat(joint_orientations[p_index]) * R.from_quat(local_orientations[m_index])).as_quat()

        distance = np.linalg.norm(end_position - target_pose)
        if distance < 0.0001:
            break

    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy

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
    path, path_name, _, path_r2r = meta_data.get_path_from_root_to_end()
    joint_initial_position = meta_data.joint_initial_position
    joint_parent = copy.deepcopy(meta_data.joint_parent)
    update_order =[]
    for i in range(len(path_r2r)):
        update_order.append(path_r2r[i])
        if i == 0:
            joint_parent[path_r2r[i]] = -1
        else:
            joint_parent[path_r2r[i]] = path_r2r[i-1]
    for i in range(len(joint_parent)):
        if i not in update_order:
            update_order.append(i)
    
    path_e2r = path
    path_e2r.reverse()
    local_orientations = []
    for m_index in range(len(joint_parent)):
        p_index = joint_parent[m_index]
        p_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        if p_index >= 0:
            p_orientation = joint_orientations[p_index]
        local_orientations.append(R.inv(R.from_quat(p_orientation)) * R.from_quat(joint_orientations[m_index]))

    for iter in range(100):
        V = target_pose - joint_positions[path_e2r[0]]
        V = V.reshape(-1, 1)
        J = np.zeros([3, len(path_e2r) - 1], dtype=float)
        normals = []

        for i in range(1, len(path_e2r)):
            p_position = joint_positions[path_e2r[i]]

            from_vec = joint_positions[path_e2r[0]] - p_position
            to_vec = target_pose - p_position
            normal = np.cross(from_vec, to_vec)
            normal = normal / np.linalg.norm(normal)

            dpdi = np.cross(normal, from_vec)
            J[0][i-1] = dpdi[0]
            J[1][i-1] = dpdi[1]
            J[2][i-1] = dpdi[2]

            normals.append(normal)

        J_T = np.transpose(J)
        dtheta = 0.5 * J_T @ V

        for i in range(1, len(path_e2r)):
            delta_theta = dtheta[i-1][0]
            normal = normals[i-1]
            delta_rot = R.from_rotvec(delta_theta * normal)
            local_orientations[path_e2r[i]] = delta_rot * local_orientations[path_e2r[i]]

        for m_index in update_order:
            p_index = joint_parent[m_index]
            if p_index >= 0:
                offset = joint_initial_position[m_index] - joint_initial_position[p_index]
                p_position = joint_positions[p_index]
                joint_positions[m_index] = p_position + np.inner(R.from_quat(joint_orientations[p_index]).as_matrix(), offset)
                joint_orientations[m_index] = (R.from_quat(joint_orientations[p_index]) * local_orientations[m_index]).as_quat()

        distance = np.linalg.norm(joint_positions[path_e2r[0]] - target_pose)
        print(iter, " ", distance)
        if distance < 0.01:
            break
    
    joint_parent = meta_data.joint_parent
    for m_index in update_order:
        p_index = joint_parent[m_index]
        if p_index >= 0:
            from_vec = joint_initial_position[m_index] - joint_initial_position[p_index]
            to_vec = joint_positions[m_index] - joint_positions[p_index]
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
            joint_orientations[p_index] = delta_qot_quat

    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    path_e2r = [11, 10, 9]
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    target_pose = np.array(joint_positions[0])
    target_pose[0] = target_pose[0] + relative_x
    target_pose[1] = target_height
    target_pose[2] = target_pose[2] + relative_z


    local_orientations = []
    for m_index in range(len(joint_parent)):
        p_index = joint_parent[m_index]
        p_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        if p_index >= 0:
            p_orientation = joint_orientations[p_index]
        local_orientations.append(R.inv(R.from_quat(p_orientation)) * R.from_quat(joint_orientations[m_index]))

    for iter in range(100):
        V = target_pose - joint_positions[path_e2r[0]]
        V = V.reshape(-1, 1)
        J = np.zeros([3, len(path_e2r) - 1], dtype=float)
        normals = []

        for i in range(1, len(path_e2r)):
            p_position = joint_positions[path_e2r[i]]

            from_vec = joint_positions[path_e2r[0]] - p_position
            to_vec = target_pose - p_position
            normal = np.cross(from_vec, to_vec)
            normal = normal / np.linalg.norm(normal)

            dpdi = np.cross(normal, from_vec)
            J[0][i-1] = dpdi[0]
            J[1][i-1] = dpdi[1]
            J[2][i-1] = dpdi[2]

            normals.append(normal)

        J_T = np.transpose(J)
        dtheta = 1.0 * J_T @ V

        for i in range(1, len(path_e2r)):
            delta_theta = dtheta[i-1][0]
            normal = normals[i-1]
            delta_rot = R.from_rotvec(delta_theta * normal)
            local_orientations[path_e2r[i]] = delta_rot * local_orientations[path_e2r[i]]

        for m_index in range(len(joint_parent)):
            p_index = joint_parent[m_index]
            if p_index >= 0:
                offset = joint_initial_position[m_index] - joint_initial_position[p_index]
                p_position = joint_positions[p_index]
                joint_positions[m_index] = p_position + np.inner(R.from_quat(joint_orientations[p_index]).as_matrix(), offset)
                joint_orientations[m_index] = (R.from_quat(joint_orientations[p_index]) * local_orientations[m_index]).as_quat()

        distance = np.linalg.norm(joint_positions[path_e2r[0]] - target_pose)
        print(iter, " ", distance)
        if distance < 0.01:
            break

    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations
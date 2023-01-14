from bvh_utils import *
#---------------你的代码------------------#
def dot(v1, v2):
    ret = np.zeros(v1.shape)
    ret[:,:,0] = v1[:,:,1] * v2[:,:,2] - v1[:,:,2] * v2[:,:,1]
    ret[:,:,1] = v1[:,:,2] * v2[:,:,0] - v1[:,:,0] * v2[:,:,2]
    ret[:,:,2] = v1[:,:,0] * v2[:,:,1] - v1[:,:,1] * v2[:,:,0]
    return ret

def multiply(quat, vec):
    qvec = quat[:,:,0:3]
    uv = dot(qvec, vec)
    uuv = dot(qvec, uv)
    uv = np.einsum('ijk, ij->ijk', uv, 2.0 * quat[:,:,3])
    uuv = 2.0 * uuv
    return vec + uv + uuv

# translation 和 orientation 都是全局的
T_offset = np.zeros((0,0,0))
def skinning(joint_translation, joint_orientation, T_pose_joint_translation, T_pose_vertex_translation, skinning_idx, skinning_weight):
    """
    skinning函数，给出一桢骨骼的位姿，计算蒙皮顶点的位置
    假设M个关节，N个蒙皮顶点，每个顶点受到最多4个关节影响
    输入：
        joint_translation: (M,3)的ndarray, 目标关节的位置
        joint_orientation: (M,4)的ndarray, 目标关节的旋转，用四元数表示
        T_pose_joint_translation: (M,3)的ndarray, T pose下关节的位置
        T_pose_vertex_translation: (N,3)的ndarray, T pose下蒙皮顶点的位置
        skinning_idx: (N,4)的ndarray, 每个顶点受到哪些关节的影响（假设最多受4个关节影响）
        skinning_weight: (N,4)的ndarray, 每个顶点受到对应关节影响的权重
    输出：
        vertex_translation: (N,3)的ndarray, 蒙皮顶点的位置
    """
    vertex_translation = T_pose_vertex_translation.copy()
    
    #---------------你的代码------------------#
    N = T_pose_vertex_translation.shape[0]

    global T_offset
    if T_offset.shape[0] == 0:
        T_offset = np.zeros((4, N, 3))
        for v_index in range(N):
            for j in range(4):
                j_index = skinning_idx[v_index][j]
                T_offset[j][v_index] = T_pose_joint_translation[j_index]

        for j in range(4):
            T_offset[j] = T_pose_vertex_translation - T_offset[j]

    j_position = np.zeros((4, N, 3))
    j_rotation = np.zeros((4, N, 4))
    for v_index in range(N):
        for j in range(4):
            j_index = skinning_idx[v_index][j]
            j_position[j][v_index] = joint_translation[j_index]
            j_rotation[j][v_index] = joint_orientation[j_index]

    j_offset = multiply(j_rotation, T_offset)
    w_position = j_offset + j_position
    vertex_translation = np.einsum('ijk, ji->jk', w_position, skinning_weight)

    return vertex_translation
import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_loader import BVHMotion
from physics_warpper import PhysicsInfo


def part1_cal_torque(pose, physics_info: PhysicsInfo, **kargs):
    '''
    输入： pose： (20,4)的numpy数组，表示每个关节的目标旋转(相对于父关节的)
           physics_info: PhysicsInfo类，包含了当前的物理信息，参见physics_warpper.py
           **kargs：指定参数，可能包含kp,kd
    输出： global_torque: (20,3)的numpy数组，表示每个关节的全局坐标下的目标力矩，根节点力矩会被后续代码无视
    '''
    # ------一些提示代码，你可以随意修改------------#
    '''
    0 'RootJoint'
    1 'pelvis_lowerback'
    2 'lowerback_torso'
    3 'rHip'
    4 'lHip'
    5 'rKnee'
    6 'lKnee'
    7 'rAnkle'
    8 'lAnkle'
    9 'rToeJoint'
    10 'lToeJoint'
    11 'torso_head'
    12 'rTorso_Clavicle'
    13 'lTorso_Clavicle'
    14 'rShoulder'
    15 'lShoulder'
    16 'rElbow'
    17 'lElbow'
    18 'rWrist'
    19 'lWrist'
    '''
    # ----------------------------|0    |1    |2    |3    |4    |5    |6    |7    |8    |9    |10   |11   |12   |13   |14   |15   |16   |17   |18   |19   #
    kp = kargs.get('kp', np.array([500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ,500  ]))
    kd = kargs.get('kd', np.array([20   ,20   ,20   ,20   ,20   ,20   ,20   ,20   ,20   ,20   ,20   ,60   ,20   ,20   ,60   ,60   ,20   ,20   ,20   ,20   ])) 
    parent_index = physics_info.parent_index
    joint_name = physics_info.joint_name
    joint_orientation = physics_info.get_joint_orientation()
    joint_avel = physics_info.get_body_angular_velocity()

    global_torque = np.zeros((20,3))
    for j in range(20):
        p_orientation = R.identity()
        p = parent_index[j]
        if p != -1:
            p_orientation = R.from_quat(joint_orientation[p])

        j_rotation = (R.inv(p_orientation) * R.from_quat(joint_orientation[j])).as_euler('XYZ', degrees=True)
        j_dst_rotation = R.from_quat(pose[j]).as_euler('XYZ', degrees=True)
        j_avel = R.inv(p_orientation).as_matrix() @ joint_avel[j]
        torque = kp[j] * (j_dst_rotation - j_rotation) - kd[j] * j_avel
        torque = p_orientation.as_matrix() @ torque
        # torque = np.clip(torque, -10, 10)

        global_torque[j] = global_torque[j] + torque
        if p != -1:
            global_torque[p] = global_torque[p] - torque
    
    return global_torque

def part2_cal_float_base_torque(target_position, pose, physics_info, **kargs):
    '''
    输入： target_position: (3,)的numpy数组，表示根节点的目标位置，其余同上
    输出： global_root_force: (3,)的numpy数组，表示根节点的全局坐标下的辅助力
          global_torque: 同上
    注意：
        1. 你需要自己计算kp和kd，并且可以通过kargs调整part1中的kp和kd
        2. global_torque[0]在track静止姿态时会被无视，但是track走路时会被加到根节点上，不然无法保持根节点朝向
    '''
    global_torque = part1_cal_torque(pose, physics_info)
    kp = kargs.get('root_kp', 4000) # 需要自行调整root的kp和kd！
    kd = kargs.get('root_kd', 20)
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    global_root_force = np.zeros((3,))

    global_root_force = kp * (target_position - root_position) - kd * root_velocity
    return global_root_force, global_torque

def part3_cal_static_standing_torque(bvh: BVHMotion, physics_info):
    '''
    输入： bvh: BVHMotion类，包含了当前的动作信息，参见bvh_loader.py
    其余同上
    Tips: 
        只track第0帧就能保持站立了
        为了保持平衡可以把目标的根节点位置适当前移，比如把根节点位置和左右脚的中点加权平均
        为了仿真稳定最好不要在Toe关节上加额外力矩
    '''
    tar_pos = bvh.joint_position[0][0]
    pose = bvh.joint_rotation[0]
    joint_name = physics_info.joint_name
    
    joint_positions = physics_info.get_joint_translation()
    # 适当前移
    tar_pos = tar_pos * 0.8 + joint_positions[9] * 0.1 + joint_positions[10] * 0.1

    torque = np.zeros((20,3))

    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    global_root_force = 4000 * (tar_pos - root_position) - 60 * root_velocity
    torque = part1_cal_torque(pose, physics_info)
    for j in range(3, 9):
        torque[j] = torque[j] + np.cross(root_position - joint_positions[j], global_root_force)

    return torque


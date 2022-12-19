# 以下部分均为可更改部分

from answer_task1 import *
from motion_data_generator import *

class CharacterController():
    def __init__(self, controller) -> None:
        self.motions = []
        self.motions.append(BVHMotion('motion_material/idle.bvh'))
        self.controller = controller
        self.cur_root_pos = None
        self.cur_root_rot = None
        self.cur_frame = 0

        self.last_joints_position = None
        self.last_joints_rotation = None
        self.cur_joints_position = None
        self.cur_joints_rotation = None
        self.key_joints = [4,9]
        self.desire_frames_delta = [20, 40, 60, 80, 100]
        self.joints_size = 25
        self.features = np.loadtxt('motion_material/_features.csv', dtype =np.float, delimiter=',', unpack=False)
        self.labels = np.loadtxt('motion_material/_labels.csv', dtype=np.float, delimiter=',', unpack=False)
        self.first_flag = True
        pass

    def get_min_lable(self, last_root_local_position, last_root_local_rotation, now_key_joint_local_positions, delta_last_key_joint_world_positions, desire_root_local_positions, desire_root_local_rotations):
        data_size = features.shape[0]
        min_cost = 1e10
        next_root_local_position, next_root_local_rotation, last_key_joint_local_rotations = deconstruct_output_data(self.labels[0], self.joints_size)
        for iter in range(0, data_size):
            input = self.features[iter]
            nlast_root_local_position, nlast_root_local_rotation, nnow_key_joint_local_positions, ndelta_last_key_joint_world_positions, ndesire_root_local_positions, ndesire_root_local_rotations = deconstruct_input_data(input, self.key_joints, self.desire_frames_delta)
            now_cost = 0
            now_cost = now_cost + np.linalg.norm(last_root_local_position - nlast_root_local_position)
            now_cost = now_cost + np.linalg.norm(last_root_local_rotation - nlast_root_local_rotation) 

            for j in range(len(now_key_joint_local_positions)):
                now_cost = now_cost + 10 * np.linalg.norm(now_key_joint_local_positions[j] - nnow_key_joint_local_positions[j])
                now_cost = now_cost + 10 * np.linalg. norm(delta_last_key_joint_world_positions[j] - ndelta_last_key_joint_world_positions[j])

            for d in range(len(desire_root_local_positions)):
                now_cost = now_cost + np.linalg.norm(desire_root_local_positions[d] - ndesire_root_local_positions[d])
                now_cost = now_cost + np.linalg.norm(desire_root_local_rotations[d] - ndesire_root_local_rotations[d])

            if (now_cost < min_cost):
                min_cost = now_cost
                next_root_local_position, next_root_local_rotation, last_key_joint_local_rotations = deconstruct_output_data(self.labels[iter], self.joints_size)

        return next_root_local_position, next_root_local_rotation, last_key_joint_local_rotations
    
    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他
        
        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        # 一个简单的例子，输出第i帧的状态
        if (self.first_flag):
            self.first_flag = False

            self.last_joints_position = self.motions[0].joint_position[0]
            self.last_joints_rotation = self.motions[0].joint_rotation[0]
            self.cur_joints_position = self.motions[0].joint_position[1]
            self.cur_joints_rotation = self.motions[0].joint_rotation[1]

        last_root_pos = self.last_joints_position[0]
        last_root_rot = self.last_joints_rotation[0]
        cur_root_pos = self.cur_joints_position[0]
        cur_root_rot = self.cur_joints_rotation[0]
        last_root_local_position = get_local_position_in_coordinate(cur_root_pos, cur_root_rot, last_root_pos)
        last_root_local_rotation = get_local_rotation_in_coordinate(cur_root_pos, cur_root_rot, last_root_rot)

        last_joint_translation, last_joint_orientation = self.motions[0].batch_forward_kinematics(np.array([self.last_joints_position]), np.array([self.last_joints_rotation]))
        cur_joint_translation, cur_joint_orientation = self.motions[0].batch_forward_kinematics(np.array([self.cur_joints_position]), np.array([self.cur_joints_rotation]))
        now_key_joint_local_positions = []
        delta_last_key_joint_world_positions = []
        for joint in self.key_joints:
            now_joint_world_position = cur_joint_translation[0][joint]
            now_joint_local_position = get_local_position_in_coordinate(cur_root_pos, cur_root_rot, now_joint_world_position)
            now_key_joint_local_positions.append(now_joint_local_position)

            last_joint_world_position = last_joint_translation[0][joint]
            delta_last_key_joint_world_position = last_joint_world_position - now_joint_world_position
            delta_last_key_joint_world_positions.append(delta_last_key_joint_world_position)

        desire_root_local_positions = []
        desire_root_local_rotations = []
        for desire_frame in range(1, len(desired_pos_list)):
            desire_root_world_position = desired_pos_list[desire_frame]
            desire_root_world_rotation = desired_rot_list[desire_frame]

            desire_root_local_position = get_local_position_in_coordinate(desired_pos_list[0], desired_rot_list[0], desire_root_world_position)
            desire_root_local_rotation = get_local_rotation_in_coordinate(desired_pos_list[0], desired_rot_list[0], desire_root_world_rotation)

            desire_root_local_positions.append(desire_root_local_position)
            desire_root_local_rotations.append(desire_root_local_rotation)

        next_joints_position = np.copy(self.cur_joints_position)
        next_joints_rotation = np.copy(self.cur_joints_rotation)
        # todo
        next_root_local_position, next_root_local_rotation, last_key_joint_local_rotations = self.get_min_lable(last_root_local_position, last_root_local_rotation, now_key_joint_local_positions, delta_last_key_joint_world_positions, desire_root_local_positions, desire_root_local_rotations)
        next_joints_position[0] = next_joints_position[0] + R.from_quat(next_joints_rotation[0]).as_matrix() @ next_root_local_position
        next_joints_rotation[0] = (R.from_quat(next_joints_rotation[0]) * R.from_quat(next_root_local_rotation)).as_quat()
        for i in range(1, self.joints_size):
            next_joints_rotation[i] = last_key_joint_local_rotations[i - 1]

        self.last_joints_position = np.copy(self.cur_joints_position)
        self.last_joints_rotation = np.copy(self.cur_joints_rotation)

        self.cur_joints_position = next_joints_position
        self.cur_joints_rotation = next_joints_rotation

        
        joint_translation, joint_orientation = self.motions[0].batch_forward_kinematics(np.array([self.cur_joints_position]), np.array([self.cur_joints_rotation]))
        joint_translation = joint_translation[0]
        joint_orientation = joint_orientation[0]
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        return self.motions[0].joint_name, joint_translation, joint_orientation
    
    
    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''
        
        # 一个简单的例子，将手柄的位置与角色对齐
        controller.set_pos(self.cur_root_pos)
        controller.set_rot(self.cur_root_rot)
        
        return character_state
    # 你的其他代码,state matchine, motion matching, learning, etc.
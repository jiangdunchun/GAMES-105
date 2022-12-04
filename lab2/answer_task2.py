# 以下部分均为可更改部分
import torch
from torch import nn

from answer_task1 import *

class CharacterController():
    def __init__(self, controller) -> None:
        self.motions = []
        self.motions.append(BVHMotion('motion_material/walk_forward.bvh'))
        self.controller = controller
        self.cur_root_pos = None
        self.cur_root_rot = None
        self.cur_frame = 0

        self.currrent_joint_position = np.array([self.motions[0].joint_position[0]])
        self.currrent_joint_position[0][0][0] = 0
        self.currrent_joint_position[0][0][2] = 0
        self.currrent_joint_rotation = np.array([self.motions[0].joint_rotation[0]])
        self.net = torch.load('motion_material/_net.pth')
        self.net = self.net.to(device=torch.device('cpu'))
        pass
    
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
        joint_name = self.motions[0].joint_name
        # joint_translation, joint_orientation = self.motions[0].batch_forward_kinematics()
        # joint_translation = joint_translation[self.cur_frame]
        # joint_orientation = joint_orientation[self.cur_frame]
        
        # self.cur_root_pos = joint_translation[0]
        # self.cur_root_rot = joint_orientation[0]
        #self.cur_frame = (self.cur_frame + 1) % self.motions[0].motion_length

        

        if self.cur_frame != 0:
            l_trans_y = self.currrent_joint_position[0][0][1]
            l_trans_xz = np.array([self.currrent_joint_position[0][0][0], self.currrent_joint_position[0][0][2]])
            l_ori_y, l_ori_xz = self.motions[0].decompose_rotation_with_yaxis(self.currrent_joint_rotation[0][0])
            l_rots = self.currrent_joint_rotation[0][1:-1]

            n_trans_xz = np.array([desired_pos_list[0][0], desired_pos_list[0][2]])
            n_ori_y, _ = self.motions[0].decompose_rotation_with_yaxis(desired_rot_list[0])

            delta_trans_xz = (n_trans_xz - l_trans_xz) / 20
            ori_euler = (R.from_quat(n_ori_y) * R.inv(R.from_quat(l_ori_y))).as_euler('YXZ', degrees=False)
            delta_ori_y = np.array([0, np.sin(ori_euler[0]/40), 0, np.cos(ori_euler[0]/40)])

            motion_input = np.zeros((1, 103))
            motion_input[0][0] = l_trans_y
            motion_input[0][1:5] = l_ori_xz
            motion_input[0][5:97] = np.resize(l_rots, (1, 92))
            motion_input[0][97:99] = delta_trans_xz
            motion_input[0][99:103] = delta_ori_y

            X = torch.tensor(motion_input, dtype=torch.float32)
            y_hat = self.net(X)

            motion_output = y_hat.detach().numpy()
            delta_trans_y = motion_output[0][0]
            delta_ori_xz = motion_output[0][1:5]
            delta_rots = np.resize(motion_output[0][5:97], (23, 4))

            n_trans_y = l_trans_y + delta_trans_y
            n_trans_xz = l_trans_xz + delta_trans_xz
            n_ori_y = (R.from_quat(l_ori_y) * R.from_quat(delta_ori_y)).as_quat()
            n_ori_xz = (R.from_quat(l_ori_xz) * R.from_quat(delta_ori_xz / np.linalg.norm(delta_ori_xz))).as_quat()
            n_rots = np.copy(l_rots)
            for j in range(23):
                n_rots[j] = (R.from_quat(l_rots[j]) * R.from_quat(delta_rots[j] / np.linalg.norm(delta_rots[j]))).as_quat()

            self.currrent_joint_position[0][0] = np.array([n_trans_xz[0], n_trans_y, n_trans_xz[1]])
            self.currrent_joint_rotation[0][0] = (R.from_quat(n_ori_y) * R.from_quat(n_ori_xz)).as_quat()
            self.currrent_joint_rotation[0][1:24] = n_rots

        joint_translation, joint_orientation = self.motions[0].batch_forward_kinematics(self.currrent_joint_position, self.currrent_joint_rotation)
        joint_translation = joint_translation[0]
        joint_orientation = joint_orientation[0]
        
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]

        self.cur_frame = 1
        
        return joint_name, joint_translation, joint_orientation
    
    
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
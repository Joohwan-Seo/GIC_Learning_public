from gym import Env
from gym import spaces

import mujoco_py
import numpy as np
import time, csv, os, copy

from gym import utils

# import matplotlib.pyplot as plt
from gic_env.utils.robot_state import RobotState
from gic_env.utils.mujoco import set_state
from gic_env.utils.base import Primitive, PrimitiveStatus



class RobotEnvSeparated(Env):
    def __init__(self, robot_name = 'fanuc', env_type = 'square_PIH', max_time = 15, show_viewer = False, 
                 obs_type = 'pos', hole_ori = 'default', testing = False, reward_version = None, window_size = 1, ECGIC = False, TCGIC = False, use_ext_force = True, act_type = 'default'):
        self.robot_name = robot_name
        self.env_type = env_type
        self.hole_ori = hole_ori
        self.testing = testing

        self.reward_version = reward_version
        self.window_size = window_size
        self.use_external_force = use_ext_force
        self.act_type = act_type
        print('===================')
        print('I AM IN DEFAULT ENV')
        print('===================')


        #NOTE(JS) The determinant of the desired rotation matrix should be always 1.
        # (by the definition of the rotational matrix.)
        if self.hole_ori == 'default':
            if self.testing:
                self.xd = np.array([0.60, 0.012, 0.05])
                self.Rd = np.array([[0, 1, 0],
                                    [1, 0, 0],
                                    [0, 0, -1]])
            else:
                self.xd = np.array([0.60, 0.012, 0.05])
                self.Rd = np.array([[0, 1, 0],
                                    [1, 0, 0],
                                    [0, 0, -1]])
            
        elif self.hole_ori == 'case1':
            self.xd = np.array([0.65, 0.1, 0.08])
            Rt = np.array([[1, 0, 0],
                           [0, 0.8660, -0.50],
                           [0,0.50,0.8660]])
            self.Rd = np.array([[0, 1, 0],
                                [1, 0, 0],
                                [0, 0, -1]])
            self.Rd = Rt @ self.Rd

        elif self.hole_ori == 'case2':
            self.xd = np.array([0.75, 0.00, 0.15])
            Rt = np.array([[0.8660, 0, -0.5],
                           [0, 1, 0],
                           [0.5, 0, 0.8660]])
            self.Rd = np.array([[0, 1, 0],
                                [1, 0, 0],
                                [0, 0, -1]])
            self.Rd = Rt @ self.Rd

        elif self.hole_ori == 'case3':
            self.xd = np.array([1.05, 0.00, 0.35])
            Rt = np.array([[0, 0, -1],
                           [0, 1, 0],
                           [1, 0, 0]])
            self.Rd = np.array([[0, 1, 0],
                                [1, 0, 0],
                                [0, 0, -1]])
            self.Rd = Rt @ self.Rd
        # self.xd = np.array([0.60, 0.012, 0.250])
        # self.Rd = np.array([[0, 0, 1],
        #                     [0, 1, 0],
        #                     [-1, 0, 0]])

        self.ECGIC = ECGIC #Exact Compensation GIC
        self.TCGIC = TCGIC #Total Compensation GIC, fixed-mass GIC
        if self.ECGIC:
            print('========ECGIC ON==========')
            self.lambda_g = 0.003

        self.show_viewer = show_viewer
        self.load_xml()
        self.obs_type = obs_type

        self.robot_state = RobotState(self.sim, "end_effector", self.robot_name)

        self.dt = 0.002
        self.max_iter = int(max_time/self.dt)
        self.dummy = 'dummy_change'

        self.logging = False
        self.csv_name = 'square_PIH_4'

        self.time_step = 0

        self.kt = 50
        self.ko = 10

        self.iter = 0

        # print('I am here')        

        if self.obs_type == 'pos_vel':
            self.num_obs = self.robot_state.N * 2
        elif self.obs_type == 'pos':
            self.num_obs = self.robot_state.N
        elif self.obs_type == 'pos_vel_force':
            self.num_obs = self.robot_state.N *2 + 6
        elif self.obs_type == 'pos_force':
            self.num_obs = self.robot_state.N + 6
        elif self.obs_type == 'feature':
            self.num_obs = 5 # distance, rot, trans, z_part, abs(dq)
        elif self.obs_type == 'pos_feature':
            self.num_obs = self.robot_state.N + 2
        elif self.obs_type == 'pos_feature2':
            self.num_obs = self.robot_state.N + 2
        elif self.obs_type == 'pos_feature3':
            self.num_obs = self.robot_state.N + 2

        if self.robot_name =='ur5e':
            self.num_act = 6
        elif self.robot_name == 'fanuc':
            self.num_act = 6

        if self.act_type == 'minimal':
            self.num_act = 2
        elif self.act_type == 'minimal2':
            self.num_act = 3
        elif self.act_type == 'minimal2-1':
            self.num_act = 3
        elif self.act_type == 'minimal3':
            self.num_act = 4

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs*self.window_size,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_act,))

        utils.EzPickle.__init__(self)

        self.prev_x = np.zeros((3,))
        self.stuck_count = 0
        self.done_count = 0

        if self.window_size is not 1:
            self.obs_memory = [np.zeros(self.num_obs)] * self.window_size

        self.Fe = np.zeros((6,1))
        self.reset()

    def load_xml(self):
        dir = "/home/joohwan/deeprl/research/GIC_RL/"
        if self.robot_name == 'ur5e':
            # model_path = "/home/joohwan/deeprl/research/GIC_learning/mujoco_models/pih/sliding_test.xml"
            if self.hole_ori == 'default':
                model_path = dir + "gic_env/mujoco_models/pih/square_pih_ur5e.xml"
            elif self.hole_ori == 'case1':
                model_path = dir + "gic_env/mujoco_models/pih/square_pih_ur5e_case1.xml"
            elif self.hole_ori == 'case2':
                model_path = dir + "gic_env/mujoco_models/pih/square_pih_ur5e_case2.xml"
            elif self.hole_ori == 'case3':
                model_path = dir + "gic_env/mujoco_models/pih/square_pih_ur5e_case3.xml"

        elif self.robot_name == 'fanuc':
            if self.hole_ori == 'default':
                if self.testing:
                    model_path = dir + "gic_env/mujoco_models/pih/square_pih_fanuc.xml"
                else:
                    model_path = dir + "gic_env/mujoco_models/pih/square_pih_fanuc.xml"
            elif self.hole_ori == 'case1':
                model_path = dir + "gic_env/mujoco_models/pih/square_pih_fanuc_case1.xml"
            elif self.hole_ori == 'case2':
                model_path = dir + "gic_env/mujoco_models/pih/square_pih_fanuc_case2.xml"
            elif self.hole_ori == 'case3':
                model_path = dir + "gic_env/mujoco_models/pih/square_pih_fanuc_case3.xml"

        elif self.robot_name == 'panda':
            # model_path = "/home/joohwan/deeprl/research/GIC_learning/mujoco_models/pih/sliding.xml"
            # model_path = "/home/joohwan/deeprl/research/GIC_learning/mujoco_models/pih/square_pih.xml"
            # model_path = "mujoco_models/pih/square_pih.xml"
            NotImplementedError

        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        if self.show_viewer:
            self.viewer = mujoco_py.MjViewer(self.sim)
        else:
            self.viewer = None

    def reset(self):
        # print('resetting')
        self.init_stage = True
        _ = self.initial_sample()
        obs = self._get_obs()

        self.iter = 0 
        self.prev_x = np.zeros((3,))
        self.stuck_count = 0
        self.done_count = 0

        xd_ori = self.xd
        Rd_ori = self.Rd

        if not self.testing:
            rand_xy = 2*(np.random.rand(2,) - 0.5) * 0.05
            rand_rpy = 2*(np.random.rand(3,) - 0.5) * 15 /180 * np.pi

            Rx = np.array([[1, 0, 0], [0, np.cos(rand_rpy[0]), -np.sin(rand_rpy[0])], [0, np.sin(rand_rpy[0]), np.cos(rand_rpy[0])]])
            Ry = np.array([[np.cos(rand_rpy[1]), 0, np.sin(rand_rpy[1])], [0, 1, 0], [-np.sin(rand_rpy[1]), 0, np.cos(rand_rpy[1])]])
            Rz = np.array([[np.cos(rand_rpy[2]), -np.sin(rand_rpy[2]), 0], [np.sin(rand_rpy[2]), np.cos(rand_rpy[2]), 0], [0, 0, 1]])

            self.xd = xd_ori.reshape((-1,1)) + Rd_ori @ np.array([rand_xy[0], rand_xy[1], -0.1]).reshape(-1,1)
            self.Rd = Rd_ori @ Rz @ Ry @ Rx

            self.xd = self.xd.reshape((-1,))
        else:
            rand_xy = 2*(np.random.rand(2,) - 0.5) * 0.04
            rand_rpy = 2*(np.random.rand(3,) - 0.5) * 12 /180 * np.pi

            Rx = np.array([[1, 0, 0], [0, np.cos(rand_rpy[0]), -np.sin(rand_rpy[0])], [0, np.sin(rand_rpy[0]), np.cos(rand_rpy[0])]])
            Ry = np.array([[np.cos(rand_rpy[1]), 0, np.sin(rand_rpy[1])], [0, 1, 0], [-np.sin(rand_rpy[1]), 0, np.cos(rand_rpy[1])]])
            Rz = np.array([[np.cos(rand_rpy[2]), -np.sin(rand_rpy[2]), 0], [np.sin(rand_rpy[2]), np.cos(rand_rpy[2]), 0], [0, 0, 1]])

            self.xd = xd_ori.reshape((-1,1)) + Rd_ori @ np.array([rand_xy[0], rand_xy[1], -0.1]).reshape(-1,1)
            self.Rd = Rd_ori @ Rz @ Ry @ Rx

            self.xd = self.xd.reshape((-1,))

        count = 0
        while True:
            action = np.array([1,1,1,1,1,1]) * 0.8
            obs, reward, done, info = self.step(action)

            eg = self.get_eg()
            ev = self.get_eV()

            count += 1

            if np.linalg.norm(eg) < 0.01 or count > 4000:
                if count > 3500:
                    print('Emergency Break')
                break

        q0 = self.robot_state.get_joint_pose()
        set_state(self.sim, q0, np.zeros(self.sim.model.nv))

        obs = self._get_obs()
        self.xd = xd_ori
        self.Rd = Rd_ori

        self.iter = 0
        self.done_count = 0

        return obs

    def initial_sample(self):
        Rd = self.Rd
        if self.robot_name == 'ur5e':
            if self.hole_ori == 'default':
                q0 = np.array([0, -2*np.pi/3, np.pi/4, -1 * np.pi/4, -np.pi/2, 0.1])
            elif self.hole_ori == 'case1':
                q0 = np.array([-0.5, -2*np.pi/3, np.pi/4, -1 * np.pi/4, -np.pi/2, 0.1])
            elif self.hole_ori == 'case2':
                q0 = np.array([0, -3*np.pi/4, 2*np.pi/4, -2 * np.pi/3, -np.pi/2, 0.1])
            elif self.hole_ori == 'case3':
                q0 = np.array([0, -2*np.pi/2.5, 2*np.pi/3, -2 * np.pi/3, -np.pi/2, 0.1])
            else:
                q0 = np.array([0, -2*np.pi/3, np.pi/4, -1 * np.pi/4, -np.pi/2, 0.1])

            while True:
                bias = np.array([0, 0.5, -0.5, 0.5, 0, 0])
                scale = np.array([0.2, 1.0, 0.6, 0.2, 0.1, 0.5])
                q0_noise = (np.random.rand(6) + bias) * scale
                q0 += q0_noise

                x,R = self.robot_state.forward_kinematics(q0)

                eR = self.vee_map(Rd.T @ R - R.T @ Rd)

                if np.linalg.norm(eR) < 2:
                    break

        elif self.robot_name == 'fanuc':
            if self.hole_ori == 'default':
                q0_ = np.array([0.0, 0.4, 0.0, 0.0, -np.pi/2 + 0.4, 0.0]) ## Bakje
                # q0_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
            elif self.hole_ori == 'case1':
                q0_ = np.array([-0.2, 0.5, 0.2, 0., -np.pi/2 + 0.5, 0.]) ## Bakje

            elif self.hole_ori == 'case2':
                q0_ = np.array([0., 0.4, 0.2, 0., -np.pi/2 + 0.4, 0.]) ## Bakje

            elif self.hole_ori == 'case3':
                q0_ = np.array([0., 0.4, 0.2, 0., 0.4, 0.]) ## Bakje

            while True:
                bias = np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
                scale = np.array([0.6, 0.8, 0.8, 0.8, 1, 1])
                q0_noise = (np.random.rand(6) + bias) * scale
                q0 = q0_ + q0_noise

                x,R = self.robot_state.forward_kinematics(q0)

                ep = R.T @ (x - self.xd).reshape((-1,1))
                eR = self.vee_map(self.Rd.T @ R - R.T @ self.Rd)

                ep_norm = np.linalg.norm(ep)
                if np.linalg.norm(eR) < 2 and ep_norm > 0.5 and ep_norm < 1:
                    break


        elif self.robot_name == 'panda':
            q0 = np.array([0, -np.pi/3, np.pi/2, -3 * np.pi/4, 0, np.pi/2, 0, 0, 0]) 

            while True:
                q0_noise = np.random.rand(9) * np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0, 0])
                q0 += q0_noise

                x,R = self.robot_state.forward_kinematics(q0)

                eR = self.vee_map(Rd.T @ R - R.T @ Rd)

                if np.linalg.norm(eR) < 0.5:
                    break

        set_state(self.sim, q0, np.zeros(self.sim.model.nv))
        ep = R.T @ (x - self.xd).reshape((-1,1))

        eg = np.vstack((ep, eR))

        return eg

    def test(self):
        return_arr = []        
        for i in range(self.max_iter):
            # print(i)
            action = self.get_expert_action()
            action = np.array([0,0,0,0,0,0])
            obs, reward, done, info = self.step(action)

            # print(obs)

            return_arr.append(reward)

            if self.show_viewer:
                self.viewer.render()

            if done:
                if info['success']:
                    print('Success!')
                else:
                    print('Failed...')
                break

            self.time_step = i

        print('total_Return:',sum(return_arr))

    def get_expert_action(self):
        x,R = self.robot_state.get_pose_mine()

        rot = np.trace(np.eye(3) - self.Rd.T @ R)
        trans = 0.5 * (x - self.xd).T @ (x - self.xd)
        dis = np.sqrt(rot + trans)

        # z_part = abs(x[2] - self.xd[2])
        eg = self.get_eg()
        z_part = abs(eg[2,0])
        # trans_part1 = np.sqrt(0.5 * (x[0:2] - self.xd[0:2]).T @ (x[0:2] - self.xd[0:2]))
        trans_part1 = np.sqrt(eg[0:2,:].T @ eg[0:2,:])

        if self.ECGIC:
            if dis > 1:
                a0 = 0.6; a1 = 0.6; a2 = 0.4; a3 = 0.6; a4 = 0.6; a5 = 0.6
            elif z_part < 0.3 and z_part > 0.10:
                a0 = 0.9; a1 = 0.9; a2 = -0.7; a3 = 0.9; a4 = 0.9; a5 = 0.9
            elif z_part < 0.10 and rot < 0.001 and trans_part1 < 0.001:
                a0 = 0.9; a1 = 0.9; a2 = 0.9; a3 = 0.9; a4 = 0.9; a5 = 0.9
            elif z_part < 0.10:
                a0 = 0.9; a1 = 0.9; a2 = -0.9; a3 = 0.9; a4 = 0.9; a5 = 0.9
            else:
                a0 = 0.8; a1 = 0.8; a2 = -0.5; a3 = 0.8; a4 = 0.8; a5 = 0.8
        else:
            # if dis > 1:
            #     a0 = 0.6; a1 = 0.6; a2 = 0.9; a3 = 0.6; a4 = 0.6; a5 = 0.6
            # elif z_part < 0.3 and z_part > 0.10:
            #     a0 = 0.9; a1 = 0.9; a2 = -0.7; a3 = 0.9; a4 = 0.9; a5 = 0.9
            # elif z_part < 0.10 and rot < 0.0005 and trans_part1 < 0.0005:
            #     a0 = 1.5; a1 = 1.5; a2 = 1.5; a3 = 0.9; a4 = 0.9; a5 = 0.9
            # elif z_part < 0.10:
            #     a0 = 1.5; a1 = 1.5; a2 = -0.9; a3 = 0.9; a4 = 0.9; a5 = 0.9
            # else:
            #     a0 = 0.8; a1 = 0.8; a2 = -0.5; a3 = 0.8; a4 = 0.8; a5 = 0.8
            
            if dis > 1:
                a0 = -0.05; a1 = -0.05; a2 = 4/15; a3 = 0.6; a4 = 0.6; a5 = 0.6
            elif z_part < 0.3 and z_part > 0.10:
                a0 = 0.25; a1 = 0.25; a2 = -0.8; a3 = 0.9; a4 = 0.9; a5 = 0.9
            elif z_part < 0.10 and rot < 0.0005 and trans_part1 < 0.0002:
                a0 = 0.5; a1 = 0.5; a2 = 2/3; a3 = 0.9; a4 = 0.9; a5 = 0.9
            elif z_part < 0.10:
                a0 = 0.5; a1 = 0.5; a2 = -0.95; a3 = 0.9; a4 = 0.9; a5 = 0.9
            else:
                a0 = 0.1; a1 = 0.1; a2 = -2/3; a3 = 0.8; a4 = 0.8; a5 = 0.8

        if self.act_type == 'default':
            action = np.array([a0,a1,a2,a3,a4,a5])

            action += np.random.randn(6,) * np.array([.05, .05, .05, .05, .05, .05])
            
            action = np.clip(action, -0.99, 0.99)
        elif self.act_type == 'minimal':
            action = np.array([a0,a2])
            action += np.random.randn(2,) * np.array([.05, .05])
            action = np.clip(action, -0.99, 0.99)

        elif self.act_type == 'minimal2':
            action = np.array([a0,a1,a2])
            action += np.random.randn(3,) * np.array([.05, .05, .05])
            action = np.clip(action, -0.99, 0.99)

        elif self.act_type == 'minimal2-1':
            a0_ = (a0 + a1)/2
            a1_ = (a0 - a1)/2
            action = np.array([a0_,a1_,a2])
            action += np.random.randn(3,) * np.array([.05, .05, .05])
            action = np.clip(action, -0.99, 0.99)

        elif self.act_type == 'minimal3':
            action = np.array([a0,a1,a2,a3])
            action += np.random.randn(4,) * np.array([.05, .05, .05, .05])
            action = np.clip(action, -0.99, 0.99)
        
        return action

    def step(self, action):
        self.robot_state.update()

        if self.TCGIC:
            tau_cmd = self.total_compensation_impedance_control(action)
        else:
            tau_cmd = self.impedance_control(action) # Here the actions are 'impedance gains'

        # if self.testing:
            # print(action)

        self.robot_state.set_control_torque(tau_cmd)

        self.robot_state.update_dynamic()

        if self.show_viewer:
            self.viewer.render()

        obs = self._get_obs()

        x,R = self.robot_state.get_pose_mine()

        dis = np.sqrt(np.trace(np.eye(3) - self.Rd.T @ R) + 0.5 * (x - self.xd).T @ (x - self.xd))

        dis_trans = np.sqrt((x - self.xd).T @ (x - self.xd))

        stuck = self.detect_stuck(x,R)
        # stuck = False
        # print(stuck)

        if not self.testing:
            if self.done_count >= 40 and not stuck:
                done = True
                success = True
            elif stuck:
                done = True
                success = False
            else:
                done = False
                success = False
        elif self.testing:
            if dis_trans < 0.026:
                done = True
                success = True
                # print('Success')
            else:
                done = False
                success = False

        if self.iter == self.max_iter -1:
            done = True

        #TODO reward function
        reward = self.get_reward(done,x,R)
        #TODO generate done functionality
        info = dict()
        info['success'] = success

        self.iter +=1 

        # print(obs, obs.shape)

        return obs, reward, done, info
    
    def _get_obs(self):
        eg = self.get_eg()
        eV = self.get_eV()
        Fe = self.Fe

        if self.obs_type == 'pos_vel':
            raw_obs = np.vstack((eg,eV)).reshape((-1,))
        elif self.obs_type == 'pos':
            raw_obs = eg.reshape((-1,)) 
        elif self.obs_type == 'pos_vel_force':
            raw_obs = np.vstack((eg,eV,Fe)).reshape((-1,))
        elif self.obs_type == 'pos_force':
            raw_obs = np.vstack((eg, Fe)).reshape((-1,))
        elif self.obs_type == 'feature':
            raw_obs = self.get_custom_obs()
        elif self.obs_type == 'pos_feature':
            eg_reshaped = eg.reshape((-1,))
            feature = self.get_custom_obs()
            feature_select = feature[1:3]
            raw_obs = np.hstack((eg_reshaped, feature_select))
        elif self.obs_type == 'pos_feature2':
            eg_reshaped = eg.reshape((-1,))
            feature = self.get_custom_obs() # dis, z_part, trans_part1, rot, dq_norm
            feature_select = np.array([feature[2],feature[4]])
            raw_obs = np.hstack((eg_reshaped, feature_select))
        elif self.obs_type == 'pos_feature3':
            eg_reshaped = eg.reshape((-1,))
            feature = self.get_custom_obs() #dis, z_part, trans_part1, rot, dq_norm
            feature_select = np.array([feature[1],feature[4]])
            raw_obs = np.hstack((eg_reshaped, feature_select))

        if self.window_size == 1:
            obs = raw_obs
        else:
            self.memorize(raw_obs)
            obs = np.asarray(self.obs_memory).reshape((-1,))
            # flat obs

        return obs

    
    def memorize(self,obs):
        _temp = copy.deepcopy(self.obs_memory)
        for i in range(self.window_size):
            if i < self.window_size - 1:
                self.obs_memory[i+1] = _temp[i]

        self.obs_memory[0] = obs
    
    def detect_stuck(self,x,R):
        if np.linalg.norm(x - self.prev_x) < 1e-05:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        if self.stuck_count >= 1500:
            stuck = True
        else:
            stuck = False

        self.prev_x = x
        return stuck

    def get_reward(self,done,x,R):#TODO()

        scale = 0.1
        scale2 = 1.0
        dis = np.sqrt(np.trace(np.eye(3) - self.Rd.T @ R) + 0.5 * (x - self.xd).T @ (x - self.xd))
        dis = np.clip(dis,0,1)
        reward = -scale * dis
        
        if dis < 0.2 and abs(x[2] - self.xd[2]) < 0.04:
            reward = scale2 * (0.04 - abs(x[2] - self.xd[2]))


        if dis < 0.1 and abs(x[2] - self.xd[2]) < 0.026:
            self.done_count += 1
            # reward = (self.max_iter - self.iter) * self.dt * 0.2 + 2
            reward = 3

            # print(reward, self.done_count)

        if self.reward_version == 'force_penalty':
            

            fe = self.robot_state.get_ee_force()
            fe_norm = np.linalg.norm(fe)
            fe_norm = abs(fe[2])
            trans_part1 = np.sqrt((x[0:2] - self.xd[0:2]).T @ (x[0:2] - self.xd[0:2]))

            if trans_part1 > 0.0002:
                reward -= 0.1 * fe_norm / 20

            # print(fe_norm, reward)         
                

        
        return reward 

    def get_eg(self):
        x, R = self.robot_state.get_pose_mine()
        ep = R.T @ (x - self.xd).reshape((-1,1))
        eR = self.vee_map(self.Rd.T @ R - R.T @ self.Rd)

        eg = np.vstack((ep,eR))

        return eg

    def get_eV(self):
        #just for symmetry of the code
        return self.robot_state.get_body_ee_velocity()

    def total_compensation_impedance_control(self, action):
        Jb = self.robot_state.get_body_jacobian()
        M,C,G = self.robot_state.get_dynamic_matrices()
        Kp, KR = self.convert_gains(action)

        #1 Calculate positional force
        x, R = self.robot_state.get_pose_mine()
        xd, Rd = self.xd, self.Rd

        fp = R.T @ Rd @ Kp @ Rd.T @ (x - xd).reshape((-1,1))
        # fp = Kp @ R.T @ (x - xd).reshape((-1,1))
        fR = self.vee_map(KR @ Rd.T @ R - R.T @ Rd @ KR)

        fg = np.vstack((fp,fR))

        eV = self.get_eV()
        Kd = np.sqrt(np.block([[Kp, np.zeros((3,3))],[np.zeros((3,3)), KR]])) * 8

        # Kd = np.eye(6) * 50

        Fe = self.robot_state.get_ee_force().reshape((-1,1))

        M_tilde_inv = Jb @ np.linalg.pinv(M) @ Jb.T
        M_tilde = np.linalg.pinv(M_tilde_inv)

        H = np.eye(6) * np.array([2., 2., 2., 2., 2., 2.])

        dq = self.robot_state.get_joint_velocity().reshape((-1,1))

        tau_tilde = M_tilde @ (np.linalg.inv(H) @ (- Kd @ eV - fg))
        tau_cmd = Jb.T @ tau_tilde + C @ dq + G

        # print(tau_cmd)

        return tau_cmd.reshape((-1,))

    def impedance_control(self, action):
        Jb = self.robot_state.get_body_jacobian()

        #### For future use 
        ### 2023/04/27: M and C is not updated right now
        M,C,G = self.robot_state.get_dynamic_matrices()
        ####

        #0 Get impedance gains
        Kp, KR = self.convert_gains(action)

        #1 Calculate positional force
        x, R = self.robot_state.get_pose_mine()
        xd, Rd = self.xd, self.Rd

        fp = R.T @ Rd @ Kp @ Rd.T @ (x - xd).reshape((-1,1))
        # fp = Kp @ R.T @ (x - xd).reshape((-1,1))
        fR = self.vee_map(KR @ Rd.T @ R - R.T @ Rd @ KR)

        fg = np.vstack((fp,fR))

        #2. get error vel vector        
        eV = self.get_eV()
        Kd = np.sqrt(np.block([[Kp, np.zeros((3,3))],[np.zeros((3,3)), KR]])) * 8

        Fe = self.robot_state.get_ee_force()
        # Fe = self.robot_state.get_ee_force_mine()
        self.Fe = Fe.reshape((-1,1))

        if self.use_external_force:
            tau_tilde = -fg -Kd @ eV + Fe.reshape((-1,1))
        else:
            tau_tilde = -fg -Kd @ eV

        det_Jb = np.linalg.det(Jb)

        if abs(det_Jb) < 0.01:
            nonsingular = False
            # print('singular!:', self.iter)
        else:
            nonsingular = True

        if self.ECGIC and nonsingular:
            Jb_dot = self.robot_state.get_body_jacobian_dot()
            Jb_inv = np.linalg.pinv(Jb)
            M_tilde = Jb_inv.T @ M @ Jb_inv
            C_tilde = Jb_inv.T @ (C - M @ Jb_inv @ Jb_dot) @ Jb_inv

            Bk_11 = R.T @ Rd @ Kp @ Rd.T @ R
            Bk_12 = self.hat_map(fp)
            Bk_21 = np.zeros((3,3))
            mat = R.T @ Rd @ KR
            Bk_22 = np.trace(mat)*np.eye(3) - mat

            Bk = np.block([
                [Bk_11, Bk_12],
                [Bk_21, Bk_22],
            ])

            term = self.lambda_g * (M_tilde @ Bk @ eV + C_tilde @ fg + Kd @ fg)
            # print(tau_tilde, term)
            tau_tilde = tau_tilde - (term)

        tau_cmd = Jb.T @ tau_tilde + G

        return tau_cmd.reshape((-1,))
    
    def convert_gains(self,action):
        if self.act_type == 'default':
            axy = action[0:2]
            az = action[2]
            ao = action[3:6]

        elif self.act_type == 'minimal':
            axy = np.array([action[0],action[0]])
            az = action[1]
            ao = np.array([1.0,1.0,1.0])
        elif self.act_type == 'minimal2':
            axy = np.array([action[0], action[1]])
            az = action[2]
            ao = np.array([1.0,1.0,1.0])
        elif self.act_type == 'minimal2-1':
            a0 = (action[0] + action[1])/2
            a1 = (action[0] - action[1])/2
            axy = np.array([a0[0], a1[1]])
            axy = np.clip(axy,-1,1)
            az = action[2]
            ao = np.array([1.0,1.0,1.0])
        elif self.act_type == 'minimal3':
            axy = np.array([action[0], action[1]])
            az = action[2]
            ao = np.array([action[3], action[3], action[3]])
        
        # kt_xy = pow(10,0.75*axy + 2) # scaling to (1.25, 2.75
        # kt_z = pow(10,1.0*az + 1.5) # 0.5 to 2.5
        # kt = np.hstack((kt_xy,kt_z))

        #update
        kt_xy = pow(10,1.0*axy + 2.5) # scaling to (1.5, 3.5)
        kt_z = pow(10,1.5*az + 2.0) # 0.5 to 3.5
        kt = np.hstack((kt_xy,kt_z))
        ko = pow(10,0.6*ao + 2.0) #scaling to 1.4, 2.6

        Kp = np.diag(kt); KR = np.diag(ko)

        return Kp, KR

    def get_custom_obs(self):
        x,R = self.robot_state.get_pose_mine()
        rot = np.trace(np.eye(3) - self.Rd.T @ R)
        trans = 0.5 * (x - self.xd).T @ (x - self.xd)
        dis = np.sqrt(rot + trans)

        eg = self.get_eg()
        z_part = abs(eg[2,0])
        trans_part1 = np.linalg.norm(eg[0:2,0])

        dq_norm = np.linalg.norm(self.robot_state.get_joint_velocity())

        obs = np.array([dis, z_part, trans_part1, rot, dq_norm])
        obs = np.tanh(obs)
        return obs
    
    def get_custom_obs_data_collection(self):
        x,R = self.robot_state.get_pose_mine()
        eg = self.get_eg()
        eV = self.get_eV()

        return eg, eV, x, R
    
    def vee_map(self,R):
        v3 = -R[0,1]
        v1 = -R[1,2]
        v2 = R[0,2]
        return np.array([v1,v2,v3]).reshape((-1,1))
    
    def hat_map(self, w):
        w = w.reshape((-1,))
        w_hat = np.array([[0, -w[2], w[1]],
                          [w[2], 0, -w[0]],
                          [-w[1], w[0], 0]])

        return w_hat


    def csv_logger(self, q, dq, tau_cmd):
        if robot_name == 'ur5e':
            header = ['q1','q2','q3','q4','q5','q6','dq1','dq2','dq3','dq4','dq5','dq6','t1','t2','t3','t4','t5','t6']

        row = q.tolist() + dq.tolist() + tau_cmd.tolist()

        if os.path.isfile('./logs/'+self.csv_name+'.csv'):
            with open('./logs/'+self.csv_name+'.csv','a') as fd:
                writer = csv.writer(fd)
                writer.writerow(row)
        else:
            with open('./logs/'+self.csv_name+'.csv','a') as fd:
                writer = csv.writer(fd)
                writer.writerow(header)

if __name__ == "__main__":
    robot_name = 'fanuc' # Panda currently unavailable - we don't have dynamic model of this right now.
    env_type = 'square_PIH'
    show_viewer = True
    RE = RobotEnvSeparated(robot_name, env_type, show_viewer = True, obs_type = 'pos', window_size = 1, hole_ori = 'default', ECGIC = False, use_ext_force = False, testing = True, act_type = 'minimal', reward_version = 'force_penalty')
    RE.test()

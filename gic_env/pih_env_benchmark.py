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



class RobotEnvBenchmark(Env):
    def __init__(self, robot_name = 'fanuc', env_type = 'square_PIH', max_time = 15, show_viewer = False, 
                 obs_type = 'pos', hole_ori = 'default', testing = False, reward_version = None, window_size = 1, 
                 use_ext_force = False, act_type = 'default', mixed_obs = False):
        
        self.robot_name = robot_name
        self.env_type = env_type
        self.hole_ori = hole_ori
        self.testing = testing

        self.reward_version = reward_version
        self.window_size = window_size
        self.use_external_force = use_ext_force
        self.act_type = act_type

        if mixed_obs: # To collect the data for CIC + GCEV
            self.obs_is_GCEV = True
        else:
            self.obs_is_GCEV = False

        print('=================================')
        print('USING CARTESIAN IMPEDANCE CONTROL')
        print('=================================')


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

        self.show_viewer = show_viewer
        self.load_xml()
        self.obs_type = obs_type

        self.robot_state = RobotState(self.sim, "end_effector", self.robot_name)

        self.dt = 0.002
        self.max_iter = int(max_time/self.dt)

        self.time_step = 0

        self.kt = 50
        self.ko = 10

        self.iter = 0     

        if self.obs_type == 'pos_vel':
            self.num_obs = self.robot_state.N * 2
        elif self.obs_type == 'pos':
            self.num_obs = self.robot_state.N

        if self.robot_name =='ur5e':
            self.num_act = 6
        elif self.robot_name == 'fanuc':
            self.num_act = 6

        if self.act_type == 'minimal':
            self.num_act = 2

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs*self.window_size,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_act,))

        self.prev_x = np.zeros((3,))
        self.stuck_count = 0
        self.done_count = 0

        if self.window_size is not 1:
            self.obs_memory = [np.zeros(self.num_obs)] * self.window_size

        self.Fe = np.zeros((6,1))
        self.reset()

    def load_xml(self):
        dir = "/home/joohwan/deeprl/research/GIC_Learning_public/"
        if self.robot_name == 'ur5e':
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
            raise NotImplementedError

        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        if self.show_viewer:
            self.viewer = mujoco_py.MjViewer(self.sim)
        else:
            self.viewer = None

    def reset(self):
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

            eg = self.get_eX()
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
                q0_ = np.array([0.0, 0.4, 0.0, 0.0, -np.pi/2 + 0.4, 0.0]) 
            elif self.hole_ori == 'case1':
                q0_ = np.array([-0.2, 0.5, 0.2, 0., -np.pi/2 + 0.5, 0.]) 

            elif self.hole_ori == 'case2':
                q0_ = np.array([0., 0.4, 0.2, 0., -np.pi/2 + 0.4, 0.]) 

            elif self.hole_ori == 'case3':
                q0_ = np.array([0., 0.4, 0.2, 0., 0.4, 0.])

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
            action = self.get_expert_action()
            obs, reward, done, info = self.step(action)
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

        eg = self.get_eX()
        z_part = abs(eg[2,0])
        trans_part1 = np.sqrt(eg[0:2,:].T @ eg[0:2,:])

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
        
        return action

    def step(self, action):
        self.robot_state.update()

        tau_cmd = self.impedance_control(action) # Here the actions are 'impedance gains'

        self.robot_state.set_control_torque(tau_cmd)

        self.robot_state.update_dynamic()

        if self.show_viewer:
            self.viewer.render()

        obs = self._get_obs()

        x,R = self.robot_state.get_pose_mine()

        dis = np.sqrt(np.trace(np.eye(3) - self.Rd.T @ R) + 0.5 * (x - self.xd).T @ (x - self.xd))

        dis_trans = np.sqrt((x - self.xd).T @ (x - self.xd))

        stuck = self.detect_stuck(x,R)

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

        reward = self.get_reward(done,x,R)
        info = dict()
        info['success'] = success

        self.iter +=1 

        return obs, reward, done, info
    
    def _get_obs(self):
        eg = self.get_eX()
        eV = self.get_eV()
        Fe = self.Fe

        if self.obs_type == 'pos_vel':
            raw_obs = np.vstack((eg,eV)).reshape((-1,))
        elif self.obs_type == 'pos':
            raw_obs = eg.reshape((-1,)) 

        if self.window_size == 1:
            obs = raw_obs
        else:
            self.memorize(raw_obs)
            obs = np.asarray(self.obs_memory).reshape((-1,))

        if self.obs_is_GCEV: # Should be always false other than obs is GCEV
            # print('collecting GCEV')
            x, R = self.robot_state.get_pose_mine()
            ep = R.T @ (x - self.xd).reshape((-1,1))
            eR = self.vee_map(self.Rd.T @ R - R.T @ self.Rd)

            eg = np.vstack((ep,eR))
            obs = eg.reshape((-1,))

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
            reward = 3

        if self.reward_version == 'force_penalty':
            

            fe = self.robot_state.get_ee_force()
            fe_norm = np.linalg.norm(fe)
            fe_norm = abs(fe[2])
            trans_part1 = np.sqrt((x[0:2] - self.xd[0:2]).T @ (x[0:2] - self.xd[0:2]))

            if trans_part1 > 0.0002:
                reward -= 0.1 * fe_norm / 20 

        return reward 

    def get_eX(self):
        x, R = self.robot_state.get_pose_mine()
        ep = (x - self.xd).reshape((-1,1))

        Rd1 = self.Rd[:,0]; Rd2 = self.Rd[:,1]; Rd3 = self.Rd[:,2]
        R1 = R[:,0]; R2 = R[:,1]; R3 = R[:,2]

        eR = -((np.cross(R1,Rd1) + np.cross(R2,Rd2) + np.cross(R3,Rd3))).reshape((-1,1))

        eX = np.vstack((ep,eR))

        return eX

    def get_eV(self):
        #just for symmetry of the code
        return self.robot_state.get_spatial_ee_velocity()

    def impedance_control(self, action):
        Je = self.robot_state.get_jacobian_mine() ## self.robot_state.get_jacobian_mine() returns exactly same value

        #### For future use
        M,C,G = self.robot_state.get_dynamic_matrices()
        ####

        #1. get error pos vector
        eX = self.get_eX()

        #2. get error vel vector        
        eV = self.get_eV()

        Kp,KR = self.convert_gains(action)

        Kg = np.block([[Kp, np.zeros((3,3))],[np.zeros((3,3)), KR]])
        Kd = np.sqrt(Kg) * 8

        spatial_quat = np.array([0.0, 0.0, 0.0, 1.0])
        Fe = self.robot_state.get_ee_force(spatial_quat)
        self.Fe = Fe.reshape((-1,1))

        if self.use_external_force:
            tau_tilde = -Kg @ eX -Kd @ eV - Fe.reshape((-1,1))
        else:
            tau_tilde = -Kg @ eX -Kd @ eV

        tau_cmd = Je.T @ tau_tilde + G    

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

        #update
        kt_xy = pow(10,1.0*axy + 2.5) # scaling to (1.5, 3.5)
        kt_z = pow(10,1.5*az + 2.0) # 0.5 to 3.5
        kt = np.hstack((kt_xy,kt_z))
        ko = pow(10,0.6*ao + 2.0) #scaling to 1.4, 2.6

        Kp = np.diag(kt); KR = np.diag(ko)

        return Kp, KR
    
    def get_custom_obs_data_collection(self):
        x,R = self.robot_state.get_pose_mine()
        eg = self.get_eX()
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

if __name__ == "__main__":
    robot_name = 'fanuc' # Panda currently unavailable - we don't have dynamic model of this right now.
    env_type = 'square_PIH'
    show_viewer = True
    RE = RobotEnvBenchmark(robot_name, env_type, show_viewer = True, obs_type = 'pos', window_size = 1, hole_ori = 'default', use_ext_force = False, testing = True, act_type = 'minimal', reward_version = 'force_penalty')
    RE.test()

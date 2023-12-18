from gym import Env
from gym import spaces

import mujoco_py
import numpy as np
import time, csv, os, copy

from gym import utils

# import matplotlib.pyplot as plt
from gic_env.utils.robot_state import RobotState
from gic_env.utils.mujoco import set_state, set_body_pose_rotm
from gic_env.utils.base import Primitive, PrimitiveStatus
from gic_env.utils.cases_handler import get_cases



class WipingEnvBenchmark(Env):
    def __init__(self, robot_name = 'fanuc', max_time = 15, show_viewer = False, obs_type = 'pos', testing = False, window_size = 1,
                 mixed_obs = False, plate_ori = 'arbitrary_x', use_ext_force = False, fix_camera = False):
        
        self.robot_name = robot_name        
        self.max_time = max_time
        self.plate_ori = plate_ori
        self.testing = testing
        self.window_size = window_size
        self.mixed_obs = mixed_obs
        self.use_external_force = use_ext_force
        
        self.file_name = "/gic_env/mujoco_models/pih/wiping_fanuc.xml"
        self.show_viewer = show_viewer
        self.load_xml()

        self.fix_camera = fix_camera

        self.Rd = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])

        if self.plate_ori == 'arbitrary_x':
            self.center_p = np.array([0.65, 0, 0.65])
            self.r = 0.6

            self.xd = self.center_p + self.r * np.array([0, np.sin(0.), -np.cos(0.)])
            Rt = self.rotmat_x(0.)

            self.Rd = Rt @ self.Rd
        else:
            raise NotImplementedError
        pass

        self.robot_state = RobotState(self.sim, "end_effector", self.robot_name)

        self.radius = 0.1

        self.dt = 0.002
        self.omega = (2*np.pi)/max_time
        self.max_iter = int(max_time/self.dt)
        self.time_step = 0

        self.num_obs = 6
        self.num_act = 6

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs*self.window_size,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_act,))

        self.Fe = np.zeros((6,1))

        self.iter = 0

        self.reset()

        if self.fix_camera:
            self.viewer_setup()

    def initial_sample(self):
        pass

    def load_xml(self):
        # dir = "/home/joohwan/deeprl/research/GIC_Learning_public/"
        dir = os.getcwd()
        model_path = dir + self.file_name

        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        if self.show_viewer:
            self.viewer = mujoco_py.MjViewer(self.sim)
        else:
            self.viewer = None
        pass

    def reset(self, angle_prefix = None):
        self.initial_sample()

        if angle_prefix is None:
            self.plate_angle = 0
        else:
            self.plate_angle = angle_prefix

        self.xd = self.center_p + self.r * np.array([0, np.sin(self.plate_angle), -np.cos(self.plate_angle)])
        Rt = self.rotmat_x(self.plate_angle)
        self.set_plate_pose(self.xd, Rt)  

        self.Rd = Rt @ self.Rd

        xd_ori = self.xd
        Rd_ori = self.Rd

        self.contact = False

        if not self.testing:
            rand_xy = 2*(np.random.rand(2,) - 0.5) * 0.03
            rand_rpy = 2*(np.random.rand(3,) - 0.5) * 15 /180 * np.pi

            Rx = np.array([[1, 0, 0], [0, np.cos(rand_rpy[0]), -np.sin(rand_rpy[0])], [0, np.sin(rand_rpy[0]), np.cos(rand_rpy[0])]])
            Ry = np.array([[np.cos(rand_rpy[1]), 0, np.sin(rand_rpy[1])], [0, 1, 0], [-np.sin(rand_rpy[1]), 0, np.cos(rand_rpy[1])]])
            Rz = np.array([[np.cos(rand_rpy[2]), -np.sin(rand_rpy[2]), 0], [np.sin(rand_rpy[2]), np.cos(rand_rpy[2]), 0], [0, 0, 1]])

            self.xd = xd_ori.reshape((-1,1)) + Rd_ori @ np.array([rand_xy[0] + self.radius, rand_xy[1], -0.1]).reshape(-1,1)
            self.Rd = Rd_ori @ Rz @ Ry @ Rx

            self.xd = self.xd.reshape((-1,))
        else:
            rand_xy = 2*(np.random.rand(2,) - 0.5) * 0.02
            rand_rpy = 2*(np.random.rand(3,) - 0.5) * 10 /180 * np.pi

            Rx = np.array([[1, 0, 0], [0, np.cos(rand_rpy[0]), -np.sin(rand_rpy[0])], [0, np.sin(rand_rpy[0]), np.cos(rand_rpy[0])]])
            Ry = np.array([[np.cos(rand_rpy[1]), 0, np.sin(rand_rpy[1])], [0, 1, 0], [-np.sin(rand_rpy[1]), 0, np.cos(rand_rpy[1])]])
            Rz = np.array([[np.cos(rand_rpy[2]), -np.sin(rand_rpy[2]), 0], [np.sin(rand_rpy[2]), np.cos(rand_rpy[2]), 0], [0, 0, 1]])

            self.xd = xd_ori.reshape((-1,1)) + Rd_ori @ np.array([rand_xy[0] + self.radius, rand_xy[1], -0.1]).reshape(-1,1)
            self.Rd = Rd_ori @ Rz @ Ry @ Rx

            self.xd = self.xd.reshape((-1,))

        count = 0

        self.in_initialization = True
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
        self.in_initialization = False
        
        q0 = self.robot_state.get_joint_pose()
        set_state(self.sim, q0, np.zeros(self.sim.model.nv))

        obs = self._get_obs()
        self.xd = xd_ori
        self.Rd = Rd_ori

        self.done_count = 0
        self.iter = 0

        return obs

    def _get_obs(self):
        eg = self.get_eX()
        eV = self.get_eV()
        Fe = self.Fe

        raw_obs = eg.reshape((-1,)) 

        if self.window_size == 1:
            obs = raw_obs
        else:
            self.memorize(raw_obs)
            obs = np.asarray(self.obs_memory).reshape((-1,))

        if self.mixed_obs:
            obs = self.get_eg()
            obs = obs.reshape((-1,))

        return obs
    
    def get_eX(self):
        x, R = self.robot_state.get_pose_mine()
        ep = (x - self.xd).reshape((-1,1))

        Rd1 = self.Rd[:,0]; Rd2 = self.Rd[:,1]; Rd3 = self.Rd[:,2]
        R1 = R[:,0]; R2 = R[:,1]; R3 = R[:,2]

        eR = -((np.cross(R1,Rd1) + np.cross(R2,Rd2) + np.cross(R3,Rd3))).reshape((-1,1))

        eX = np.vstack((ep,eR))

        return eX

    def step(self, action):
        self.robot_state.update()

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

        if self.iter == self.max_iter -1:
            done = True
        else:
            done = False

        reward = self.get_reward(done,x,R)
        info = dict()
        xd, Rd = self.get_trajectory(self.iter)
        info['xd'] = xd
        info['Rd'] = Rd
        info['x'] = x
        info['R'] = R
        info['Fe'] = self.Fe.reshape((-1,))

        self.iter +=1 

        return obs, reward, done, info
    
    def get_reward(self,done,x,R):
        reward = 0

        return reward 
    
    def get_trajectory(self, iter):
        t = iter * self.dt

        if self.in_initialization:
            x_traj, R_traj = self.xd, self.Rd
        else:
            x_traj = self.xd.reshape((-1,1)) + self.Rd @ np.array([self.radius * np.cos(t*self.omega),self.radius * np.sin(t*self.omega),0.00]).reshape((-1,1))

            R_traj = self.Rd

        return x_traj.reshape((-1,)), R_traj
    
    def impedance_control(self, action):
        Je = self.robot_state.get_jacobian_mine() ## self.robot_state.get_jacobian_mine() returns exactly same value
        # Jb = self.robot_state.get_body_jacobian()

        x, R = self.robot_state.get_pose_mine()

        #### For future use
        M,C,G = self.robot_state.get_dynamic_matrices()
        ####

        #1. get error pos vector
        # eX = self.get_eX()
        xd, Rd = self.get_trajectory(self.iter)
        ep = (x - xd).reshape((-1,1))
        Rd1 = Rd[:,0]; Rd2 = Rd[:,1]; Rd3 = Rd[:,2]
        R1 = R[:,0]; R2 = R[:,1]; R3 = R[:,2]

        eR = -((np.cross(R1,Rd1) + np.cross(R2,Rd2) + np.cross(R3,Rd3))).reshape((-1,1))

        eX = np.vstack((ep,eR))

        #2. get error vel vector        
        eV = self.get_eV()

        Kp,KR = self.convert_gains(action)

        # print(Kp)

        Kg = np.block([[Kp, np.zeros((3,3))],[np.zeros((3,3)), KR]])

        # Kg = Je.T @ Kg @ Je

        eig_val, eig_vec = np.linalg.eig(Kg)

        Kd = eig_vec @ np.eye(6) * np.sqrt(eig_val) * 8 @ eig_vec.T

        # print(Kp)

        spatial_quat = np.array([0.0, 0.0, 0.0, 1.0])
        Fe = self.robot_state.get_ee_force()
        self.Fe = Fe.reshape((-1,1))

        if self.use_external_force:
            tau_tilde = -Kg @ eX -Kd @ eV - Fe.reshape((-1,1))
        else:
            tau_tilde = -Kg @ eX -Kd @ eV

        tau_cmd = Je.T @ tau_tilde + G    

        return tau_cmd.reshape((-1,))
    
    def test(self):
        return_arr = []        
        for i in range(self.max_iter):
            action = self.get_expert_action()
            # action = np.array([0,0,0,0,0,0])
            obs, reward, done, info = self.step(action)


            return_arr.append(reward)

            if self.show_viewer:
                self.viewer.render()

            if done:
                print('finished')

            self.time_step = i

        print('total_Return:',sum(return_arr))

    def get_expert_action(self):
        x,R = self.robot_state.get_pose_mine()

        rot = np.trace(np.eye(3) - self.Rd.T @ R)
        trans = 0.5 * (x - self.xd).T @ (x - self.xd)
        dis = np.sqrt(rot + trans)

        eg = self.get_eg()
        z_part = abs(eg[2,0])
        trans_part1 = np.sqrt(eg[0:2,:].T @ eg[0:2,:])

        if self.contact:
            if z_part < 0.03 and rot < 0.0005 and eg[1,0] < 0 :
                a0 = 0.9; a1 = 0.9; a2 = 0.5493; a3 = 0.9; a4 = 0.9; a5 = 0.9
            elif z_part < 0.03 and rot < 0.0005 and eg[1,0] >= 0:
                a0 = 0.9; a1 = 0.9; a2 = 0.3486; a3 = 0.9; a4 = 0.9; a5 = 0.9
            else:
                a0 = 0.5; a1 = 0.5; a2 = 0; a3 = 0.8; a4 = 0.8; a5 = 0.8
        else:
            a0 = 0.5; a1 = 0.5; a3 = 0.8; a4 = 0.8; a5 = 0.8
            a2 = -1 + z_part * 7.5

        if abs(self.Fe[2]) >= 0.5:
            self.contact = True
        
        # if z_part < 0.3 and z_part > 0.10:
        #     a0 = 0.5; a1 = 0.5; a2 = 0.5; a3 = 0.9; a4 = 0.9; a5 = 0.9
        # elif z_part < 0.1 and z_part > 0.03:
        #     a0 = 0.5; a1 = 0.5; a2 = -0.1; a3 = 0.9; a4 = 0.9; a5 = 0.9
        # elif z_part < 0.03 and rot < 0.0005 and eg[1,0] < 0 :
        #     a0 = 0.9; a1 = 0.9; a2 = 0.6; a3 = 0.9; a4 = 0.9; a5 = 0.9
        # elif z_part < 0.03 and rot < 0.0005 and eg[1,0] >= 0:
        #     a0 = 0.9; a1 = 0.9; a2 = 0.2; a3 = 0.9; a4 = 0.9; a5 = 0.9
        # else:
        #     a0 = 0.5; a1 = 0.5; a2 = -2/3; a3 = 0.8; a4 = 0.8; a5 = 0.8

        action = np.array([a0,a1,a2,a3,a4,a5])
        action += np.random.randn(6,) * np.array([.05, .05, .05, .05, .05, .05])
        action = np.clip(action, -0.99, 0.99)

        return action
    
    
    def get_eg(self):
        x, R = self.robot_state.get_pose_mine()
        ep = R.T @ (x - self.xd).reshape((-1,1))
        eR = self.vee_map(self.Rd.T @ R - R.T @ self.Rd)

        eg = np.vstack((ep,eR))

        return eg

    def get_eV(self):
        return self.robot_state.get_spatial_ee_velocity()


    def rotmat_x(self, th):
        R = np.array([[1,0,0],
                      [0,np.cos(th),-np.sin(th)],
                      [0,np.sin(th), np.cos(th)]])

        return R

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
    
    def set_plate_pose(self, pos, R):
        set_body_pose_rotm(self.model, 'plate', pos, R)

    def convert_gains(self,action):

        axy = action[0:2]
        az = action[2]
        ao = action[3:6]

        #update
        kt_xy = pow(10,1.0*axy + 2.5) # scaling to (1.5, 3.5)
        kt_z = pow(10,1.5*az + 2.0) # 0.5 to 3.5
        kt = np.hstack((kt_xy,kt_z))
        ko = pow(10,0.6*ao + 2.0) #scaling to 1.4, 2.6

        Kp = np.diag(kt); KR = np.diag(ko)

        return Kp, KR
    
    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.type = mujoco_py.generated.const.CAMERA_FREE
        self.viewer.cam.trackbodyid = 1
            
        # self.viewer.cam.distance = self.model.stat.extent * 0.4
        # self.viewer.cam.lookat[0] = 0.65
        # self.viewer.cam.lookat[1] = 0
        # self.viewer.cam.lookat[2] = 0.2
        # self.viewer.cam.elevation = -30
        # self.viewer.cam.azimuth = 90

        self.viewer.cam.distance = self.model.stat.extent * 0.4
        self.viewer.cam.lookat[0] = 0.65
        self.viewer.cam.lookat[1] = -0.2
        self.viewer.cam.lookat[2] = 0.2
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 180

if __name__ == "__main__":
    robot_name = 'fanuc'
    show_viewer = True
    angle = 0
    angle_rad = angle / 180 * np.pi
    WE = WipingEnvBenchmark(robot_name, show_viewer = True)
    WE.reset(angle_rad)
    WE.test()

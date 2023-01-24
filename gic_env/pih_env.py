from gym import Env
from gym import spaces

import mujoco_py
import numpy as np
import time, csv, os

from gym import utils

import matplotlib.pyplot as plt
from gic_env.utils.robot_state import RobotState
from gic_env.utils.mujoco import set_state
from gic_env.utils.base import Primitive, PrimitiveStatus


class RobotEnv(Env):
    def __init__(self, robot_name = 'ur5e', env_type = 'square_PIH', max_time = 10, show_viewer = False):
        self.robot_name = robot_name
        self.env_type = env_type

        self.xd = np.array([0.60, 0.012, 0.05])
        self.Rd = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])

        self.show_viewer = show_viewer
        self.load_xml()

        self.robot_state = RobotState(self.sim, "end_effector", self.robot_name)
        self.sim_primitive = Primitive(self.robot_state, controller=None)

        self.initialize_sim()

        self.dt = 0.002
        self.max_iter = int(max_time/self.dt)
        self.dummy = 'dummy_change'

        self.logging = False
        self.csv_name = 'square_PIH_4'

        self.time_step = 0

        self.kt = 50
        self.ko = 10

        self.iter = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_state.N * 2,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))

        utils.EzPickle.__init__(self)

    def load_xml(self):
        if self.robot_name == 'ur5e':
            # model_path = "/home/joohwan/deeprl/research/GIC_learning/mujoco_models/pih/sliding_test.xml"
            model_path = "/home/joohwan/deeprl/research/GIC_learning/mujoco_models/pih/square_pih_ur5e.xml"
        elif self.robot_name == 'panda':
            # model_path = "/home/joohwan/deeprl/research/GIC_learning/mujoco_models/pih/sliding.xml"
            # model_path = "/home/joohwan/deeprl/research/GIC_learning/mujoco_models/pih/square_pih.xml"
            model_path = "mujoco_models/pih/square_pih.xml"

        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        if self.show_viewer:
            self.viewer = mujoco_py.MjViewer(self.sim)
        else:
            self.viewer = None

    def initialize_sim(self):
        eg = self.initial_sample()

    def reset(self):
        eg = self.initial_sample()
        eV = np.zeros((self.robot_state.N,1))
        obs = np.vstack((eg,eV)).reshape((-1))
        self.iter = 0 
        return obs

    def initial_sample(self):
        Rd = self.Rd
        if self.robot_name == 'ur5e':
            q0 = np.array([0, -2*np.pi/3, np.pi/4, -1 * np.pi/4, -np.pi/2, 0.1])

            while True:
                bias = np.array([0, 0.5, -0.5, 0.5, 0, 0])
                scale = np.array([0.3, 1.0, 0.6, 0.2, 0.1, 0.5])
                q0_noise = (np.random.rand(6) + bias) * scale
                q0 += q0_noise

                x,R = self.robot_state.forward_kinematics(q0)

                eR = self.vee_map(Rd.T @ R - R.T @ Rd)

                if np.linalg.norm(eR) < 2:
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
        for i in range(self.max_iter):
            action = np.array([self.kt,self.kt,0.2*self.kt,self.ko,self.ko,self.ko])
            obs, reward, done, info = self.step(action)

            if self.show_viewer:
                self.viewer.render()

            self.time_step = i

    def step(self, action):
        self.robot_state.update()

        tau_cmd = self.impedance_control(action) # Here the actions are 'impedance gains' 

        self.robot_state.set_control_torque(tau_cmd)

        self.robot_state.update_dynamic()

        if self.show_viewer:
            self.viewer.render()

        eg = self.get_eg()
        eV = self.get_eV()

        obs = np.vstack((eg,eV)).reshape((-1,))

        x,R = self.robot_state.get_pose_mine()

        dis = np.sqrt(np.trace(np.eye(3) - self.Rd.T @ R) + 0.5 * (x - self.xd).T @ (x - self.xd))

        if dis < 0.2 and abs(x[2] - self.xd[2]) < 0.024:
            done = True
        else:
            done = False

        #TODO reward function
        reward = self.get_reward(done,x,R)
        #TODO generate done functionality
        info = dict()

        self.iter +=1 

        return obs, reward, done, info

    def get_reward(self,done,x,R):#TODO()
        scale = 0.1
        scale2 = 2
        dis = np.sqrt(np.trace(np.eye(3) - self.Rd.T @ R) + 0.5 * (x - self.xd).T @ (x - self.xd))
        dis = np.clip(dis,0,1)
        reward = -scale * dis
        if dis < 0.2 and abs(x[2] - self.xd[2]) < 0.04:
            reward = scale2 * (0.04 - abs(x[2] - self.xd[2]))
        if done:
            reward = 2

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

    def impedance_control(self, action):
        Jb = self.robot_state.get_body_jacobian()

        #### For future use
        M,C,G = self.robot_state.get_dynamic_matrices()
        ####

        #1. get error pos vector
        eg = self.get_eg()

        #2. get error vel vector        
        eV = self.get_eV()

        Kg = self.convert_gains(action)
        Kd = 5*np.sqrt(Kg)

        tau_tilde = -Kg @ eg -Kd @ eV

        tau_cmd = Jb.T @ tau_tilde + G    

        return tau_cmd.reshape((-1,))
    
    def convert_gains(self,action):
        # print(action)
        axy = action[0:2]
        az = action[2]
        ao = action[3:6]

        kt_xy = pow(10,0.85*axy + 1.85) # scaling to (1,2.7)
        kt_z = pow(10,0.5*az + 1.0) #scaling to 0.5, 1.5 >> 3, 30
        kt = np.hstack((kt_xy,kt_z))
        ko = pow(10,0.5*ao + 1.2) #scaling to 0.7, 1.7

        ko_tilde = np.array([(ko[1]+ko[2])/2, (ko[0]+ko[2])/2, (ko[0]+ko[1])/2])

        Kg = np.diag(np.hstack((kt,ko_tilde)))

        return Kg

    
    def vee_map(self,R):
        v3 = -R[0,1]
        v1 = -R[1,2]
        v2 = R[0,2]
        return np.array([v1,v2,v3]).reshape((-1,1))


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
    robot_name = 'ur5e' # Panda currently unavailable - we don't have dynamic model of this right now.
    env_type = 'square_PIH'
    show_viewer = True
    RE = RobotEnv(robot_name, env_type, show_viewer)
    RE.test()

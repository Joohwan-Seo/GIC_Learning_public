from gym import Env
from gym import spaces
import numpy as np
import random

from scipy.integrate import odeint
from numpy.linalg import inv, norm
from scipy.linalg import block_diag

class DoubleIntegrator2D(Env):
    def __init__(self, args = None, max_iter = 2000):
        self.args = args
        
        self.dt = 0.01
        self.max_iter = max_iter
        self.info = {}

        self.mass_fictious = 1
        self.total_E = 100

        self.gamma = 1.5

        if args is not None:
            self.obs_type = args['obs_type'] # pos_only and pos_vel
            self.task = args['task']
        else:
            self.obs_type = None
            self.task = None

        if self.obs_type == 'pos_only':
            self.num_obs = 2
        elif self.obs_type == 'pos_vel':
            self.num_obs = 4

        self.num_act = 2

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_act,))
        self.reset()

    def reset(self):
        self.step_cnt = 0

        self.target = self.get_target()
        self.old_Vr = np.array([0])

        x1, x2 = self.initial_sample()

        self.x = np.array([x1,0,x2,0]) #(2,)

        if self.obs_type == 'pos_only':
            obs = np.array([self.x[0], self.x[2]])
        elif self.obs_type == 'pos_vel':
            obs = np.array([self.x[0], self.x[1], self.x[2], self.x[3]])

        return obs #NOTE

    def get_target(self):
        t = self.step_cnt * self.dt
        if self.task == 'tracking':
            r = 0.75
            w = 1/np.pi
            t1 = r * np.sin(w * t)
            t2 = r * np.cos(w * t)

            self.dx_ref = w * r * np.cos(w*t)
            self.dy_ref = -w * r * np.sin(w*t)

            target= np.array([[t1],[t2]])
        else:
            target = np.array([[0],[0]])
        
        return target

    def initial_sample(self):
     
        while True:
            x1 = (np.random.rand() - 0.5) * 5
            x2 = (np.random.rand() - 0.5) * 5
            if min(abs(x1), abs(x2)) > 1:
                break

        x1 = 2
        x2 = 2

        return x1, x2

    def step(self, ac):

        # if np.isnan(ac).any():
            # print('q_in_step',self.q, self.step_cnt, ac)

        self.target = self.get_target()

        t = np.linspace(0,self.dt,2)
        self.u = self.pvfc_control(self.x, ac)
        sol = odeint(self.double_integrator_2d, self.x, t)
        
        #NOTE is this correct?
        self.x = sol[-1,:]        

        # obs = self.x #NOTE

        done = False

        if self.task == 'tracking':
            if self.obs_type == 'pos_only':
                obs = np.array([self.x[0] - self.target[0,0], self.x[2] - self.target[1,0]])
            elif self.obs_type == 'pos_vel':
                obs = np.array([self.x[0] - self.target[0,0], self.x[1],
                                self.x[2] - self.target[1,0], self.x[3]])
        else:
            if self.obs_type == 'pos_only':
                obs = np.array([self.x[0], self.x[2]])
            elif self.obs_type == 'pos_vel':
                obs = np.array([self.x[0], self.x[1], self.x[2], self.x[3]])

        if self.step_cnt >= self.max_iter:
            done = True

        self.step_cnt += 1

        reward = self.getReward()

        # obs = np.array([x,y])

        info = {'full_state':self.x}
        
        return obs, reward, done, info

    def pvfc_control(self, x, Vr):
        dVr = (Vr - self.old_Vr)/self.dt

        self.old_Vr = Vr

        Vr = Vr.reshape((-1,1))
        dVr = dVr.reshape((-1,1))

        M = np.array([[1, 0], [0, 1]])

        mass_aug = block_diag(M, self.mass_fictious)

        Vr = np.clip(Vr,-9,9)

        vel = np.array([x[1], x[3]]).reshape((-1,1))

        vel = np.clip(vel,-9,9)
        vel_aux = np.sqrt(2/self.mass_fictious * (self.total_E - 1/2 * vel.T @ M @ vel))
        vel_aug = np.concatenate((vel, vel_aux.reshape((-1,1))))

        Vr_aux = np.sqrt(2/self.mass_fictious * (self.total_E - 1/2 * Vr.T @ M @ Vr))
        Vr_aug = np.concatenate((Vr, Vr_aux.reshape((-1,1))))

        dVr_aug = np.concatenate((dVr, np.array([[0]])))

        p = mass_aug @ vel_aug
        P = mass_aug @ Vr_aug
        w = mass_aug @ dVr_aug
        
        tau_c = 1/(2*self.total_E) * (w @ P.T - P @ w.T) @ vel_aug
        tau_f = self.gamma * (P @ p.T - p @ P.T) @ vel_aug

        tau = tau_c + tau_f

        return tau[0:2,:]

    def getReward(self):
        x1, dx1, x2, dx2 = self.x        

        X = np.array([[x1],[x2]])
        
        metric = norm(X - self.target)
        
        #NOTE Reward ver1
        if metric < 0.1:
            reward = 10
        #NOTE Reward ver2
        # if metric < 0.1:
        #     reward = 1000 * (0.1 - metric)**2
        # elif norm(X) <= 0.3:
        #     reward = 0
        # elif x < 0:
        #     reward = 0
        else:
            reward = 0

        return reward

    def double_integrator_2d(self,x,t):
        x1, dx1, x2, dx2 = x

        # tau = self.u.reshape((-1,1))

        ddx1 = self.u[0,0]
        ddx2 = self.u[1,0]

        dxdt = np.array([dx1, ddx1, dx2, ddx2])

        return dxdt

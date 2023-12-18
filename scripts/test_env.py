from gic_env.wiping_env import WipingEnv
from gic_env.wiping_env_benchmark import WipingEnvBenchmark

import numpy as np
import matplotlib.pyplot as plt

def run_simulation():
    dt = 0.002
    max_time = 15
    max_iter = int(max_time / dt)

    angle = 30/180 * np.pi
    # env = WipingEnv(robot_name='fanuc',show_viewer = False, max_time = max_time)
    env = WipingEnvBenchmark(robot_name='fanuc',show_viewer = False, max_time = max_time)
    env.reset(angle_prefix = angle)

    x_list = []
    xd_list = []
    Fe_list = []
    _, Rd = env.get_trajectory(0)


    for k in range(max_iter):
        action = env.get_expert_action()
        # print(action)
        next_obs, rew, done, info = env.step(action)

        Fe_list.append(info['Fe'])
        xd_list.append(info['xd'])
        x_list.append(info['x'])

    Fe = np.asarray(Fe_list)
    xd = np.asarray(xd_list)
    x = np.asarray(x_list)

    t = np.linspace(0, max_time, max_iter)

    plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(t,x[:,0])
    plt.plot(t,xd[:,0])

    plt.subplot(3,1,2)
    plt.plot(t,x[:,1])
    plt.plot(t,xd[:,1])

    plt.subplot(3,1,3)
    plt.plot(t,x[:,2])
    plt.plot(t,xd[:,2])

    plt.figure(2)
    plt.plot(t,Fe[:,2])

    plt.show()

        




if __name__ == "__main__":
    robot_name = 'fanuc'
    show_viewer = True
    angle = 0
    angle_rad = angle / 180 * np.pi
    run_simulation()
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from data.Fanuc_success.Behavior_cloning.policy_model import BCPolicy, BCPolicyLarger
from gic_env.wiping_env import WipingEnv
from gic_env.wiping_env_benchmark import WipingEnvBenchmark
import argparse
import torch
import matplotlib.pyplot as plt

import pickle
from rlkit.core import logger

import numpy as np

import time


def test_policy(args):

    # num_obs = 6 if args.obs_type == 'pos' else 12
    if args.obs_type == 'pos':
        num_obs = 6
    elif args.obs_type == 'pos_vel':
        num_obs = 12

    policy_weights = torch.load(args.file)
    policy = BCPolicy(input_shape = num_obs * args.window_size)
    policy.load_state_dict(policy_weights)
    policy.float()

    # angles = [45]
    angles = [-30]

    dt = 0.002
    # cases = ['case3']

    num_testing = args.num_testing

    use_ext_force = args.use_ext_force

    if args.benchmark:
        print('==========================')
        print('Testing Benchmark Scenario')
        print('==========================')
    else:
        print('==========================')
        print('Testing Original Scenario')
        print('==========================')

    dataset = {}

    for angle in angles:
        # print('Case:',case)
        if args.benchmark:
            env = WipingEnvBenchmark(show_viewer = args.vis, obs_type = args.obs_type, plate_ori = 'arbitrary_x',
                                    window_size = args.window_size, mixed_obs = args.mixed_obs, testing = True, fix_camera = args.fix_camera)
        else:
            env = WipingEnv(show_viewer = args.vis, obs_type = args.obs_type, plate_ori = 'arbitrary_x',
                           window_size = args.window_size, use_ext_force=use_ext_force, mixed_obs = args.mixed_obs, testing = True, 
                           fix_camera = args.fix_camera)

        tic = time.time()

        x_list = []
        xd_list = []
        Fe_list = []
        eg_list = []
        _, Rd = env.get_trajectory(0)

        if args.gpu:
            set_gpu_mode(True)
            policy.cuda()
            cuda0 = torch.device('cuda:0')

        obs = env.reset(angle_prefix=angle / 180.0 * np.pi)

        for i in range(args.H):
            obs_tensor = torch.from_numpy(obs)

            obs_tensor = obs_tensor.to(torch.float32)                
            action = policy(obs_tensor.to(cuda0))
            action = action.cpu().detach().numpy()
            next_obs, rew, done, info =env.step(action)

            obs = next_obs

            R, x, xd = info['R'], info['x'], info['xd']

            eg = R.T @ (x - xd).reshape((-1,1))

            Fe_list.append(info['Fe'])
            xd_list.append(info['xd'])
            x_list.append(info['x'])
            eg_list.append(eg.reshape((-1,)))

            
            if done :
                break        

        toc = time.time() - tic

        print('Total time', toc)

        Fe = np.asarray(Fe_list)
        xd = np.asarray(xd_list)
        x = np.asarray(x_list)
        eg = np.asarray(eg_list)

        t = np.linspace(0, args.H * dt, args.H)

        dataset_small = dict(
        xd_list = xd_list,
        x_list = x_list,
        Fe_list = Fe_list,
        eg_list = eg_list,
        t = t
        )

        # plt.figure(1, figsize = (8,6))
        # plt.subplot(3,1,1)
        # plt.plot(t,x[:,0],'r')
        # plt.plot(t,xd[:,0], 'g--')

        # plt.subplot(3,1,2)
        # plt.plot(t,x[:,1],'r')
        # plt.plot(t,xd[:,1], 'g--')

        # plt.subplot(3,1,3)
        # plt.plot(t,x[:,2],'r')
        # plt.plot(t,xd[:,2], 'g--')

        # plt.figure(2, figsize = (8,6))

        # plt.plot(t,Fe[:,2],'r')

        # plt.show()

        dataset[str(angle)] = dataset_small

    if args.saving:

        if args.benchmark:
            file = open("./analyzing_data/dataset_CIC_CEV_wiping.pkl","wb")
            pickle.dump(dataset, file)
            file.close()

        else:
            file = open("./analyzing_data/dataset_GIC_GCEV_wiping.pkl","wb")
            pickle.dump(dataset, file)
            file.close()


    if args.mixed_obs:
        print('mixed obs used')
    else:
        print('General obs used')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the sna1pshot file')
    parser.add_argument('--H', type=int, default=7500,
                        help='Max length of rollout')
    parser.add_argument('--gpu', default = True, action='store_true')
    parser.add_argument('--case', type=str, default='default')
    parser.add_argument('--vis', type=str, default = 'False')
    parser.add_argument('--benchmark', type=str, default = 'False')
    parser.add_argument('--num_testing', type=int, default = 100)
    parser.add_argument('--window_size', type=int, default = 1)
    parser.add_argument('--obs_type', type = str, default = 'pos')
    parser.add_argument('--use_ext_force', type =bool, default = False)
    parser.add_argument('--saving', type =bool, default = False)
    parser.add_argument('--fix_camera', type =bool, default = True)

    parser.add_argument('--mixed_obs', type = bool, default = False)

    args = parser.parse_args()

    args.benchmark = True if args.benchmark == 'True' else False
    args.vis = True if args.vis == 'True' else False


    print('====**** This is Behavior Cloning ****====')

    print(args)

    test_policy(args)

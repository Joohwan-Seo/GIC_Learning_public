from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic

from rlkit.envs.wrappers import NormalizedBoxEnv

import argparse
import torch
import uuid
from rlkit.core import logger

import pickle

import numpy as np

import time

from gic_env.pih_env_TCIC import RobotEnv
from gic_env.pih_env_benchmark_TCIC import RobotEnvBenchmark

filename = str(uuid.uuid4())


def test_policy(args):


    data = torch.load(args.file)
    policy = data['evaluation/policy']

    use_ext_force = args.use_ext_force
    act_type = args.action_type

    cases = ['default','case1','case2','case3']

    # cases = ['default']

    num_testing = args.num_testing

    if args.benchmark:
        print('==========================')
        print('Testing Benchmark Scenario')
        print('==========================')
    else:
        print('==========================')
        print('Testing Original Scenario')
        print('==========================')
    for case in cases:
        if args.benchmark:
            env = NormalizedBoxEnv(RobotEnvBenchmark(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = case, testing = True,
                                                     window_size = args.window_size, use_ext_force=use_ext_force, act_type=act_type))
        else:
            env = NormalizedBoxEnv(RobotEnv(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = case, testing = True, 
                                            window_size = args.window_size, use_ext_force=use_ext_force, act_type=act_type))
        
        num_success = 0

        tic = time.time()

        if args.gpu:
            set_gpu_mode(True)
            policy.cuda()
        for i in range(num_testing):
            path = rollout(
                env,
                policy,
                max_path_length=args.H,
                render=False,
            )

            last_info = path['env_infos'][-1]
            if last_info['success']:
                num_success += 1

        toc = time.time() - tic
        print('Total time', toc)
        print('Case:',case,'--','success rate:',num_success/num_testing * 100)

    print("Policy loaded")

def test_policy_arbitrary(args):


    data = torch.load(args.file)
    policy = data['evaluation/policy']

    use_ext_force = args.use_ext_force
    act_type = args.action_type

    angles = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    # angles = [-80, 0, 50]

    num_testing = args.num_testing

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
        angle_rad = angle / 180 * np.pi
        if args.benchmark:
            env = NormalizedBoxEnv(RobotEnvBenchmark(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = 'arbitrary_x', testing = True, hole_angle = angle_rad,
                                                     window_size = args.window_size, use_ext_force=use_ext_force, act_type=act_type))
        else:
            env = NormalizedBoxEnv(RobotEnv(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = 'arbitrary_x', testing = True, hole_angle = angle_rad,
                                            window_size = args.window_size, use_ext_force=use_ext_force, act_type=act_type))
        
        num_success = 0

        tic = time.time()

        if args.gpu:
            set_gpu_mode(True)
            policy.cuda()

        reward_list = []
        length_list = []
        reward_all_list = []
        for i in range(num_testing):
            path = rollout(
                env,
                policy,
                max_path_length=args.H,
                render=False,
            )
            # dict_keys(['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'dones', 'agent_infos', 'env_infos', 'full_observations', 'full_next_observations'])
            # print(path.keys())
            # print(path['rewards'].shape)

            last_info = path['env_infos'][-1]
            if last_info['success']:
                num_success += 1

            reward_list.append(np.sum(path['rewards'], axis = 0))
            length_list.append(path['rewards'].shape[0])
            reward_all_list.append(path['rewards'])

        # dataset_small = dict(
        # reward = reward_list,
        # length_list = length_list,
        # )
        dataset_small = dict(
        reward = reward_all_list,
        )

        dataset[str(angle)] = dataset_small

        toc = time.time() - tic
        print('Total time', toc)
        print('Angle:',angle,'--','success rate:',num_success/num_testing * 100)

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the sna1pshot file')
    parser.add_argument('--H', type=int, default=10000,
                        help='Max length of rollout')
    parser.add_argument('--gpu', default = True, action='store_true')
    parser.add_argument('--case', type=str, default='default') # Testing different cases; tilted...
    parser.add_argument('--vis', type=str, default = 'False') # True will enable visualization
    parser.add_argument('--benchmark', type=str, default = 'False') # Testing for CIC
    parser.add_argument('--num_testing', type=int, default = 20) 
    parser.add_argument('--window_size', type=int, default = 1)
    parser.add_argument('--obs_type', type = str, default = 'pos_vel')
    parser.add_argument('--use_ext_force', type =bool, default = False) # Use always False
    parser.add_argument('--action_type', type = str, default = 'default')
    args = parser.parse_args()

    args.benchmark = True if args.benchmark == 'True' else False
    args.vis = True if args.vis == 'True' else False

    # test_policy(args)
    dataset = test_policy_arbitrary(args)

    file = open("./dataset/class_GIC_GCEV_RL_reward_all_posvel.pkl","wb")
    pickle.dump(dataset, file)
    file.close()

from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from gic_env.pih_env_benchmark import RobotEnvBenchmark
from data.Behavior_Cloning.policy_model import BCPolicy
from gic_env.pih_env_separated import RobotEnvSeparated
from gic_env.pih_env_separated_benchmark import RobotEnvSeparatedBenchmark
import argparse
import torch
import uuid
from rlkit.core import logger

import numpy as np

import time

from gic_env.pih_env import RobotEnv

filename = str(uuid.uuid4())


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

    ECGIC = args.ECGIC

    cases = ['default','case1','case2','case3']
    cases = ['case2']

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
    for case in cases:
        if args.benchmark:
            env = RobotEnvSeparatedBenchmark(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = case, testing = True, 
                                             window_size = args.window_size, ECGIC = ECGIC, mixed_obs = args.mixed_obs)
        else:
            env = RobotEnvSeparated(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = case, testing = True, 
                                    window_size = args.window_size, ECGIC = ECGIC, use_ext_force=use_ext_force, mixed_obs = args.mixed_obs)
        
        num_success = 0

        tic = time.time()

        if args.gpu:
            set_gpu_mode(True)
            policy.cuda()
            cuda0 = torch.device('cuda:0')
        for i in range(num_testing):
            obs = env.reset()

            for i in range(args.H):
                obs_tensor = torch.from_numpy(obs)

                obs_tensor = obs_tensor.to(torch.float32)                
                # norm_obs_tensor = norm_obs_tensor.double()
                # print(norm_obs_tensor)
                action = policy(obs_tensor.to(cuda0))
                action = action.cpu().detach().numpy()
                # action = env.get_expert_action()
                next_obs, rew, done, info =env.step(action)

                # print(type(next_obs))
                obs = next_obs
                
                if done :
                    # print(info['success'])
                    if info['success']:
                        num_success += 1
                    break        

        toc = time.time() - tic
        print('Total time', toc)
        print('Case:',case,'--','success rate:',num_success/num_testing * 100)

    print("Policy loaded")


def testing(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    print(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the sna1pshot file')
    parser.add_argument('--H', type=int, default=5000,
                        help='Max length of rollout')
    parser.add_argument('--gpu', default = True, action='store_true')
    parser.add_argument('--case', type=str, default='default')
    parser.add_argument('--vis', type=bool, default = True)
    parser.add_argument('--benchmark', type=bool, default = True)
    parser.add_argument('--num_testing', type=int, default = 100)
    parser.add_argument('--window_size', type=int, default = 1)
    parser.add_argument('--ECGIC', type = str, default = False)
    parser.add_argument('--obs_type', type = str, default = 'pos')
    parser.add_argument('--use_ext_force', type =bool, default = False)
    parser.add_argument('--mixed_obs', type = bool, default = True)
    args = parser.parse_args()


    print('====**** This is Behavior Cloning ****====')

    # print(args)

    # file_name = '/deeprl/research/GIC-RL/data/GIC-RL-nominal-expert-long/GIC_RL_nominal_expert_long_2023_01_27_22_27_18_0000--s-0/itr_6140.pkl'
    # args['file'] = file_name

    # simulate_policy(args)
    test_policy(args)
    # testing(args)

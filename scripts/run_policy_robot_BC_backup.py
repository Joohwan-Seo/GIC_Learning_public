from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from gic_env.pih_env_benchmark import RobotEnvBenchmark
from data.Fanuc_success.Behavior_cloning.policy_model import BCPolicy, BCPolicyLarger
from gic_env.pih_env import RobotEnv
from gic_env.pih_env_benchmark import RobotEnvBenchmark
import argparse
import torch
from rlkit.core import logger

import numpy as np

import time

from gic_env.pih_env import RobotEnv


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

    # if args.ood_test:
    #     cases = ['default','case1','case1b','case1c','case1d','case1e','case1f','case1g','case1h']
    #     # args.num_testing = 10
    # else:
    cases = ['default','case1','case2','case3']

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
        # print('Case:',case)
        if args.benchmark:
            env = RobotEnvBenchmark(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = case, testing = True, 
                                    window_size = args.window_size, mixed_obs = args.mixed_obs)
        else:
            env = RobotEnv(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = case, testing = True, 
                           window_size = args.window_size, use_ext_force=use_ext_force, mixed_obs = args.mixed_obs)
        
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
                action = policy(obs_tensor.to(cuda0))
                action = action.cpu().detach().numpy()
                next_obs, rew, done, info =env.step(action)

                obs = next_obs
                
                if done :
                    if info['success']:
                        num_success += 1
                    break        

        toc = time.time() - tic
        print('Total time', toc)
        print('Case:',case,'--','success rate:',num_success/num_testing * 100)

def test_policy_aug(args):

    # num_obs = 6 if args.obs_type == 'pos' else 12
    if args.obs_type == 'pos':
        num_obs = 6
    elif args.obs_type == 'pos_vel':
        num_obs = 12

    policy_weights = torch.load(args.file)
    # print(policy_weights.keys())
    if args.aug_test and args.mixed_obs:
        policy = BCPolicyLarger(input_shape = num_obs * args.window_size, output_shape = 6)
    else:
        policy = BCPolicy(input_shape = num_obs * args.window_size)
    policy.load_state_dict(policy_weights)
    policy.float()

    # if args.ood_test:
    #     cases = ['default','case1','case1b','case1c','case1d','case1e','case1f','case1g','case1h']
    #     # args.num_testing = 10
    # else:
    if args.in_dist:
        angles = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    else:
        angles = [-150,-140,-129, -120, -110, -100, 100, 110, 129, 140, 150]
    # angles = [-120, -90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90, 120]
    # angles = [-60, 0, 15, 30, 45]

    num_testing = args.num_testing

    use_ext_force = args.use_ext_force



    for angle in angles:
        # print('Case:',case)
        if args.benchmark:
            print('==========================')
            print('Testing Benchmark Scenario')
            print('==========================')
            env = RobotEnvBenchmark(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = 'arbitrary_x', testing = True, 
                            window_size = args.window_size, use_ext_force=use_ext_force, mixed_obs = args.mixed_obs,
                            hole_angle = np.pi * angle / 180)
        else:
            print('==========================')
            print('Testing Original Scenario')
            print('==========================')
            env = RobotEnv(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = 'arbitrary_x', testing = True, 
                            window_size = args.window_size, use_ext_force=use_ext_force, mixed_obs = args.mixed_obs,
                            hole_angle = np.pi * angle / 180)
        
        num_success = 0

        tic = time.time()

        if abs(angle) >= 120:
            simulation_step = 7000
        else:
            simulation_step = args.H

        if args.gpu:
            set_gpu_mode(True)
            policy.cuda()
            cuda0 = torch.device('cuda:0')
        for i in range(num_testing):
            obs = env.reset()

            for i in range(args.H):
                obs_tensor = torch.from_numpy(obs)

                obs_tensor = obs_tensor.to(torch.float32)                
                action = policy(obs_tensor.to(cuda0))
                action = action.cpu().detach().numpy()
                next_obs, rew, done, info =env.step(action)

                obs = next_obs
                
                if done :
                    if info['success']:
                        num_success += 1
                    break        

        toc = time.time() - tic
        print('Total time', toc)
        print('Case:',angle,'--','success rate:',num_success/num_testing * 100)

    if args.mixed_obs:
        print('GIC + CEV')
    else:
        print('GIC + GCEV')

def test_policy_aug_random(args):

    # num_obs = 6 if args.obs_type == 'pos' else 12
    if args.obs_type == 'pos':
        num_obs = 6
    elif args.obs_type == 'pos_vel':
        num_obs = 12

    policy_weights = torch.load(args.file)
    # print(policy_weights.keys())
    if args.aug_test and args.mixed_obs:
        policy = BCPolicyLarger(input_shape = num_obs * args.window_size, output_shape = 6)
    else:
        policy = BCPolicy(input_shape = num_obs * args.window_size)
    policy.load_state_dict(policy_weights)
    policy.float()

    num_testing = args.num_testing

    use_ext_force = args.use_ext_force



    # print('Case:',case)
    if args.benchmark:
        print('==========================')
        print('Testing Benchmark Scenario')
        print('==========================')
        env = RobotEnvBenchmark(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = 'arbitrary_x', testing = True, 
                        window_size = args.window_size, use_ext_force=use_ext_force, mixed_obs = args.mixed_obs,
                        hole_angle = 'random')
    else:
        print('==========================')
        print('Testing Original Scenario')
        print('==========================')
        env = RobotEnv(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = 'arbitrary_x', testing = True, 
                        window_size = args.window_size, use_ext_force=use_ext_force, mixed_obs = args.mixed_obs,
                        hole_angle = 'random', in_dist = args.in_dist)
    
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
            action = policy(obs_tensor.to(cuda0))
            action = action.cpu().detach().numpy()
            next_obs, rew, done, info = env.step(action)

            obs = next_obs
            
            if done :
                if info['success']:
                    num_success += 1
                break        

    toc = time.time() - tic
    print('Total time', toc)
    print('Case:', 'success rate:',num_success/num_testing * 100)

    if args.mixed_obs:
        print('GIC + CEV')
    else:
        print('GIC + GCEV')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the sna1pshot file')
    parser.add_argument('--H', type=int, default=7000,
                        help='Max length of rollout')
    parser.add_argument('--gpu', default = True, action='store_true')
    parser.add_argument('--case', type=str, default='default')
    parser.add_argument('--vis', type=str, default = 'False')
    parser.add_argument('--benchmark', type=str, default = 'False')
    parser.add_argument('--num_testing', type=int, default = 100)
    parser.add_argument('--window_size', type=int, default = 1)
    parser.add_argument('--obs_type', type = str, default = 'pos')
    parser.add_argument('--use_ext_force', type =bool, default = False)

    parser.add_argument('--mixed_obs', type = bool, default = False)
    parser.add_argument('--aug_test', type=bool, default = True)
    parser.add_argument('--in_dist', type=bool, default = False)

    parser.add_argument('--rand_test', type=bool, default = True)
    args = parser.parse_args()

    args.benchmark = True if args.benchmark == 'True' else False
    args.vis = True if args.vis == 'True' else False


    print('====**** This is Behavior Cloning ****====')

    print(args)
    if args.aug_test:
        if args.rand_test:
            test_policy_aug_random(args)
        else:
            test_policy_aug(args)
    else:
        test_policy(args)

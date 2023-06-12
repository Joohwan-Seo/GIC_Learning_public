from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from gic_env.pih_env_benchmark import RobotEnvBenchmark
from gic_env.pih_env_separated_benchmark import RobotEnvSeparatedBenchmark
from gic_env.pih_env_residual import RobotEnvResidual
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic

from rlkit.envs.wrappers import NormalizedBoxEnv

import argparse
import torch
import uuid
from rlkit.core import logger

import time

from gic_env.pih_env import RobotEnv
from gic_env.pih_env_separated import RobotEnvSeparated

filename = str(uuid.uuid4())


def test_policy(args):


    data = torch.load(args.file)
    # policy = data['evaluation/policy']

    if args.BC == 'behavior_cloning':
        print("============== Behavior Cloning ===========")
        
        policy = TanhGaussianPolicy(
            obs_dim=6,
            action_dim=6,
            hidden_sizes=[128, 128, 128],
        )
        policy.load_state_dict(data)
        policy = MakeDeterministic(policy)
    else:
        policy = data['evaluation/policy']
        # policy = data['exploration/policy']

    ECGIC = args.ECGIC
    TCGIC = args.TCGIC
    use_ext_force = args.use_ext_force
    act_type = args.action_type

    cases = ['default','case1','case2','case3']
    # cases = ['case2']
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
            env = NormalizedBoxEnv(RobotEnvSeparatedBenchmark(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = case, testing = True, window_size = args.window_size, ECGIC = ECGIC, TCGIC = TCGIC, use_ext_force=use_ext_force, act_type=act_type))
        else:
            if args.residual:
                env = NormalizedBoxEnv(RobotEnvResidual(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = case, testing = True, window_size = args.window_size, ECGIC = ECGIC, TCGIC = TCGIC))
            else:
                # env = NormalizedBoxEnv(RobotEnv(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = case, testing = True, window_size = args.window_size, ECGIC = ECGIC, TCGIC = TCGIC, use_ext_force=use_ext_force, act_type=act_type))
                env = NormalizedBoxEnv(RobotEnvSeparated(show_viewer = args.vis, obs_type = args.obs_type, hole_ori = case, testing = True, window_size = args.window_size, ECGIC = ECGIC, TCGIC = TCGIC, use_ext_force=use_ext_force, act_type=act_type))
        
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
                # print('I success')

        toc = time.time() - tic
        print('Total time', toc)
        print('Case:',case,'--','success rate:',num_success/num_testing * 100)

    print("Policy loaded")

def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    print(type(policy))
    if args.residual:
        env = NormalizedBoxEnv(RobotEnvResidual(show_viewer = True, obs_type = args.obs_type, hole_ori = args.case, testing = True, window_size = args.window_size, ECGIC = args.ECGIC, TCGIC = args.TCGIC))
    else:
        env = RobotEnv(show_viewer = True, obs_type = args.obs_type, hole_ori = args.case, testing = True, window_size = args.window_size, ECGIC = args.ECGIC, TCGIC = args.TCGIC, use_ext_force=args.use_ext_force, act_type = args.action_type)
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()

def testing(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']

    print(policy.state_dict())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the sna1pshot file')
    parser.add_argument('--H', type=int, default=10000,
                        help='Max length of rollout')
    parser.add_argument('--gpu', default = True, action='store_true')
    parser.add_argument('--case', type=str, default='default')
    parser.add_argument('--vis', type=bool, default = False)
    parser.add_argument('--benchmark', type=bool, default = True)
    parser.add_argument('--num_testing', type=int, default = 100)
    parser.add_argument('--window_size', type=int, default = 1)
    parser.add_argument('--ECGIC', type = str, default = False)
    parser.add_argument('--TCGIC', type = bool, default = False)
    parser.add_argument('--obs_type', type = str, default = 'pos')
    parser.add_argument('--use_ext_force', type =bool, default = False)
    parser.add_argument('--BC', type=str, default = None)
    parser.add_argument('--residual', type=bool, default = False)
    parser.add_argument('--action_type', type = str, default = 'minimal')
    args = parser.parse_args()

    if args.ECGIC:
        print('===========Utilizing ECGIC =============')
    else:
        print('===========Not Using ECGIC =============')


    if args.TCGIC:
        print('===========Utilizing TCGIC =============')
    else:
        print('===========Not Using TCGIC =============')

    # print(args)

    # file_name = '/deeprl/research/GIC-RL/data/GIC-RL-nominal-expert-long/GIC_RL_nominal_expert_long_2023_01_27_22_27_18_0000--s-0/itr_6140.pkl'
    # args['file'] = file_name

    def identity(x):
        return x

    # simulate_policy(args)
    test_policy(args)
    # testing(args)

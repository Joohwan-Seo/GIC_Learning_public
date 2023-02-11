from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger

from gic_env.pih_env import RobotEnv

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    print(type(policy))
    env = RobotEnv(show_viewer = True, obs_type = 'pos_vel', hole_ori = args.case, testing = True)
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
    print(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the sna1pshot file')
    parser.add_argument('--H', type=int, default=5000,
                        help='Max length of rollout')
    parser.add_argument('--gpu', default = True, action='store_true')
    parser.add_argument('--case', type=str, default='default')
    args = parser.parse_args()

    # print(args)

    # file_name = '/deeprl/research/GIC-RL/data/GIC-RL-nominal-expert-long/GIC_RL_nominal_expert_long_2023_01_27_22_27_18_0000--s-0/itr_6140.pkl'
    # args['file'] = file_name

    simulate_policy(args)
    # testing(args)

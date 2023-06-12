from gic_env.pih_env import RobotEnv
from gic_env.pih_env_residual import RobotEnvResidual
from gic_env.pih_env_benchmark import RobotEnvBenchmark

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import torch


def experiment(variant):
    if variant['obs_type'] is not None:
        obs_type = variant['obs_type']


    torch.manual_seed(variant['seed'])

    ECGIC = variant['ECGIC']
    window_size = variant['window_size']
    use_ext_force = variant['use_ext_force']


    if not variant['benchmark']:
        if not variant['residual']:
            expl_env = NormalizedBoxEnv(RobotEnv(show_viewer = False, obs_type = obs_type, window_size = window_size, ECGIC = ECGIC, use_ext_force=use_ext_force))
        else:
            expl_env = NormalizedBoxEnv(RobotEnvResidual(show_viewer = False, obs_type = obs_type, window_size = window_size, ECGIC = ECGIC))
        # eval_env = NormalizedBoxEnv(RobotEnv(show_viewer = False, obs_type = obs_type))
    elif variant['benchmark']:
        expl_env = NormalizedBoxEnv(RobotEnvBenchmark(show_viewer = False, obs_type = obs_type, window_size = window_size, ECGIC = ECGIC))

    eval_env = expl_env
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )

    if variant['use_pretrained_policy']:
        if not variant['benchmark']:
            if variant['obs_type'] == 'pos':
                pretrained_weight = torch.load('data/Behavior_Cloning/BC_policy_GIC_no_failcase_itr_39.pt')
            else:
                raise NotImplementedError('obs type not correct')
        else:
            if variant['obs_type'] == 'pos':
                pretrained_weight = torch.load('data/Behavior_Cloning/BC_policy_CIC_no_failcase_itr_39.pt')
            else:
                raise NotImplementedError('obs type not correct')
            
        print(pretrained_weight.keys())

        d1 = {
        "fc1.weight" : "fc0.weight", 
        "fc1.bias" : "fc0.bias", 
        "fc2.weight" : "fc1.weight", 
        "fc2.bias" : "fc1.bias" , 
        "fc3.weight" : "last_fc.weight", 
        "fc3.bias" : "last_fc.bias", 
        }

        changed_dict = dict((d1[key], value) for (key, value) in pretrained_weight.items())
            
        policy.load_state_dict(changed_dict)
        target_policy.load_state_dict(changed_dict)

    es = GaussianStrategy(
        action_space=expl_env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        exploration_policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=7500,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=7500,
            min_num_steps_before_training=15000,
            max_path_length=7500,
            batch_size=10000,
            use_expert_policy=True,
        ),
        trainer_kwargs=dict(
            discount=0.99,
        ),
        qf_kwargs=dict(
            hidden_sizes=[128, 128, 128],
        ),
        policy_kwargs=dict(
            hidden_sizes=[128, 128, 128],
        ),
        replay_buffer_size=int(1E6),

        ECGIC = False,
        obs_type = 'pos',
        benchmark = False,
        seed = int(9),
        window_size = int(1),
        use_ext_force = True,
        use_pretrained_policy = False,
        residual = True,
    )
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    setup_logger('TD3_GIC_residual_default', variant=variant)
    experiment(variant)

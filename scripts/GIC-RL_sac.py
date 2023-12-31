from gic_env.pih_env_TCIC import RobotEnv
from gic_env.pih_env_benchmark_TCIC import RobotEnvBenchmark

import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.launchers.launcher_util import set_seed
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm





def experiment(variant):
    if variant['obs_type'] is not None:
        obs_type = variant['obs_type']


    torch.manual_seed(variant['seed'])
    window_size = variant['window_size']
    use_ext_force = variant['use_ext_force']
    act_type = variant['action_type']
    reward_version = variant['reward_version']
    

    if not variant['benchmark']:
        expl_env = NormalizedBoxEnv(RobotEnv(show_viewer = False, 
                                             obs_type = obs_type, 
                                             window_size = window_size,  
                                             use_ext_force=use_ext_force,
                                             reward_version = reward_version, 
                                             act_type = act_type))
    elif variant['benchmark']:
        expl_env = NormalizedBoxEnv(RobotEnvBenchmark(show_viewer = False, 
                                                      obs_type = obs_type, 
                                                      window_size = window_size, 
                                                      use_ext_force=use_ext_force,
                                                      reward_version = reward_version,
                                                      act_type = act_type,
                                                      ))

    eval_env = expl_env
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M],
    )
    

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
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
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=128,
        replay_buffer_size=int(1E6),
        obs_type = "pos_vel",
        algorithm_kwargs=dict(
            num_epochs = 500,
            num_eval_steps_per_epoch=4000,
            num_trains_per_train_loop= 300,
            num_expl_steps_per_train_loop=4000,
            min_num_steps_before_training=200000,
            max_path_length=4000,
            batch_size = 4096,
            use_expert_policy= True
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=10,
            policy_lr=1e-3, 
            qf_lr=1e-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        benchmark = True,
        seed = int(5),
        window_size = int(1),
        use_ext_force = False,
        action_type = 'default',
        env_type = 'benchmark', # 'default' or 'benchmark' default is Geometric approach, benchmark is Cartesian approach
        reward_version = 'force_penalty' # None or 'force_penalty' None is without force penalty, force_penalty version is the version in the paper.
    )

    print('============================================')
    print('benchmark:', variant['benchmark']) # GIC or CIC - some scripts use this, and some scripts use the env_type argument
    print('window_size:', variant['window_size']) # testing for time-windowed observation input; not used, i.e., window_size = 1
    print('seed:', variant['seed']) # for changing the random seed
    print('==========================================')


    set_seed(variant['seed'])
    setup_logger('Fanuc_TCIC_Cartesian_6_action_posvel', variant=variant, snapshot_mode = "gap", snapshot_gap = 5)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)

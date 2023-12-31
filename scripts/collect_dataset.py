from gic_env.pih_env import RobotEnv
from gic_env.pih_env_benchmark import RobotEnvBenchmark
import numpy as np

from rlkit.envs.wrappers import NormalizedBoxEnv
import pickle

def collect_traj(variant):
    obs_type = variant['obs_type']
    dataset_size = variant['dataset_size']
    max_path_length = variant['max_path_length']
    max_episode_call = variant['max_episode_call']
    window_size = variant['window_size']
    use_ext_force = variant['use_ext_force']
    action_type = variant['action_type']
    mixed_obs = variant['mixed_obs']

    if variant['env_name'] == 'default':
        print('========== Default ENV ===========')
        env = NormalizedBoxEnv(RobotEnv(show_viewer = False, 
                                        obs_type = obs_type, 
                                        window_size = window_size, 
                                        use_ext_force = use_ext_force,
                                        act_type = action_type,
                                        mixed_obs = mixed_obs,
                                        ))
        
    elif variant['env_name'] == 'benchmark':
        print('========== Benchmark ENV ===========')
        env = NormalizedBoxEnv(RobotEnvBenchmark(show_viewer = False, 
                                                 obs_type = obs_type,
                                                 testing = False,
                                                 window_size = window_size,
                                                 use_ext_force = use_ext_force,
                                                 act_type = action_type,
                                                 mixed_obs = mixed_obs,
                                                 ))

    else:
        print('Error: Environment NOT SELECTED')
        raise NotImplementedError

    list_done = []
    list_action = []

    list_obs = []

    total_call = 0
    success_count = 0
    fail_count = 0
    num_episode = 0

    for _ in range(max_episode_call):
        obs = env.reset()

        current_length = 0
        
        for i in range (max_path_length):
            expert_action = env.get_expert_action()
            next_obs, rew, done, info = env.step(expert_action)
            obs = next_obs

            list_obs.append(obs)
            list_done.append(done)
            list_action.append(expert_action)

            total_call += 1
            current_length += 1

            if total_call % 1e5 == 0:
                print(total_call)

            if done or total_call == dataset_size:
                if total_call == dataset_size:
                    info['success'] = True
                break

        
        if info['success']:
            # print('success')
            success_count += 1
            num_episode += 1
        elif info['success'] == False:
            p = np.random.rand()
            if p < 1.2: # failcase true: 0.5, otherwise 1.2 or something larger than 1
                del list_obs[-current_length:]
                del list_done[-current_length:]
                del list_action[-current_length:]
                total_call -= current_length
            else:
                fail_count += 1
                num_episode += 1
                pass

        if total_call >= dataset_size:
            break

    list_done[-1] = True
    

    dataset = dict(
        obs = list_obs,
        done = list_done,
        actions = list_action,
    )

    print('total episodes:', num_episode)
    print('success counts:', success_count)
    print('fail counts:', fail_count)
    print('total_call', total_call)

    # print(dataset['observations'].shape)


    return dataset

def collect_traj_aug(variant):
    obs_type = variant['obs_type']
    dataset_size = variant['dataset_size']
    max_path_length = variant['max_path_length']
    max_episode_call = variant['max_episode_call']
    window_size = variant['window_size']
    use_ext_force = variant['use_ext_force']
    action_type = variant['action_type']
    mixed_obs = variant['mixed_obs']

    list_obs = []
    list_action = []
    list_done = []

    # angles = [-90, -80, -75, -70, -60, -50, -45, -40, -30, -20, -15, -10, 0, 10, 15, 20, 30, 40, 45, 50, 60, 70, 75, 80, 90]
    angles = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]

    dataset = {}

    for angle in angles:
        list_obs = []
        list_action = []
        list_done = []
        print('Collecting with the case:', angle)

        if variant['env_name'] == 'default':
            raise RuntimeError('Should not be default environment')
            env = NormalizedBoxEnv(RobotEnv(show_viewer = False, 
                                            obs_type = obs_type, 
                                            window_size = window_size, 
                                            use_ext_force = use_ext_force,
                                            act_type = action_type,
                                            mixed_obs = mixed_obs,
                                            hole_ori = 'arbitrary_x',
                                            hole_angle = np.pi * angle/180,
                                            ))
            
        elif variant['env_name'] == 'benchmark':
            print('========== Benchmark ENV ===========')
            env = NormalizedBoxEnv(RobotEnvBenchmark(show_viewer = False, 
                                                    obs_type = obs_type,
                                                    testing = False,
                                                    window_size = window_size,
                                                    use_ext_force = use_ext_force,
                                                    act_type = action_type,
                                                    mixed_obs = mixed_obs,
                                                    hole_ori = 'arbitrary_x',
                                                    hole_angle = np.pi * angle/180,
                                                    ))

        total_call = 0
        success_count = 0
        fail_count = 0
        num_episode = 0

        for _ in range(max_episode_call):
            obs = env.reset()

            current_length = 0
            
            for i in range (max_path_length):
                expert_action = env.get_expert_action()
                next_obs, rew, done, info = env.step(expert_action)

                expert_action_mod = env.get_expert_action_modified()
                # print(expert_action[3:], expert_action_pos)
                expert_action = np.hstack((expert_action_mod))
                
                obs = next_obs

                list_obs.append(obs)
                list_done.append(done)
                list_action.append(expert_action)

                total_call += 1
                current_length += 1

                if total_call % 1e5 == 0:
                    print(total_call)

                if done or total_call == dataset_size:
                    if total_call == dataset_size:
                        info['success'] = True
                    break

            
            if info['success']:
                # print('success')
                success_count += 1
                num_episode += 1
            elif info['success'] == False:
                # print('fail')
                p = np.random.rand()
                if p < 1.2: # failcase true: 0.5, otherwise 1.2 or something larger than 1
                    del list_obs[-current_length:]
                    del list_done[-current_length:]
                    del list_action[-current_length:]
                    total_call -= current_length
                else:
                    fail_count += 1
                    num_episode += 1
                    pass

            if total_call >= dataset_size:
                break

        list_done[-1] = True

        dataset_small = dict(
        obs = list_obs,
        done = list_done,
        actions = list_action,
        )

        dataset[str(angle)] = dataset_small

        # list_obs_large.append(list_obs)
        # list_action_large.append(list_action)
        # list_done_large.append(list_done)

        print('total episodes:', num_episode)
        print('success counts:', success_count)
        print('fail counts:', fail_count)
        print('total_call', total_call)


    return dataset

if __name__ == "__main__":
    variant = dict(
        obs_type = 'pos',
        dataset_size = int(1e6),
        max_path_length = 4000,
        max_episode_call = 300,
        env_name = 'benchmark',  #default vs benchmark
        window_size = 1,
        use_ext_force = False,
        residual = False,
        action_type = 'default',
        env_type = 'pih_env_separated',
        mixed_obs = True,
        data_augmentation = False,
    )

    print('Current Environment is', variant['env_name'])
    if variant['data_augmentation']:
        dataset = collect_traj_aug(variant)
    else:
        dataset = collect_traj(variant)

    file = open("./dataset/dataset_CIC_BC_GCEV_300_default.pkl","wb")
    pickle.dump(dataset, file)
    file.close()
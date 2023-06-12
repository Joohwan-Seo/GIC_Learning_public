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
    ECGIC = variant['ECGIC']


    if variant['env_name'] == 'default':
        print('========== Default ENV ===========')
        env = NormalizedBoxEnv(RobotEnv(show_viewer = False, obs_type = obs_type, window_size = window_size, ECGIC = ECGIC))
        
    elif variant['env_name'] == 'benchmark':
        print('========== Benchmark ENV ===========')
        env = NormalizedBoxEnv(RobotEnvBenchmark(show_viewer = False, obs_type = obs_type, window_size = window_size))

    else:
        print('Error: Environment NOT SELECTED')
        raise NotImplementedError

    list_x = []
    list_R = []
    list_eg = []
    list_ev = []
    list_rew = []
    list_done = []
    list_action = []

    total_call = 0
    success_count = 0
    fail_count = 0
    num_episode = 0

    for _ in range(max_episode_call):
        obs = env.reset()

        current_length = 0
        
        for i in range (max_path_length):
            eg, ev, x, R = env.get_custom_obs_data_collection()
            expert_action = env.get_expert_action()
            _,_,done,info = env.step(expert_action)
            
            list_x.append(x)
            list_R.append(R)
            list_eg.append(eg)
            list_ev.append(ev)

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
                del list_x[-current_length:]
                del list_R[-current_length:]
                del list_eg[-current_length:]
                del list_ev[-current_length:]
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
        x = list_x,
        R = list_R,
        eg = list_eg,
        ev = list_ev,
        done = list_done,
        actions = list_action,
    )

    print('total episodes:', num_episode)
    print('success counts:', success_count)
    print('fail counts:', fail_count)
    print('total_call', total_call)

    # print(dataset['observations'].shape)


    return dataset

if __name__ == "__main__":
    variant = dict(
        obs_type = 'pos',
        dataset_size = int(2e6),
        max_path_length = 7500,
        max_episode_call = 10000,
        env_name = 'default',  #default vs benchmark
        window_size = 1,
        ECGIC = False,
    )

    print('Current Environment is', variant['env_name'])
    dataset = collect_traj(variant)

    file = open("./dataset/dataset_GIC_potential_learning.pkl","wb")
    pickle.dump(dataset, file)
    file.close()
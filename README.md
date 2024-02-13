# GIC-RL + BC
Written by Joohwan Seo, Ph.D. Student in Mechanical Engineering, UC Berkeley.

## Author's Comment 
This is really nasty code. A lot of unnecessary function/variables are still in the code - I tried my best to cleanse them. 

### Expert trajectory injection part can be found in:
``rlkit/core/batch_rl_algorithm.py``
### Scripts files
1. ``collect_dataset.py`` is collecting expert dataset for the BC.
2. ``GIC-RL_sac.py`` is for the the reinforcement learning part. 
3. ``data_analyzer.py`` is to check the learning curve during the RL agent training. - I did not upload the whole training files, only fraction or the success cases can be found in ``data/Fanuc_success/``
4. ``run_policy_robot_BC.py`` is to check the trained policy by the Behavior Cloning.
5. ``run_policy_robot.py`` is to check the trained policy by the RL. 

Hyperparameters and the statements are denoted as annotations in each file as much as possible. 

## Example files to run
### Behavior Cloning 
For GIC + GCEV
```
python scripts/run_policy_robot_BC.py data/Fanuc_success/Behavior_cloning/BC_policy_GIC_GCEV_300_itr_39.pt --benchmark False --vis True
```

For CIC + CEV
```
python scripts/run_policy_robot_BC.py data/Fanuc_success/Behavior_cloning/BC_policy_CIC_CEV_300_default_39.pt --benchmark True --vis True
```

If you want a mixed observation result, change ``mixed_obs`` to ``True``

### Reinforcement Learning
For GIC + GCEV
```
python scripts/run_policy_robot.py data/Fanuc_success/final_GIC_minimal_separated_pos_3x128_reward2/itr_60.pkl --benchmark False --vis True
```

For CIC + CEV
```
python scripts/run_policy_robot.py data/Fanuc_success/final_CIC_minimal_separated_pos_3x128_reward2/itr_90.pkl --benchmark True --vis True
```

## Based on
Berkeley RL Kit for the Reinforcement Learning\
Directly Imported from ``https://github.com/rail-berkeley/rlkit``

Environmental Setup using Mujoco\
``https://github.com/deanpham98/learn-seq``

## Geometric Impedance Control from
https://doi.org/10.48550/arXiv.2211.07945 \
and \
https://github.com/Joohwan-Seo/Geometric-Impedance-Control-Public

## Python path work should be done before running this code
```
export PYTHONPATH=/your_directory_to_the_GIC_Learning_public_folder:$PYTHONPATH
```

## Version
mujoco: 2.0.0 \
python: 3.6.13\
cuda: 11.4 (Trained with GPU RTX3060 12GB)

## Accepted and will be presented at
Robotics and Automation Letters (RAL) and IROS 2024

Seo et al., Contact-rich SE(3)-Equivariant Robot Manipulation Task Learning via Geometric Impedance Control

### Bibtex Citation
@article{seo2023contact,\
  title={Contact-rich SE (3)-Equivariant Robot Manipulation Task Learning via Geometric Impedance Control},\
  author={Seo, Joohwan and Prakash, Nikhil PS and Zhang, Xiang and Wang, Changhao and Choi, Jongeun and Tomizuka, Masayoshi and Horowitz, Roberto},\
  journal={IEEE Robotics and Automation Letters},\
  year={2023},\
  publisher={IEEE}\
}

## Project Website
https://sites.google.com/berkeley.edu/equivariant-task-learning/home

import numpy as np
import matplotlib.pyplot as plt

from src.util import load_tools, plot_tools
from src.lpvds_class import lpvds_class
import os, pickle


def plot_expert_data(expert_path, num_rollouts, load_type='EXPERT'):
    import matplotlib

    matplotlib.use('TkAgg')
    rollouts_dir = os.path.join(expert_path, 'files/' + load_type + '/rollouts')
    all_files = os.listdir(rollouts_dir)

    pkl_files = [f for f in all_files if f.endswith('.pkl')]
    pkl_files = sorted(pkl_files, key=lambda x: int(x.split('.')[0]))
    if num_rollouts > len(pkl_files):
        print("Warning! Requested number of rollouts exceeds the available files. Using all available files.")
        num_rollouts = len(pkl_files)

    selected_indices = np.arange(num_rollouts)
    selected_files = [pkl_files[idx] for idx in selected_indices]
    expert_length = []


    # # 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # for idx, file_name in enumerate(selected_files):
    #     with open(os.path.join(rollouts_dir, file_name), 'rb') as f:
    #         data = pickle.load(f)
    #
    #     expert_length.append(int(data['lengths']))
    #     expert_obs = data['observations'][:expert_length[-1]]
    #
    #     # Plot the expert_obs
    #     ax.plot(expert_obs[:, 0], expert_obs[:, 1], expert_obs[:, 2], label=f'Trajectory {idx}')
    #
    #     ax.text(expert_obs[0, 0], expert_obs[0, 1], expert_obs[0, 2], f'{file_name}', color='red')
    #     if len(expert_obs) > 5:
    #         ax.text(expert_obs[5, 0], expert_obs[5, 1], expert_obs[5, 2], f'{file_name}', color='blue')
    # plt.show()

    # 2D plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    for idx, file_name in enumerate(selected_files):
        with open(os.path.join(rollouts_dir, file_name), 'rb') as f:
            data = pickle.load(f)

        expert_length.append(int(data['lengths']))
        expert_obs = data['observations'][:expert_length[-1]]

        # Plot the expert_obs
        ax.plot(expert_obs[:, 0], expert_obs[:, 1], label=f'Trajectory {idx}')

        ax.text(expert_obs[0, 0], expert_obs[0, 1], f'{file_name}', color='red')
        if len(expert_obs) > 5:
            ax.text(expert_obs[5, 0], expert_obs[5, 1], f'{file_name}', color='blue')
    plt.show()


def load_expert_data(expert_path, num_rollouts, load_type='EXPERT'):
    rollouts_dir = os.path.join(expert_path, 'files/' + load_type + '/rollouts')
    all_files = os.listdir(rollouts_dir)

    pkl_files = [f for f in all_files if f.endswith('.pkl')]
    pkl_files = sorted(pkl_files, key=lambda x: int(x.split('.')[0]))
    if num_rollouts > len(pkl_files):
        print("WWarning! Requested number of rollouts exceeds the available files. Using all available files.")
        num_rollouts = len(pkl_files)

    selected_indices = np.arange(num_rollouts)
    # set_random_seed(int(time.time()))
    # selected_indices = np.random.choice(len(pkl_files), num_rollouts, replace=False)
    # selected_indices = np.array([1,2,4,6,7,12,13,14,18,21,24,25,26,27])
    # selected_indices = np.array([1,2, 3])

    selected_files = [pkl_files[idx] for idx in selected_indices]
    expert_length = []
    for idx, file_name in enumerate(selected_files):
        with open(os.path.join(rollouts_dir, file_name), 'rb') as f:
            data = pickle.load(f)
        if idx == 0:
            expert_obs = data['observations']
            expert_acs = data['actions']
            expert_rew = data['rewards']
        else:
            expert_obs = np.concatenate([expert_obs, data['observations']], axis=0)
            expert_acs = np.concatenate([expert_acs, data['actions']], axis=0)
            expert_rew = np.concatenate([expert_rew, data['rewards']], axis=0)
        expert_length.append(int(data['lengths']))
    expert_mean_reward = np.mean(expert_rew)
    expert_length = np.array(expert_length)
    return (expert_obs, expert_acs, expert_rew), expert_length, expert_mean_reward

expert_path = '/home/baiyu/PycharmProjects/icrl-master/icrl/expert_data/PointDS'
# plot_expert_data(expert_path,99)
(x, x_dot, _), length, _ = load_expert_data(expert_path,99)
x_att = np.array([[0., 0.]])
x_init = [x[sum(length[:i])].reshape(1, 2) for i in range(len(length))]
# if x.shape[1] == 2:# if x.shape[1] == 2:
#     x = np.c_[x, np.zeros(x.shape[0])]
#     x_dot = np.c_[x_dot, np.zeros(x_dot.shape[0])]
#     x_att = np.c_[x_att, np.zeros(x_att.shape[0])]
#     for i in range(len(x_init)):
#         x_init[i] = np.c_[x_init[i], np.zeros(x_init[i].shape[0])]

# run lpvds
lpvds = lpvds_class(x, x_dot, x_att)
# lpvds.begin()
lpvds.logIN('output_pos.json')
# evaluate results
x_test_list = []
for x_0 in x_init:
    x_test_list.append(lpvds.sim(x_0, dt=0.01))

# plot results
plot_tools.plot_gmm(x, lpvds.assignment_arr, lpvds.damm)
if x.shape[1] == 2:
    plot_tools.plot_ds_2d(x, x_test_list, lpvds)
else:
    plot_tools.plot_ds_3d(x, x_test_list)
plt.show()
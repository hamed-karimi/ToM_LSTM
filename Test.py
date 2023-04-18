import os.path
import matplotlib.pyplot as plt

import torch
import Utilities
from torch.utils.tensorboard import SummaryWriter
from Visualizer import visualizer
from DataSet import get_agent_appearance
from ObjectFactory import ObjectFactory


def load_tom_net(factory, utility):
    tom_net = factory.get_tom_net()
    weights = torch.load('./Model/ToM_RNN.pt')
    tom_net.load_state_dict(weights)
    return tom_net


def test(test_data_generator):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utility = Utilities.Utilities()
    writer = SummaryWriter()
    test_figures_dir = './Test/'
    if not os.path.exists(test_figures_dir):
        os.mkdir(test_figures_dir)

    agent_appearance = torch.tensor(get_agent_appearance(),
                                    dtype=torch.float32)  # add one dimension for batch
    factory = ObjectFactory(utility=utility)
    tom_net = load_tom_net(factory, utility).to(device)
    global_index = 0
    width, height = 5, 5
    grids_in_fig = width*height
    env_input = []
    pred_goals, pred_actions = [], []
    true_goals, true_needs, true_actions = [], [], []
    for test_idx, data in enumerate(test_data_generator):
        # environment_batch.shape: [batch_size, step_num, objects+agent(s), height, width]
        # target_goal: 2 is staying
        environments_batch, goals_batch, actions_batch, needs_batch = data
        agent_appearance_batch = agent_appearance.repeat(environments_batch.shape[0], 1, 1, 1)
        step_num = environments_batch.shape[1]

        seq_start = True

        for step in range(step_num):
            print(global_index)
            goals, goals_prob, actions, actions_prob = tom_net(environments_batch[:, step, :, :, :],
                                                               agent_appearance_batch,
                                                               seq_start)
            seq_start = False

            true_goals.append(goals_batch[:, step])
            true_actions.append(actions_batch[:, step])
            true_needs.append(needs_batch[:, step, :])
            env_input.append(environments_batch[:, step, :, :, :])

            pred_goals.append(goals_prob)
            pred_actions.append(actions_prob)

            if (global_index + 1) % grids_in_fig == 0:
                fig, ax = visualizer(height, width, env_input, true_goals,
                                     true_actions, true_needs, pred_goals, pred_actions)
                env_input = []
                pred_goals, pred_actions = [], []
                true_goals, true_needs, true_actions = [], [], []
                fig.savefig('{0}/{1}_{2}.png'.format(test_figures_dir, global_index-width*height+1, global_index+1))
                plt.close()

            global_index += 1



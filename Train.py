import os.path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler, BatchSampler, DataLoader, SubsetRandomSampler, Sampler
import Utilities
from DataSet import AgentActionDataSet, get_agent_appearance
from ObjectFactory import ObjectFactory


def get_data_loader(utility):
    def get_generator_from_sampler(sampler: Sampler, batch_size):
        batch_sampler = BatchSampler(sampler=sampler,
                                     batch_size=batch_size,
                                     drop_last=False)
        params = {'batch_sampler': batch_sampler,
                  'pin_memory': True}

        generator = DataLoader(dataset, **params)
        return generator

    dataset = AgentActionDataSet()
    train_batch_size = utility.params.BATCH_SIZE
    train_sampler = SubsetRandomSampler(np.arange(int(utility.params.TRAIN_PROPORTION * len(dataset))))
    test_sampler = SequentialSampler(np.arange(int(utility.params.TRAIN_PROPORTION * len(dataset)), len(dataset)))

    train_generator = get_generator_from_sampler(train_sampler, batch_size=train_batch_size)
    test_generator = get_generator_from_sampler(test_sampler, batch_size=1)

    return train_generator, test_generator


def change_require_grads(model, goal_grad, action_grad):
    for params in model.parameters():
        params.requires_grad = goal_grad
    for params in model.fc_action.parameters():
        params.requires_grad = action_grad


def train(train_data_generator):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utility = Utilities.Utilities()
    writer = SummaryWriter()

    agent_appearance = torch.tensor(get_agent_appearance(),
                                    dtype=torch.float32)  # add one dimension for batch
    factory = ObjectFactory(utility=utility)
    tom_net = factory.get_tom_net().to(device)
    optimizer = torch.optim.Adam(tom_net.parameters(),
                                 lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    global_index = 0

    for epoch in range(utility.params.NUM_EPOCHS):
        for train_idx, data in enumerate(train_data_generator):
            # environment_batch.shape: [batch_size, step_num, objects+agent(s), height, width]
            # target_goal: 2 is staying
            environments_batch, goals_batch, actions_batch, needs_batch = data
            agent_appearance_batch = agent_appearance.repeat(environments_batch.shape[0], 1, 1, 1)
            step_num = environments_batch.shape[1]
            batch_goal_loss = 0
            batch_action_loss = 0
            seq_start = True
            criterion = nn.NLLLoss()
            for step in range(step_num):
                optimizer.zero_grad()
                goals, goals_prob, actions, actions_prob = tom_net(environments_batch[:, step, :, :, :],
                                                                   agent_appearance_batch,
                                                                   seq_start)
                seq_start = False

                change_require_grads(tom_net, goal_grad=True, action_grad=False)

                # loss of goal nets
                goal_loss = criterion(goals_prob, goals_batch[:, step].long())
                goal_loss.backward(retain_graph=True)

                # loss of action net
                change_require_grads(tom_net, goal_grad=False, action_grad=True)
                correct_goals_mask = (torch.argmax(goals_prob, dim=1) == goals_batch[:, step])
                correct_goal_indices = torch.argwhere(correct_goals_mask).squeeze()
                action_loss = criterion(actions_prob[correct_goal_indices, :], actions_batch[correct_goal_indices, step].long())
                action_loss.backward()

                optimizer.step()

                batch_goal_loss += goal_loss.item()
                batch_action_loss += action_loss.item()
                # tom_net.mental_net.h_0.detach_()
                # tom_net.mental_net.c_0.detach_()

            writer.add_scalar("Loss/goal", batch_goal_loss / step_num, global_index)
            writer.add_scalar("Loss/action", batch_action_loss / step_num, global_index)
            global_index += 1

            print('epoch: ', epoch, ', batch: ', train_idx)
    writer.flush()
    if not os.path.exists('./Model'):
        os.mkdir('./Model')
    torch.save(tom_net.state_dict(), './Model/ToM_RNN.pt')

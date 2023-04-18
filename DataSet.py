import gc
import pickle
import random
from datetime import datetime
from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join
import Utilities


class AgentActionDataSet(Dataset):
    def __init__(self):
        self.utility = Utilities.Utilities()
        params = self.utility.get_params()
        self.dir_path = params.DATA_DIRECTORY
        # environment_file_object = open(join(self.dir_path, 'environments.pt'), 'rb')
        # goal_file_object = open(join(self.dir_path, 'selected_goals.pt'), 'rb')
        # actions_file_object = open(join(self.dir_path, 'actions.pt'), 'rb')

        self.environments = torch.load(join(self.dir_path, 'environments.pt'))
        self.target_goals = torch.load(join(self.dir_path, 'selected_goals.pt'))
        self.target_actions = torch.load(join(self.dir_path, 'actions.pt'))
        self.target_needs = torch.load(join(self.dir_path, 'needs.pt'))

        # environment_file_object.close()
        # goal_file_object.close()
        # actions_file_object.close()

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.environments)

    def __getitem__(self, episode: int):
        return self.environments[episode, :, :, :, :], \
            self.target_goals[episode, :], \
            self.target_actions[episode, :], \
            self.target_needs[episode, :, :]
        # upper_bound = min(self.__len__()-episode, 100)
        # lower_bound = min(upper_bound, 10)
        # sequence_len = random.randint(lower_bound, upper_bound)
        # return self.environments[episode: episode+sequence_len],\
        #     self.target_goals[episode: episode+sequence_len],\
        #     self.target_actions[episode: episode+sequence_len], \
        #     sequence_len


def get_agent_appearance():
    utility = Utilities.Utilities()
    params = utility.get_params()
    dir_path = params.DATA_DIRECTORY
    agent_face_file_object = open(join(dir_path, 'agent_face.pkl'), 'rb')
    agent_face = pickle.load(agent_face_file_object)
    return np.expand_dims(agent_face, axis=0)

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

        self.environments = torch.load(join(self.dir_path, 'environments.pt'))
        self.target_goals = torch.load(join(self.dir_path, 'selected_goals.pt'))
        self.target_actions = torch.load(join(self.dir_path, 'actions.pt'))
        self.target_needs = torch.load(join(self.dir_path, 'needs.pt'))
        self.reached_goal = torch.load(join(self.dir_path, 'goal_reached.pt'))
        print('dataset size: ', self.environments.shape[:2])

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.environments)

    def __getitem__(self, episode: int):
        return self.environments[episode, :, :, :, :], \
            self.target_goals[episode, :], \
            self.target_actions[episode, :], \
            self.target_needs[episode, :, :], \
            self.reached_goal[episode, :]


def get_agent_appearance():
    utility = Utilities.Utilities()
    params = utility.get_params()
    dir_path = params.DATA_DIRECTORY
    agent_face_file_object = open(join(dir_path, 'agent_face.pkl'), 'rb')
    agent_face = pickle.load(agent_face_file_object)
    return np.expand_dims(agent_face, axis=0)

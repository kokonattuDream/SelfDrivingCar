# -*- coding: utf-8 -*-
import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

import torch.autograd as autograd
from torch.autograd import Variable

#Neural Network of ai
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action

        #Connection with input layer and hidden layer
        self.fc1 = nn.Linear(input_size, 30)
        #Connection with hidden layer and output layer
        self.fc2 = nn.Linear(30, nb_action)
    
    #Q-values of each action
    def forward(self, state):
        
        x = f.relu(self.fc1(state))
        q_values = self.fc2(x)
        
        return q_values
        
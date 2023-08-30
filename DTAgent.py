import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple, deque
from itertools import count
import random
import time
import os
import re

class DTAgent:
    def __init__(self, env, lr=1e-4, device='cpu', experience_pool=100000, batch_size=200):
        self.env = env
        self.lr = lr
        self.device=device
        self.dim_action, self.dim_state = env.action_space.shape[0], env.observation_space.shape[0]
        self.batch_size = batch_size

        # the model approximator
        self.dt = Net(self.dim_state+self.dim_action, 20, self.dim_state).to(self.device)
        self.optimizer = torch.optim.SGD(self.dt.parameters(), lr=lr)  # optimizer
        self.loss_func = torch.nn.MSELoss()  # loss function, MSE loss
        self.memory = ReplayMemory(experience_pool)  # experience replay pool

    def choose_action(self, state):
        # generate random actions
        return self.env.action_space.sample()

    def learn(self, writer=None, episode=None):
        # writer: the TensorFlow writer
        # episode: episode numbers
        if len(self.memory) < self.batch_size:
            return
        # start_time = time.time()
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state)
        next_state_batch = torch.stack(batch.next_state)
        action_batch = torch.stack(batch.action)
        # Compute loss
        loss = self.loss_func(self.dt(torch.cat([state_batch, action_batch], dim=1)), next_state_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

# define the experience pool
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        # self.bn1 = nn.BatchNorm1d(n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        # self.bn1 = nn.BatchNorm1d(n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        # out = self.bn1(out)
        out = F.relu(out)
        out = self.hidden2(out)
        # out = self.bn2(out)
        out = F.relu(out)
        out = self.predict(out)

        return out

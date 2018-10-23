import torch
import torch.nn as nn
import numpy as np

init_w = 3e-3
init_b = 3e-4

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(in_features=obs_size, out_features=400)
        self.l2 = nn.Linear(in_features=400, out_features=300)
        self.l3 = nn.Linear(in_features=300, out_features=n_actions)

        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.fill_(init_b)

    def __call__(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x

class Critic(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(in_features=obs_size + n_actions, out_features=400)
        self.l2 = nn.Linear(in_features=400, out_features=300)
        self.l3 = nn.Linear(in_features=300, out_features=n_actions)

        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.fill_(init_b)

    def __call__(self, x, a):
        x = torch.cat((x,a), 1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x
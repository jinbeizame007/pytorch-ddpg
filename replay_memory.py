import torch
import numpy as np

class ReplayMemory:
    def __init__(self, memory_size=100000, batch_size=128, obs_size=3, n_action=1):
        self.index = 0
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.obs = np.zeros((self.memory_size, obs_size), dtype=np.float32)
        self.action = np.zeros((self.memory_size, n_action), dtype=np.float32)
        self.reward = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.memory_size, obs_size), dtype=np.float32)
        self.terminal = np.zeros((self.memory_size, 1), dtype=np.int32)
        self.priority = np.zeros((self.memory_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, terminal):
        self.obs[self.index % self.memory_size] = obs
        self.action[self.index % self.memory_size] = action
        self.reward[self.index % self.memory_size][0] = reward
        self.next_obs[self.index % self.memory_size] = next_obs
        self.terminal[self.index % self.memory_size][0] = terminal
        self.index += 1
    
    def sample(self):
        indexes = np.random.randint(0, min(self.memory_size, self.index), self.batch_size)
        obs = torch.Tensor(self.obs[indexes]).cuda()
        action = torch.Tensor(self.action[indexes]).cuda()
        reward = torch.Tensor(self.reward[indexes]).cuda()
        next_obs = torch.Tensor(self.next_obs[indexes]).cuda()
        terminal = torch.Tensor(self.terminal[indexes]).cuda()
        return obs, action, reward, next_obs, terminal
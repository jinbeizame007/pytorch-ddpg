import torch
import numpy as np

def soft_update(target_model, model, tau):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def get_obs(observation):
    obs = []
    for o in observation.values():
        if type(o) is np.ndarray:
            obs += list(o)
        else:
            obs.append(o)
    return np.array([obs], dtype=np.float32)

class OrnsteinUhlenbeckProcess():
    def __init__(self, n_action, scale=0.01, mu=0, theta=0.15, sigma=0.2):
        self.n_action = n_action
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x = np.ones(self.n_action) * self.mu
        self.reset()

    def reset(self):
        self.x = np.ones(self.n_action) * self.mu

    def noise(self):
        dx = -self.theta * (self.x - self.mu) * self.scale + self.sigma * np.sqrt(self.scale) * np.random.randn(len(self.x))
        self.x = self.x + dx
        return self.x
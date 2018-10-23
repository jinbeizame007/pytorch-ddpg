import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from dm_control import suite
import numpy as np
from copy import deepcopy
import random
import math
import csv

from models import Critic, Actor
from replay_memory import ReplayMemory
from utils import soft_update, get_obs, OrnsteinUhlenbeckProcess

env = suite.load(domain_name="walker", task_name="run")
n_actions = env.action_spec().shape[0]
obs_size = get_obs(env.reset().observation).shape[1]

actor_lr = 1e-4
critic_lr = 1e-3
lambda_l2 = 0.0001
gamma = 0.9
tau = 0.001
epsilon = 1.0
gradient_step = 1
warmup_step = 5000
memory_size = 1000000
batch_size = 64
memory = ReplayMemory(memory_size, batch_size, obs_size, n_actions)
ouprocess = OrnsteinUhlenbeckProcess(n_actions)

actor = Actor(obs_size, n_actions).cuda()
critic = Critic(obs_size, n_actions).cuda()
target_actor = deepcopy(actor).cuda()
target_critic = deepcopy(critic).cuda()

optimizer_actor = optim.Adam(actor.parameters(), lr=actor_lr)
optimizer_critic = optim.Adam(critic.parameters(), lr=critic_lr)

actor_criterion = nn.MSELoss()
critic_criterion = nn.MSELoss()

reward_tmp = 0
episode = 0
step = 0
action = None
rewards = []

while True:
    time_step = env.reset()
    obs = get_obs(time_step.observation)

    if episode != 0:
        print("episode:",episode ," steps:", step_in_episode, " reward:", reward_tmp)
    ouprocess.reset()
    reward_tmp = 0
    episode += 1
    step_in_episode = 0

    while not time_step.last():
        step += 1
        step_in_episode += 1

        action = actor(torch.from_numpy(obs).cuda()).detach().cpu().numpy()[0]
        action += max(epsilon, 0) * ouprocess.noise()
        action = np.clip(action, -1, 1)
        epsilon -= 1./50000.

        reward = 0.
        for i in range(4):
            time_step = env.step(action)
            next_obs = get_obs(time_step.observation)
            reward += time_step.reward
            if time_step.last():
                break

        terminal = 0.
        if time_step.last():
            terminal = 1.
        memory.add(obs, action, reward, next_obs, terminal)
        obs = deepcopy(next_obs)
        reward_tmp += reward
            
        if step < warmup_step:
            if time_step.last():
                break
            continue
        
        for i in range(gradient_step):
            obs_batch, action_batch, reward_batch, next_obs_batch, terminal_batch = memory.sample()

            ### update critic ###
            q_value = critic(obs_batch, action_batch)
            next_q_value = target_critic(next_obs_batch, target_actor(next_obs_batch))
            q_target = reward_batch + (1. - terminal_batch) * gamma * next_q_value
            critic_loss = critic_criterion(q_value, q_target.detach())

            # l2 loss
            critic_params = torch.cat([x.view(-1) for x in critic.parameters()])
            l2_loss = lambda_l2 * torch.norm(critic_params, 2)
            critic_loss += l2_loss

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            ### update actor ###
            actor_loss = -critic(obs_batch, actor(obs_batch))
            actor_loss = actor_loss.mean()

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            ### update target networks ###
            soft_update(target_critic, critic, tau)
            soft_update(target_actor, actor, tau)
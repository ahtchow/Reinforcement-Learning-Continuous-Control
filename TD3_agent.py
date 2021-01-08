# Imports
import numpy as np
import random
from collections import namedtuple, deque
from ornstien_uhlenbeck_noise import OUNoise
import copy

# Deep Learning Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import Actor, Critic
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 2e-3              # for soft update of target parameters
LR_ACTOR = 0.003        # learning rate of the actor
LR_CRITIC = 0.003       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NOISE_CLIP = 0.5        # Max Noise
UPDATE_EVERY = 1        # LEARN EVERY N STEPS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3:

    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        # Noise used for exploration, added to action
        self.noise = OUNoise(action_size, random_seed) # Noise process

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, device)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        # Recall we will only update the local network, then perform soft update on target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network 1 (w/ Target Network)
        self.critic_1 = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target_1 = Critic(state_size, action_size, random_seed).to(device)
        # Recall we will only update the local network, then perform soft update on target
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Critic Network 2 (w/ Target Network)
        self.critic_2 = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target_2 = Critic(state_size, action_size, random_seed).to(device)
        # Recall we will only update the local network, then perform soft update on target
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor_local(state).cpu().data.numpy().flatten()

        # Add some noise for exploration
        if add_noise:
            action += self.noise.sample()
        action = action.clip(-1, 1)
        return action


    def step(self, state, action, reward, next_state, done, t_step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA, t_step)


    def learn(self, experiences, gamma, t_step):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        # Unpack Tuple
        states, actions, rewards, next_states, dones = experiences

        # Select next action according to target policy:
        noise = torch.from_numpy(self.noise.sample()).float().to(device) # (float)
        noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
        next_actions = (self.actor_target(next_states) + noise)
        next_actions = next_actions.clamp(-1, 1)

        # Compute target Q-value:
        target_Q1 = self.critic_target_1(next_states, next_actions)
        target_Q2 = self.critic_target_2(next_states, next_actions)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + ((1-dones) * gamma * target_Q).detach()

        # Optimize Critic 1:
        current_Q1 = self.critic_1(states, actions)
        loss_Q1 = F.mse_loss(current_Q1, target_Q)
        self.critic_optimizer_1.zero_grad()
        loss_Q1.backward()
        self.critic_optimizer_1.step()

        # Optimize Critic 2:
        current_Q2 = self.critic_2(states, actions)
        loss_Q2 = F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer_2.zero_grad()
        loss_Q2.backward()
        self.critic_optimizer_2.step()

        # Delayed policy updates:
        if t_step % UPDATE_EVERY == 0:
            # Compute actor loss:
            policy_actions = self.actor_local(states)
            actor_loss = -self.critic_1(states, policy_actions).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft Updates
            self.soft_update(self.critic_1, self.critic_target_1, TAU)
            self.soft_update(self.critic_2, self.critic_target_2, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data  + (1.0 - tau) * target_param.data)
    
    
    def reset(self):
        self.noise.reset()
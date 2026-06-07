"""
DQN      - Standard DQN 
DDQN     - Double DQN 
RainbowDQN - Double DQN + Dueling + NoisyNet
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from model import DuelingNet


# replay buffer
"""
s -> state
a -> action
r -> reward
ns -> next state
d  -> done?
"""
class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# base kısmı
class BaseAgent:
    GAMMA       = 0.99
    LR          = 1e-3
    BATCH_SIZE  = 128
    BUFFER_CAP  = 50_000
    UPDATE_TGT  = 500      

    def __init__(self, obs_size, n_actions, device, noisy=False):
        self.obs_size  = obs_size
        self.n_actions = n_actions
        self.device    = device
        self.noisy     = noisy

        self.online = DuelingNet(obs_size, n_actions, noisy=noisy).to(device)
        self.target = DuelingNet(obs_size, n_actions, noisy=noisy).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.LR)
        self.buffer    = ReplayBuffer(self.BUFFER_CAP)
        self.steps     = 0

    def _to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def _train_step(self, double=False):
        if len(self.buffer) < self.BATCH_SIZE:
            return

        s, a, r, ns, d = self.buffer.sample(self.BATCH_SIZE)
        s  = self._to_tensor(s)
        a  = torch.as_tensor(a, dtype=torch.long,  device=self.device)
        r  = self._to_tensor(r)
        ns = self._to_tensor(ns)
        d  = self._to_tensor(d)

        # Current Q values
        q_values = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if double:
                """
                online yani policy network asıl öğrenen yer
                backpropagation
                beyin
                """
                best_actions = self.online(ns).argmax(dim=1, keepdim=True) # action secimi online
                next_q = self.target(ns).gather(1, best_actions).squeeze(1) # actionu target ölcüyor
            else:
                next_q = self.target(ns).max(dim=1).values  # actionu target ölcüyor

            target_q = r + self.GAMMA * next_q * (1.0 - d) # gamma, (1-d) maskdir
 
        loss = nn.SmoothL1Loss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.UPDATE_TGT == 0:
            self._update_target()

    def save(self, path: str):
        torch.save(self.online.state_dict(), path)

    def load(self, path: str):
        self.online.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.target.load_state_dict(self.online.state_dict())

class DQNAgent(BaseAgent):
    def __init__(self, obs_size, n_actions, device,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995):
        super().__init__(obs_size, n_actions, device)
        self.epsilon   = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

    @property
    def epsilon_str(self):
        return f"{self.epsilon:.3f}"

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = self._to_tensor(state).unsqueeze(0)
            return int(self.online(s).argmax().item())
    # buffer pushla, train oll
    def train_step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self._train_step(double=False)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

class DDQNAgent(BaseAgent):
    def __init__(self, obs_size, n_actions, device,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995):
        super().__init__(obs_size, n_actions, device)
        self.epsilon   = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

    @property
    def epsilon_str(self):
        return f"{self.epsilon:.3f}"

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = self._to_tensor(state).unsqueeze(0)
            return int(self.online(s).argmax().item())

    def train_step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self._train_step(double=True)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


# Rainbow DQN Agent (Double DQN + Dueling + NoisyNet)
class RainbowAgent(BaseAgent):
    def __init__(self, obs_size, n_actions, device):
        super().__init__(obs_size, n_actions, device, noisy=True)

    @property
    def epsilon_str(self):
        return "Noisy"

    def select_action(self, state):
        # NoisyNet 
        self.online.sample_noise() # weightlere noise ekle, forwardda farklı ol kral?
        with torch.no_grad():
            s = self._to_tensor(state).unsqueeze(0)
            return int(self.online(s).argmax().item()) # q üret

    def train_step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self._train_step(double=True)

    def decay_epsilon(self):
        pass  # NoisyNet handles exploration

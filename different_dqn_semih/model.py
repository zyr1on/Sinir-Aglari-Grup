
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# NoisyLinear (Factorised Gaussian Noise)
# ======================================================================

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration (Rainbow DQN)."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma0       = sigma0

        # Learnable params
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not learnable)
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.register_buffer("bias_eps",   torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu,    -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma0 / math.sqrt(self.in_features))
        nn.init.uniform_(self.bias_mu,      -bound, bound)
        nn.init.constant_(self.bias_sigma,   self.sigma0 / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        device = self.weight_eps.device
        eps_i = self._scale_noise(self.in_features,  device)
        eps_j = self._scale_noise(self.out_features, device)
        self.weight_eps.copy_(eps_j.outer(eps_i))
        self.bias_eps.copy_(eps_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# ======================================================================
# Dueling Network (shared by DQN, DDQN, Rainbow)
# ======================================================================

class DuelingNet(nn.Module):
    """
    Dueling architecture:
      Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    When noisy=True, value/advantage heads use NoisyLinear.
    """

    def __init__(self, obs_size: int, n_actions: int,
                 hidden: int = 256, noisy: bool = False):
        super().__init__()
        self.noisy = noisy
        Linear = NoisyLinear if noisy else nn.Linear

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            Linear(hidden, hidden // 2),
            nn.ReLU(),
            Linear(hidden // 2, 1),
        )

        # Advantage stream
        self.adv_stream = nn.Sequential(
            Linear(hidden, hidden // 2),
            nn.ReLU(),
            Linear(hidden // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        v    = self.value_stream(feat)               # (B, 1)
        a    = self.adv_stream(feat)                 # (B, A)
        q    = v + a - a.mean(dim=1, keepdim=True)  # dueling combination
        return q

    def sample_noise(self):
        """Re-sample noise for all NoisyLinear layers."""
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()

import torch
import torch.nn as nn
import torch.distributions as D

class GaussianHead(nn.Module):
    def __init__(self, in_features: int, action_dim: int,
                 learn_log_std: bool = True, log_std_init: float = -0.5,
                 tanh_squash: bool = True, bias: bool = True):
        super().__init__()
        self.mu_layer = nn.Linear(in_features, action_dim, bias=bias)
        self.tanh_squash = tanh_squash
        if learn_log_std:
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        else:
            self.register_buffer("log_std", torch.ones(action_dim) * log_std_init)

    def forward(self, x):
        mu = self.mu_layer(x)
        if self.tanh_squash:
            mu = torch.tanh(mu)
        std = torch.exp(self.log_std)
        return mu, std

    def dist(self, x):
        mu, std = self.forward(x)
        return D.Normal(mu, std)

    def sample(self, x, deterministic=False):
        dist = self.dist(x)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy  = dist.entropy().sum(-1).mean()
        return action, log_prob, entropy

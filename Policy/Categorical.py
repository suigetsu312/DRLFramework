import torch
import torch.nn as nn
import torch.distributions as D

class CategoricalHead(nn.Module):
    def __init__(self, in_features: int, num_actions: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, num_actions, bias=bias)

    def forward(self, x):
        return self.linear(x)  # logits

    def dist(self, x):
        return D.Categorical(logits=self.forward(x))

    def sample(self, x, deterministic=False):
        dist = self.dist(x)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action, log_prob, entropy

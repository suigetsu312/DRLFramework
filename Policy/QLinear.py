import torch.nn as nn

class QLinearHead(nn.Module):
    def __init__(self, in_features: int, out_actions: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_actions, bias=bias)
    def forward(self, x):
        return self.linear(x)  # [B, num_actions]

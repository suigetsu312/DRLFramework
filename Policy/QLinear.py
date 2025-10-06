class QLinearHead(nn.Module):
    def __init__(self, in_dim: int, out_actions: int, bias=True, init=None, bias_init=None):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_actions, bias=bias)
        _init(self.linear, init, bias_init)
    def forward(self, feat): return self.linear(feat)

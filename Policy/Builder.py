# Policy/Builder.py
import math
import torch
import torch.nn as nn

def _act(name: str, params=None):
    name = name.lower()
    params = params or {}
    if name == "relu": return nn.ReLU()
    if name == "leakyrelu": return nn.LeakyReLU(**params)
    if name == "tanh": return nn.Tanh()
    if name == "sigmoid": return nn.Sigmoid()
    if name == "gelu": return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")

def _init(m: nn.Module, init_cfg: dict | None, bias_cfg: dict | None):
    if init_cfg is None: return
    scheme = str(init_cfg.get("scheme", "xavier_uniform")).lower()
    gain = float(init_cfg.get("gain", 1.0))
    nonlin = init_cfg.get("nonlinearity", "relu")
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if   scheme == "xavier_uniform": nn.init.xavier_uniform_(m.weight, gain=gain)
        elif scheme == "xavier_normal":  nn.init.xavier_normal_(m.weight,  gain=gain)
        elif scheme == "kaiming_uniform":nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlin)
        elif scheme == "kaiming_normal": nn.init.kaiming_normal_(m.weight,  nonlinearity=nonlin)
        else: raise ValueError(f"Unknown init scheme: {scheme}")
        if m.bias is not None and bias_cfg and "value" in bias_cfg:
            nn.init.constant_(m.bias, bias_cfg["value"])

class QLinearHead(nn.Module):
    def __init__(self, in_dim: int, out_actions: int, bias=True, init=None, bias_init=None):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_actions, bias=bias)
        _init(self.linear, init, bias_init)
    def forward(self, feat): return self.linear(feat)

def build_model_from_config(model_cfg: dict, obs_space, action_space) -> nn.Module:
    obs_format = model_cfg.get("obs_format", "vector")
    layers = model_cfg["layers"]
    ginit = model_cfg.get("init", {})
    default_init = ginit.get("default")
    bias_init = ginit.get("bias")

    seq: list[nn.Module] = []
    in_dim = None
    c = h = w = None

    if obs_format == "vector":
        in_dim = int(obs_space.shape[0])
    elif obs_format == "image":
        c, h, w = map(int, obs_space.shape)
    else:
        raise ValueError("Model.obs_format must be 'vector' or 'image'")

    for layer in layers:
        t = str(layer["type"]).lower()
        if t == "linear":
            out_f = int(layer["out_features"])
            bias = bool(layer.get("bias", True))
            mod = nn.Linear(in_dim, out_f, bias=bias)
            _init(mod, layer.get("init", default_init), bias_init)
            seq.append(mod)
            in_dim = out_f
        elif t == "conv2d":
            oc = int(layer["out_channels"])
            k = layer.get("kernel", 3); s = layer.get("stride", 1); p = layer.get("padding", 0)
            bias = bool(layer.get("bias", True))
            mod = nn.Conv2d(c, oc, k, s, p, bias=bias)
            _init(mod, layer.get("init", default_init), bias_init)
            seq.append(mod)
            def _dim(L, k, s, p): return math.floor((L + 2*p - k)/s + 1)
            h, w = _dim(h,k,s,p), _dim(w,k,s,p)
            c = oc
        elif t == "flatten":
            seq.append(nn.Flatten())
            if obs_format == "image": in_dim = c * h * w
        elif t == "activation":
            seq.append(_act(layer["name"], layer.get("params")))
        elif t == "batchnorm1d":
            seq.append(nn.BatchNorm1d(in_dim))
        elif t == "batchnorm2d":
            seq.append(nn.BatchNorm2d(c))
        elif t == "dropout":
            seq.append(nn.Dropout(layer.get("p", 0.5)))
        else:
            raise ValueError(f"Unsupported layer type: {layer['type']}")

    backbone = nn.Sequential(*seq)

    # Head
    head_cfg = model_cfg["head"]
    htype = head_cfg["type"].lower()
    hparams = head_cfg.get("params", {})
    if htype != "qlinear":
        raise ValueError(f"DQN only supports head.type='QLinear', got {head_cfg['type']}")
    out_actions = hparams.get("out_actions", "auto")
    if out_actions == "auto": out_actions = int(action_space.n)
    head = QLinearHead(in_dim, out_actions,
                       bias=hparams.get("bias", True),
                       init=hparams.get("init"),
                       bias_init=bias_init)

    return nn.Sequential(backbone, head)

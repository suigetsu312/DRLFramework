# Policy/Builder.py
import torch.nn as nn
from gymnasium import spaces
from .QLinear import QLinearHead
from .Categorical import CategoricalHead
from .Gaussian import GaussianHead

HEADS = {
    "QLINEAR": QLinearHead,
    "CATEGORICAL": CategoricalHead,
    "GAUSSIAN": GaussianHead,
}

def _act(name):
    name = name.lower()
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}.get(name, nn.ReLU)()

def _infer_action_dim(action_space):
    if isinstance(action_space, spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, spaces.Box):
        return int(action_space.shape[0])
    else:
        raise ValueError(f"Unsupported action space: {type(action_space)}")

def _apply_init_linear(module: nn.Module, init_cfg: dict | None, bias_cfg: dict | None = None):
    if init_cfg is None: return
    scheme = str(init_cfg.get("scheme", "xavier_uniform")).lower()
    gain = float(init_cfg.get("gain", 1.0))
    nonlin = init_cfg.get("nonlinearity", "relu")
    if isinstance(module, nn.Linear):
        if   scheme == "xavier_uniform": nn.init.xavier_uniform_(module.weight, gain=gain)
        elif scheme == "xavier_normal":  nn.init.xavier_normal_(module.weight,  gain=gain)
        elif scheme == "kaiming_uniform":nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlin)
        elif scheme == "kaiming_normal": nn.init.kaiming_normal_(module.weight,  nonlinearity=nonlin)
        else: raise ValueError(f"Unknown init scheme: {scheme}")
        if module.bias is not None and bias_cfg and "value" in bias_cfg:
            nn.init.constant_(module.bias, bias_cfg["value"])

def build_model_from_config(model_cfg: dict, obs_space, action_space) -> nn.Module:
    obs_format = model_cfg.get("obs_format", "vector").lower()
    if obs_format != "vector":
        raise NotImplementedError("Only vector obs in this minimal builder")

    in_dim = int(obs_space.shape[0])
    seq = []
    for layer in model_cfg["layers"]:
        t = layer["type"].lower()
        if t == "linear":
            out_f = int(layer["out_features"]); bias = bool(layer.get("bias", True))
            lin = nn.Linear(in_dim, out_f, bias=bias)
            # 可選：層級初始化（若你有 global init 想支援）
            if "init" in layer:
                _apply_init_linear(lin, layer["init"], model_cfg.get("init", {}).get("bias"))
            seq.append(lin)
            in_dim = out_f
        elif t == "activation":
            seq.append(_act(layer["name"]))
        else:
            raise ValueError(f"Unsupported layer: {t}")
    backbone = nn.Sequential(*seq)

    # --- head ---
    head_cfg = model_cfg["head"]
    htype = head_cfg["type"].strip().upper()
    params = dict(head_cfg.get("params", {}))  # 拷貝
    head_init = params.pop("init", None)       # ← 吃掉 init，不傳入 head.__init__

    # 自動補 action 維度
    adim = _infer_action_dim(action_space)
    if params.get("out_actions") == "auto": params["out_actions"] = adim
    if params.get("num_actions") == "auto": params["num_actions"] = adim
    if params.get("action_dim") == "auto":  params["action_dim"]  = adim

    Head = HEADS[htype]
    head = Head(in_dim, **params)

    # 對不同 head 套初始化
    bias_cfg = model_cfg.get("init", {}).get("bias")
    if htype == "QLINEAR":
        _apply_init_linear(head.linear, head_init, bias_cfg)
    elif htype == "CATEGORICAL":
        _apply_init_linear(head.linear, head_init, bias_cfg)
    elif htype == "GAUSSIAN":
        _apply_init_linear(head.mu_layer, head_init, bias_cfg)
        # log_std 通常不用特別初始化，保留預設

    return nn.Sequential(backbone, head)

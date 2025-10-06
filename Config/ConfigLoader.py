# config_loader.py
import yaml

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 必要鍵健檢（早炸）
    for k in ("Agent", "Env", "Model", "DRL"):
        if k not in cfg:
            raise KeyError(f"Config missing top-level key: {k}")

    # Optimizer lr 若為 null，用 DRL.Params.learning_rate 回填
    drl = cfg["DRL"]
    params = drl.get("Params", {})
    opt = drl.get("Optimizer", {"params": {}})
    opt.setdefault("params", {})
    if opt["params"].get("lr", None) is None:
        lr = params.get("learning_rate", 5e-4)
        opt["params"]["lr"] = lr
    cfg["DRL"]["Optimizer"] = opt

    return cfg

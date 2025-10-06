# loggers/tensorboard_logger.py
from torch.utils.tensorboard import SummaryWriter
import numpy as np, torch
from typing import Dict, Any, Union

class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, name: str, value: Union[int, float], global_step: int):
        if value is None: return
        if isinstance(value, torch.Tensor):
            value = value.detach().float().item() if value.numel()==1 else float(value.mean().item())
        self.writer.add_scalar(name, float(value), global_step=global_step)

    def log_histogram(self, name: str, values: Union[np.ndarray, list, tuple, torch.Tensor], global_step: int):
        if values is None: return
        if isinstance(values, torch.Tensor): values = values.detach().cpu().numpy()
        arr = np.asarray(values)
        if arr.size == 0: return
        if not np.isfinite(arr).all(): arr = np.nan_to_num(arr, copy=False)
        self.writer.add_histogram(name, arr, global_step=global_step)

    def _flatten(self, prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
        """把巢狀 dict 攤平成 'a/b/c': value 的扁平字典"""
        flat = {}
        def rec(p, obj):
            if obj is None: return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    rec(f"{p}/{k}" if p else str(k), v)
            else:
                flat[p] = obj
        rec(prefix, d)
        return flat

    def log_dict(self, prefix: str, data_dict: Dict[str, Any], global_step: int):
        if not data_dict: return
        for name, value in self._flatten(prefix, data_dict).items():
            # 選擇性地把較長向量記成 histogram，其餘當 scalar
            if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
                arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                if arr.ndim == 0 or arr.size <= 4:
                    self.log_scalar(name, float(arr.mean()), global_step)
                else:
                    self.log_histogram(name, arr, global_step)
            elif isinstance(value, (int, float)) or (isinstance(value, np.generic)):
                self.log_scalar(name, float(value), global_step)
            # 其他型別略過或自訂處理

    def close(self):
        self.writer.flush()
        self.writer.close()

# csv_logger.py
import csv, os, datetime
from typing import Dict, Any

class CSVLogger:
    def __init__(self, prefix: str, fields: list[str], save_dir: str = "./logs"):
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.path = os.path.join(save_dir, f"{prefix}_{ts}.csv")
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()

    def log(self, row: Dict[str, Any]):
        self.w.writerow({k: row.get(k, "") for k in self.w.fieldnames})

    def close(self):
        self.f.flush(); self.f.close()

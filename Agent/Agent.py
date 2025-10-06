# rl/agent/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class Transition:
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    info: Dict = field(default_factory=dict)

class AgentBase(ABC):
    @abstractmethod
    def act(self, state: Any, step: int, deterministic: bool = False) -> Any: ...
    @abstractmethod
    def record(self, t: Transition) -> None: ...
    @abstractmethod
    def learn_if_ready(self, step: int) -> Dict[str, float]: ...

    def on_episode_start(self, episode: int) -> None: pass
    def on_episode_end(self, episode: int, episode_return: float, steps: int) -> None: pass
    def should_terminate(self, steps_in_ep: int) -> bool: return False

    @abstractmethod
    def save(self, path: str, step: Optional[int] = None, episode: Optional[int] = None) -> None: ...
    @abstractmethod
    def load(self, path: str) -> None: ...
    def train_mode(self) -> None: pass
    def eval_mode(self) -> None: pass

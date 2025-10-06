# Agent/registry.py
from typing import Dict, Type
from Agent.Agent import AgentBase

_AGENT_REGISTRY: Dict[str, Type[AgentBase]] = {}


def register_agent(name: str):
    """
    用 decorator 註冊一個 Agent 類別到全域 registry。
    用法:
        @register_agent("DQN")
        class DQNAgent(AgentBase): ...
    """
    def decorator(cls):
        name_upper = name.strip().upper()
        if name_upper in _AGENT_REGISTRY:
            raise ValueError(f"Agent '{name_upper}' already registered.")
        _AGENT_REGISTRY[name_upper] = cls
        return cls
    return decorator


def get_agent_class(name: str) -> Type[AgentBase]:
    """根據名稱取得 Agent 類別"""
    name_upper = name.strip().upper()
    if name_upper not in _AGENT_REGISTRY:
        raise KeyError(f"Agent '{name_upper}' not found in registry.")
    return _AGENT_REGISTRY[name_upper]


def list_registered_agents():
    return list(_AGENT_REGISTRY.keys())

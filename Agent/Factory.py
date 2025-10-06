# Agent/Factory.py
import Agent
from Agent.Registry import get_agent_class

class AgentFactory:
    @staticmethod
    def from_config(cfg, obs_space, action_space):
        """
        從 config 中自動生成對應的 Agent 實例。
        cfg:
          Agent:
            type: DQN
        """
        agent_type = cfg.get("Agent", {}).get("type", None)
        if not agent_type:
            raise ValueError("Config must contain Agent.type (e.g. 'DQN', 'PPO')")

        agent_cls = get_agent_class(agent_type)
        return agent_cls(cfg, obs_space, action_space)

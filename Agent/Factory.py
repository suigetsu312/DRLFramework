from typing import Dict
from .DQN import DQNAgent

class AgentFactory:
    @staticmethod
    def create(config: Dict, state_shape, action_shape):
        config = config["DLParameter"]
        dl_type = config["MethodParameter"]["type"].lower()
        config["MethodParameter"]["DQN"]["observation_space"] = state_shape
        config["MethodParameter"]["DQN"]["action_space"] = action_shape

        if dl_type == "drl":
            return AgentFactory.__createDRLAgent(config)
        else:
            raise ValueError(f"Unsupported DL method: {dl_type}")
    @staticmethod
    def __createDRLAgent(config: Dict):
        method = config["MethodParameter"]["method"].lower()

        if method == "dqn":
            return DQNAgent(config)
        elif method == "ppo":
            pass
        else:
            raise ValueError(f"Unsupported DRL method: {method}")

if __name__ == "__main__":
    with open("./networkConfig/example.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    print(config)

    agent = AgentFactory.create(config)

    if isinstance(agent, DQNAgent):
        print(agent.QNet)
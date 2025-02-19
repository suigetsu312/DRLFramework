
from ConfigParser import ModelFactory, LossFunctionFactory, OptimizerFactory, TrainingFactory
from Memory import ReplayBuffer
import copy
import torch
import numpy as np
from typing import Dict
import yaml
# ==============================
# 1. DRL Method Factory
# ==============================
class AgentFactory:
    @staticmethod
    def create(config: Dict):
        config = config["DLParameter"]
        dl_type = config["MethodParameter"]["type"].lower()

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

class AgentBase:
    def __init__(self):
        pass
    @torch.no_grad()
    def choose_action(self, state):
        pass
    def update_epsilon(self, step):
        pass
    def train_step(self, step):
        pass

class DQNAgent(AgentBase):
    def __init__(self, config):
        self.config = config
        self.gamma = config["MethodParameter"]["DQN"]["gamma"]
        self.epsilon_init = config["MethodParameter"]["DQN"]["exploration_initial_eps"]
        self.epsilon = config["MethodParameter"]["DQN"]["exploration_initial_eps"]
        self.epsilon_min = config["MethodParameter"]["DQN"]["exploration_final_eps"]
        self.replay_buffer_size = config["MethodParameter"]["DQN"]["replay_buffer_size"]
        self.epsilon_decay = config["MethodParameter"]["DQN"]["exploration_fraction"]
        self.action_space = config["MethodParameter"]["DQN"]["action_space"]
        self.observation_space = config["MethodParameter"]["DQN"]["observation_space"]
        self.reward_shape = config["MethodParameter"]["DQN"]["reward_shape"]    
        self.target_update_freq = config["MethodParameter"]["DQN"]["target_update_freq"]
        self.replay_buffer = ReplayBuffer(  self.action_space,
                                            self.observation_space,
                                            self.reward_shape,
                                            self.replay_buffer_size)
        
        self.qnet = ModelFactory(config["Model"]["qnet"],
                                 self.observation_space, 
                                self.action_space).create_model()
        self.targetNet = copy.deepcopy(self.qnet)

        self.targetNet.to(self.device)
        self.qnet.to(self.device)

        self.optimizer = OptimizerFactory.create(self.qnet, config["Optimizer"]["qnet"])
        self.training_params = TrainingFactory.create(config)
        self.device = self.training_params["device"]
        self.timestep = self.training_params["timestep"]
        self.batch_size = self.training_params["batch_size"]
        self.loss = LossFunctionFactory.create(self.training_params["loss_function"])


    @torch.no_grad()
    def choose_action(self, state):
        if np.random.uniform(0,1) < 1 - self.epsilon:
            state = torch.tensor(state).to(self.device)
            action = torch.argmax(self.qnet(state)).item()
        else:
            action = np.random.choice(self.action_space)
        return action 

    def update_epsilon(self, step):
        self.epsilon = max(self.epsilon_min, self.epsilon_init - (step/  self.timestep) * (self.epsilon_init - self.epsilon_min))

    def train_step(self, step, state, action, reward, next_state, done):

        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        q_values = self.qnet(states) 
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1) 

        with torch.no_grad():
            next_q_values = self.targetNet(next_states).max(1)[0]  
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        loss = self.loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % self.target_update_freq == 0:
            self.targetNet.load_state_dict(self.qnet.state_dict())
if __name__ == "__main__":
    with open("./networkConfig/example.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    print(config)

    agent = AgentFactory.create(config)

    if isinstance(agent, DQNAgent):
        print(agent.qnet)
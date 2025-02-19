import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict
import copy
from Memory import ReplayBuffer
# ==============================
# 1. DRL Method Factory
# ==============================
class MethodFactory:
    @staticmethod
    def create(config: Dict):
        config = config["DLParameter"]
        dl_type = config["MethodParameter"]["type"].lower()

        if dl_type == "drl":
            return MethodFactory.__createDRLAgent(config)
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
# ==============================
# 2. Model Factory
# ==============================

class ActivationFactory:    
    @staticmethod
    def create(config: str):
        activation = config["type"]
        if activation == "ReLU":
            return nn.ReLU()
        elif activation == "Sigmoid":
            return nn.Sigmoid()
        elif activation == "Tanh":
            return nn.Tanh()
        elif activation == "LeakyReLU":
            return nn.LeakyReLU(config.get("negative_slope", 0.01)) 
        else:
            return nn.Identity()

class CNNLayer(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride, 
                 padding,
                 activation="ReLU"):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = ActivationFactory.create(activation)

    def forward(self, x):
        return self.activation(self.conv(x))

class MLP(nn.Module):
    def __init__(self, in_features, out_features, activation="ReLU"):
        super(MLP, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = ActivationFactory.create(activation)
    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x

class ModelFactory:
    def __init__(self, config):
        self.config = config

    def create_model(self):
        layers = []
        for layer_config in self.config["layers"]:
            layer_type = layer_config["type"]
            
            if layer_type == "Conv2d":
                layers.append(CNNLayer(
                    in_channels=layer_config["in_channels"],
                    out_channels=layer_config["out_channels"],
                    kernel_size=layer_config["kernel_size"],
                    stride=layer_config["stride"],
                    padding=layer_config["padding"],
                    activation=layer_config["activation"]
                ))
            elif layer_type == "MaxPool2d":
                layers.append(nn.MaxPool2d(
                    kernel_size=layer_config["kernel_size"],
                    stride=layer_config["stride"],
                    padding=layer_config["padding"]
                ))
            elif layer_type == "averagePool2d":
                layers.append(nn.AvgPool2d(
                    kernel_size=layer_config["kernel_size"],
                    stride=layer_config["stride"],
                    padding=layer_config["padding"]
                ))
            elif layer_type == "Flatten":
                layers.append(nn.Flatten())
            elif layer_type == "Linear":
                layers.append(MLP(
                    in_features=layer_config["in_features"],
                    out_features=layer_config["out_features"],
                    activation=layer_config["activation"]
                ))

        return nn.Sequential(*layers)
# ==============================
# 3. Optimizer Factory
# ==============================
class OptimizerFactory:
    @staticmethod
    def create(model, config: Dict):
        optimizer_type = config["type"].lower()
        lr = config["lr"]
        weight_decay = float(config["weight_decay"])
        betas = tuple(config["betas"])
        eps = float(config["eps"])

        if optimizer_type == "adam":
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas,eps=eps)
        elif optimizer_type == "sgd":
            return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

# ==============================
# 4. Training Factory
# ==============================
class TrainingFactory:
    @staticmethod
    def create(config: Dict):
        return {
            "batch_size": config["Training"]["batch_size"],
            "timestemp": config["Training"]["timestemp"],
            "loss_function": config["Training"]["loss_function"],
            "device": config["Training"]["device"]
        }


class LossFunctionFactory:
    @staticmethod
    def create(methodName: str):
        if methodName == "MSELoss":
            return F.mse_loss  # 返回函數，而不是執行它
        raise ValueError(f"Unsupported loss function: {methodName}")
    
# ==============================
# 5. DQN Agent
# ==============================
class DQNAgent:
    def __init__(self, config):
        # config
        self.config = config

        # model and target network
        self.qnet = ModelFactory(config["Model"]["qnet"]).create_model()
        self.targetNet = copy.deepcopy(self.qnet)

        # optimizer and training parameters
        self.optimizers = OptimizerFactory.create(self.qnet, config["Optimizer"]["qnet"])
        self.training_params = TrainingFactory.create(config)
        self.device = self.training_params["device"]
        self.timestemp = self.training_params["timestemp"]
        self.loss = LossFunctionFactory.create(self.training_params["loss_function"])
        # dqn agent parameters
        self.gamma = config["MethodParameter"]["DQN"]["gamma"]
        self.epsilon_init = config["MethodParameter"]["DQN"]["exploration_initial_eps"]
        self.epsilon = config["MethodParameter"]["DQN"]["exploration_initial_eps"]
        self.epsilon_min = config["MethodParameter"]["DQN"]["exploration_final_eps"]
        self.replay_buffer_size = config["MethodParameter"]["DQN"]["replay_buffer_size"]
        self.epsilon_decay = config["MethodParameter"]["DQN"]["exploration_fraction"]
        self.action_space = config["MethodParameter"]["DQN"]["action_space"]
        self.observation_space = config["MethodParameter"]["DQN"]["observation_space"]
        self.reward_shape = config["MethodParameter"]["DQN"]["reward_shape"]    
        self.termination = config["MethodParameter"]["DQN"]["termination_step"]
        self.target_update_freq = config["MethodParameter"]["DQN"]["target_update_freq"]
        self.replay_buffer = ReplayBuffer(  self.action_space,
                                            self.observation_space,
                                            self.reward_shape,
                                            self.replay_buffer_size)
        self.targetNet.to(self.device)
        self.qnet.to(self.device)

    @torch.no_grad()
    def choose_action(self, state):
        if np.random.uniform(0,1) < 1 - self.epsilon:
            state = torch.tensor(state)
            action = torch.argmax(self.onlineNet(state)).item()
        else:
            action = np.random.choice(self.action_space)
        return action 

    def update_epsilon(self, step):
        self.epsilon = max(self.epsilon_min, self.epsilon_init - (step/  self.timestemp) * (self.epsilon_init - self.epsilon_min))

    def train_step(self, state, step):

        """執行 DQN 訓練步驟"""

        # 確保 ReplayBuffer 有足夠的數據
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 從 replay buffer 取樣
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 轉換成 tensor
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # 計算當前 Q 值
        q_values = self.onlineNet(states)  # shape: (batch_size, num_actions)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # 只取 action 對應的 Q 值

        # 計算目標 Q 值
        with torch.no_grad():
            next_q_values = self.targetNet(next_states).max(1)[0]  # 取最大 Q 值 (DQN 更新方式)
            target_q_values = rewards + self.gamma * next_q_values * (~dones)  # 如果 done=True，則不累積未來獎勵

        # 計算損失函數 (MSE loss)
        loss = self.loss(q_values, target_q_values)

        # 反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目標網路 (Target Network)
        if step % self.target_update_freq == 0:
            self.targetNet.load_state_dict(self.onlineNet.state_dict())

                
             

# ==============================
# 6. 測試 Factory
# ==============================
if __name__ == "__main__":
    with open("./networkConfig/example.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    print(config)

    agent = MethodFactory.create(config)  # 創建 DQN Agent
    print(agent.qnet)  # 印出模型架構

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        for layer_config in self.config["Model"]["layers"]:
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
        optimizer_type = config["Optimizer"]["type"].lower()
        lr = config["Optimizer"]["lr"]
        weight_decay = float(config["Optimizer"]["weight_decay"])
        betas = tuple(config["Optimizer"]["betas"])
        eps = float(config["Optimizer"]["eps"])

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
            "epochs": config["Training"]["epochs"],
            "loss_function": config["Training"]["loss_function"],
            "device": config["Training"]["device"]
        }

# ==============================
# 5. DQN Agent
# ==============================
class DQNAgent:
    def __init__(self, config):
        # config
        self.config = config

        # model and target network
        self.targetNet = ModelFactory(config).create_model()
        self.onlineNet = copy.deepcopy(self.targetNet)

        # optimizer and training parameters
        self.optimizer = OptimizerFactory.create(self.model, config)
        self.training_params = TrainingFactory.create(config)

        # dqn agent parameters
        self.gamma = config["MethodParameter"]["DQN"]["gamma"]
        self.epsilon = config["MethodParameter"]["DQN"]["exploration_initial_eps"]
        self.epsilon_min = config["MethodParameter"]["DQN"]["exploration_final_eps"]
        self.epsilon_decay = config["MethodParameter"]["DQN"]["exploration_fraction"]
        self.action_space = config["MethodParameter"]["DQN"]["action_space"]
        self.observation_space = config["MethodParameter"]["DQN"]["observation_space"]
        self.replay_buffer = ReplayBuffer() # 簡單示意，實際應該用 ring buffer

    @torch.no_grad()
    def choose_action(self, state):
        pass
    def update_epsilon(self, step, total_steps):
        pass
    def train_step(self, batch):
        pass


# ==============================
# 6. 測試 Factory
# ==============================
if __name__ == "__main__":
    with open("./networkConfig/example.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    print(config)

    agent = MethodFactory.create(config)  # 創建 DQN Agent
    print(agent.model)  # 印出模型架構

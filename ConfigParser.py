import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict
import copy
from Memory import ReplayBuffer
from Model import CNNLayer, MLP

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
class ModelFactory:
    def __init__(self, config, observation_space, action_space):
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space

    def channelShape(self, config):
        if config == "StateShape":
            return self.observation_space
        elif config == "ActionShape":
            return self.action_space
        elif isinstance(config, int):
            return config
        else:
            assert "必須為整數或action state space"

    def create_model(self):
        layers = []
        for layer_config in self.config["layers"]:
            layer_type = layer_config["type"]
            
            if layer_type == "Conv2d":
                layers.append(CNNLayer(
                    in_channels=self.channelShape(layer_config["in_features"]),
                    out_channels=self.channelShape(layer_config["out_features"]),
                    kernel_size=layer_config["kernel_size"],
                    stride=layer_config["stride"],
                    padding=layer_config["padding"],
                    activation=ActivationFactory.create(layer_config["activation"])
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
                    in_channels=self.channelShape(layer_config["in_features"]),
                    out_channels=self.channelShape(layer_config["out_features"]),
                    activation=ActivationFactory.create(layer_config["activation"])
                ))

        return nn.Sequential(*layers)
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
class TrainingFactory:
    @staticmethod
    def create(config: Dict):
        return {
            "batch_size": config["Training"]["batch_size"],
            "timestep": config["Training"]["timestep"],
            "loss_function": config["Training"]["loss_function"],
            "device": config["Training"]["device"]
        }
class LossFunctionFactory:
    @staticmethod
    def create(methodName: str):
        if methodName == "MSELoss":
            return F.mse_loss  # 返回函數，而不是執行它
        raise ValueError(f"Unsupported loss function: {methodName}")
    
if __name__ == "__main__":
    with open("./networkConfig/example.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    print(config)
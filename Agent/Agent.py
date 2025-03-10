
import torch
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


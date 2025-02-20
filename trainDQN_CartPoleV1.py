from Agent import AgentFactory, DQNAgent
from Trainer import DRLTrainer
import yaml
import gymnasium as gym
import torch
class CustomCartPoleEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = gym.make("CartPole-v1", render_mode = "human")  # 使用預設的 CartPole 環境

    def step(self, action):
        next_state, _, done, truncated, info = self.env.step(action)

        # # 自訂 Reward：這裡根據杆的角度來設置 reward
        # x, x_dot, theta, theta_dot = next_state
        # reward = 1.0 - abs(theta)  # 讓 reward 基於平衡杆的角度來變動

        # 可以加強懲罰機制，若 done 為 True 時，給予懲罰
        if done:
            reward -= 10  # 結束時懲罰

        return next_state, reward, done, truncated, info

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()        

# 建立並使用自訂環境
env = CustomCartPoleEnv()
if __name__ == "__main__":
    with open("./networkConfig/example.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 創建 CartPole 環境
    env = gym.make("CartPole-v1", render_mode='human')

    # 更新 config 以適配環境的 observation_space 和 action_space
    config["DLParameter"]["MethodParameter"]["DQN"]["observation_space"] = env.observation_space.shape[0]
    config["DLParameter"]["MethodParameter"]["DQN"]["action_space"] = int(env.action_space.n)

    agent: DQNAgent = AgentFactory.create(config)

    trainer = DRLTrainer(env, agent)
    trainer.fit()  # 開始訓練

    trainer.save("./results/dqn_cartpole_full.pth")  # 儲存模型

    checkpoint = torch.load("./results/dqn_cartpole_full.pth")
    agent.qnet.load_state_dict(checkpoint["qnet"])

    # 測試模型

    state = env.reset()
    done = False
    step = 0
    acccumulated = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        acccumulated += reward
        step += 1
        env.render()

        if step > 500:
            break   
    
    print(f"Test done, accumulated reward: {acccumulated}")
        
    env.close()
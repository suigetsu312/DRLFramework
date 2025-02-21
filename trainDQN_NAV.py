from Agent import AgentFactory, DQNAgent
from Trainer import DRLTrainer
import yaml
import gymnasium as gym
import torch
import os 
from CustomEnv.NAV import NAV

if __name__ == "__main__":
    with open("./networkConfig/example.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    env = NAV()
    config["DLParameter"]["MethodParameter"]["DQN"]["observation_space"] = 5
    config["DLParameter"]["MethodParameter"]["DQN"]["action_space"] = 4

    agent: DQNAgent = AgentFactory.create(config)
    print(agent.qnet)
    trainer = DRLTrainer(env, agent)
    trainer.fit()

    trainer.save("dqn_nav_full_1.pth")

    checkpoint = torch.load(os.path.join(agent.save_path, "dqn_nav_full_1.pth"))
    agent.qnet.load_state_dict(checkpoint["qnet"])


    state, _ = env.reset()
    done = False
    step = 0
    accumulated = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        accumulated += reward
        step += 1
        env.render()

        if step > 500:
            break   
    
    print(f"Test done, accumulated reward: {accumulated}")
        
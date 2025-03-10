from Agent.Factory import AgentFactory
from Agent.DQN import DQNAgent
from Trainer import DRLTrainer
import yaml
import gymnasium as gym
import torch
import os 
from CustomEnv.NAV import NAV
from test.Inference import DRLInference

if __name__ == "__main__":
    with open("./networkConfig/NAV_OBS.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    env = NAV()
    agent: DQNAgent = AgentFactory.create(config, env.observation_space, env.action_space)
    print(agent.QNet)
    agent.QNet.train()

    trainer = DRLTrainer(env, agent)
    trainer.fit()
    trainer.save("dqn_nav_full_2.pth")

    checkpoint = torch.load(os.path.join(agent.save_path, "dqn_nav_full_2.pth"))
    agent.QNet.load_state_dict(checkpoint["qnet"])
    agent.QNet.eval()

    state, _ = env.reset()
    count = 0
    infRunner = DRLInference(agent, env)
    while True:
        acc, step = infRunner.run()
        count += 1
        print(f"epoch {count} : {acc} , {step}")
        if count > 100:
            break
        
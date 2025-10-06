import yaml, gymnasium as gym
from Agent.Factory import AgentFactory
from Trainer import RLTrainer
from Logger.TensorboardLogger import TensorBoardLogger

if __name__ == "__main__":
    cfg = yaml.safe_load(open("Config/dqn_cartpole.yaml"))
    env = gym.make(cfg["Env"]["id"])
    obs_space, act_space = env.observation_space, env.action_space

    # ✅ 自動從 config 建立 agent
    agent = AgentFactory.from_config(cfg, obs_space, act_space)

    logger = TensorBoardLogger(log_dir=f"runs/{cfg['Logging']['run_name']}")
    trainer = RLTrainer(env, agent, max_steps=100_000, log_freq=cfg["Logging"]["log_interval"], logger=logger)
    trainer.fit()

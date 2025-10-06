# run_dqn_inference.py
import yaml
import gymnasium as gym
from Agent.DQN import DQNAgent
from Inference.DRLInference import DRLInference
from Agent.Factory import AgentFactory

if __name__ == "__main__":
    # 讀取設定
    cfg = yaml.safe_load(open("Config/dqn_cartpole.yaml"))

    # 建立可視化環境（一定要加 render_mode="human"）
    env = gym.make(cfg["Env"]["id"], render_mode="human")
    obs_space, act_space = env.observation_space, env.action_space
    # 建 agent 並載入權重
    # ✅ 自動從 config 建立 agent
    agent = AgentFactory.from_config(cfg, obs_space, act_space)
    agent.load("checkpoints/run/ckpt_step100000.pt")

    # 建推論器，開啟動畫即可
    infer = DRLInference(env, agent, render_mode="human", deterministic=True)

    # 跑幾回合看動畫
    metrics = infer.run(episodes=5)

    print("\nInference finished.")
    for m in metrics:
        print(f"Episode {m['episode']}: return={m['return']:.2f}, steps={m['length']}")

    env.close()

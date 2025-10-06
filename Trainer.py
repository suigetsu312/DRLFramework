# Trainer.py
import os
from typing import Optional, Dict, Any
from Agent.Agent import AgentBase, Transition
from Logger.TensorboardLogger import TensorBoardLogger


class RLTrainer:
    def __init__(self, env, agent: AgentBase, max_steps: int,
                 log_freq: int = 1000, logger: Optional[TensorBoardLogger] = None):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.log_freq = max(1, int(log_freq))
        self.logger = logger or TensorBoardLogger(log_dir="./dummy_logs")

        # 若是 dummy logger，避免寫檔
        if logger is None:
            print("Warning: No logger provided. Skipping logging to disk.")

        # 從 config 擷取 save 設定（若存在）
        self.save_interval = None
        self.ckpt_dir = None
        if hasattr(agent, "cfg"):
            log_cfg = agent.cfg.get("Logging", {})
            self.save_interval = log_cfg.get("save_interval", 0)
            self.ckpt_dir = log_cfg.get("ckpt_dir", "checkpoints/run")
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def fit(self):
        state, _ = self.env.reset()
        episode, ep_return, ep_len = 0, 0.0, 0

        for env_step in range(1, self.max_steps + 1):
            # === act ===
            action, agent_meta = self.agent.act(state, env_step, deterministic=False)

            # === env step ===
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated or self.agent.should_terminate(ep_len + 1))

            # === record transition ===
            self.agent.record(Transition(state, action, reward, next_state, done, info or {}))
            ep_return += float(reward)
            ep_len += 1

            # === learn ===
            learn_logs = self.agent.learn_if_ready(env_step)

            # === logging ===
            if env_step % self.log_freq == 0:
                env_info = {
                    "raw_reward": (info or {}).get("raw_reward"),
                    "shaped_reward": (info or {}).get("shaped_reward"),
                    "action": action,
                }
                self.logger.log_dict("env", env_info, global_step=env_step)
                self.logger.log_dict("reward", (info or {}).get("reward_terms", {}) or {}, global_step=env_step)
                self.logger.log_dict("obs", (info or {}).get("obs_dict", {}) or {}, global_step=env_step)
                self.logger.log_dict("agent", agent_meta or {}, global_step=env_step)
                self.logger.log_dict("learn", learn_logs or {}, global_step=env_step)

            # === checkpoint ===
            if self.save_interval and env_step % self.save_interval == 0:
                try:
                    self.agent.save(self.ckpt_dir, step=env_step)
                    print(f"[Trainer] Saved checkpoint at step {env_step}")
                except Exception as e:
                    print(f"[Trainer] Warning: failed to save checkpoint at step {env_step}: {e}")

            # === episode end ===
            if done:
                self.agent.on_episode_end(episode + 1, ep_return, ep_len)
                self.logger.log_scalar("episode/return", ep_return, global_step=episode)
                self.logger.log_scalar("episode/length", ep_len, global_step=episode)

                if hasattr(self.env, "on_episode_end"):
                    try:
                        self.env.on_episode_end()
                    except Exception:
                        pass

                state, _ = self.env.reset()
                episode += 1
                ep_return, ep_len = 0.0, 0
                self.agent.on_episode_start(episode)
            else:
                state = next_state

        # === final save ===
        if self.ckpt_dir:
            try:
                self.agent.save(self.ckpt_dir, step=self.max_steps)
                print(f"[Trainer] Final checkpoint saved at {self.ckpt_dir}")
            except Exception as e:
                print(f"[Trainer] Warning: failed to save final checkpoint: {e}")

        self.logger.close()

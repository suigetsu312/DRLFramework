# drl_inference.py
from typing import Optional, List, Dict, Any, Literal
from contextlib import nullcontext
import os

try:
    import torch
except ImportError:
    torch = None

from Logger.TensorboardLogger import TensorBoardLogger


class DRLInference:
    """
    Run a trained DRL agent for evaluation/inference, with flexible rendering/recording.
    - agent.act(state, step, deterministic=True/False) -> (action, meta)
    - agent.eval_mode() / load(path) optional
    """

    def __init__(
        self,
        env,
        agent,
        render_mode: Literal["human", "rgb_array", "auto", "none"] = "auto",
        record_video: bool = False,
        video_dir: str = "videos",
        video_prefix: str = "episode",
        save_mp4: bool = True,
        save_gif: bool = False,
        fps: int = 30,
        deterministic: bool = True,
    ):
        self.env = env
        self.agent = agent
        self.render_mode = render_mode
        self.record_video = record_video
        self.video_dir = video_dir
        self.video_prefix = video_prefix
        self.save_mp4 = save_mp4
        self.save_gif = save_gif
        self.fps = fps
        self.deterministic = deterministic

        if hasattr(self.agent, "eval_mode"):
            self.agent.eval_mode()

        os.makedirs(self.video_dir, exist_ok=True)

    def load_checkpoint(self, path: str):
        if hasattr(self.agent, "load"):
            self.agent.load(path)
        else:
            raise AttributeError("Agent has no .load(path) method")

    def _decide_mode(self) -> str:
        if self.render_mode == "auto":
            # 若 env 有宣告 render_mode 就跟它；否則預設 human
            rm = getattr(self.env, "render_mode", None)
            return rm if rm in ("human", "rgb_array") else "human"
        return self.render_mode

    def _write_video_if_needed(self, frames, ep: int):
        if not frames:
            return
        if self.save_mp4:
            try:
                import imageio
                path = os.path.join(self.video_dir, f"{self.video_prefix}_ep{ep}.mp4")
                imageio.mimsave(path, frames, fps=self.fps, macro_block_size=None)
                print(f"[Inference] Saved MP4: {path}")
            except Exception as e:
                print(f"[Inference] MP4 save failed: {e}")
        if self.save_gif:
            try:
                import imageio
                path = os.path.join(self.video_dir, f"{self.video_prefix}_ep{ep}.gif")
                imageio.mimsave(path, frames, fps=self.fps)
                print(f"[Inference] Saved GIF: {path}")
            except Exception as e:
                print(f"[Inference] GIF save failed: {e}")

    def run(
        self,
        episodes: int = 1,
        logger: Optional[TensorBoardLogger] = None,
        episode_seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        mode = self._decide_mode()
        use_frames = (mode == "rgb_array") and (self.record_video or self.save_mp4 or self.save_gif)

        nograd_ctx = torch.no_grad() if (torch is not None) else nullcontext()
        all_metrics: List[Dict[str, Any]] = []
        global_step = 0

        with nograd_ctx:
            for ep in range(episodes):
                if episode_seed is not None:
                    state, _ = self.env.reset(seed=episode_seed + ep)
                else:
                    state, _ = self.env.reset()

                if hasattr(self.agent, "eval_mode"):
                    self.agent.eval_mode()

                done = False
                step = 0
                ret = 0.0
                frames = []

                # 若是 human 模式，一些環境需要先 render 才開窗
                if mode == "human":
                    try:
                        self.env.render()
                    except Exception:
                        pass

                while not done:
                    action, meta = self.agent.act(state, step, deterministic=self.deterministic)
                    out = self.env.step(action)
                    # gymnasium step 回傳 5-tuple
                    next_state, reward, terminated, truncated, info = out
                    done = bool(terminated or truncated)

                    if logger:
                        logger.log_dict("inference/agent", meta or {}, global_step=global_step)
                        logger.log_dict("inference/reward_terms", (info or {}).get("reward_terms", {}) or {}, global_step=global_step)
                        logger.log_dict("inference/obs", (info or {}).get("obs_dict", {}) or {}, global_step=global_step)
                        if "shaped_reward" in (info or {}):
                            logger.log_scalar("inference/shaped_reward", info["shaped_reward"], global_step=global_step)

                    if mode == "human":
                        # 視窗即時動畫（需要在 make env 時設定 render_mode='human'）
                        try:
                            self.env.render()
                        except Exception:
                            pass
                    elif mode == "rgb_array":
                        # 取得單張影格（需要在 make env 時設定 render_mode='rgb_array'）
                        try:
                            frame = self.env.render()
                            if use_frames and frame is not None:
                                frames.append(frame)
                        except Exception:
                            pass

                    state = next_state
                    ret += float(reward)
                    step += 1
                    global_step += 1

                if use_frames:
                    self._write_video_if_needed(frames, ep)

                metrics = {"episode": ep, "return": ret, "length": step}
                all_metrics.append(metrics)

                if logger:
                    logger.log_scalar("inference/episode_reward", ret, global_step=ep)
                    logger.log_scalar("inference/episode_length", step, global_step=ep)

        return all_metrics

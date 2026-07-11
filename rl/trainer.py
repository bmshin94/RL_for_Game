"""
训练器（重构版）
================

提供三类训练封装：
  1. BCTrainer        —— 行为克隆，state -> 动作的监督学习。
  2. OfflineRLTrainer —— 离线 RL（CQL / AWR），从 BC 热启动，用人工奖励精炼。
  3. OnlineTrainer    —— 在线交互 + 回放（保留，标记为实验性）。

以及一个轻量 Trainer 兼容旧推理入口（供 bot_controller 调用）。
"""

from typing import Optional

import numpy as np

from rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils.my_logger import logger


# ======================================================================
# 轻量兼容包装（供实时推理 bot_controller 使用）
# ======================================================================
class Trainer:
    """最小封装：只暴露推理选动作接口，兼容旧的 BotController。"""

    def __init__(self, agent, infer_head: str = "policy"):
        self.agent = agent
        self.infer_head = infer_head  # BC 后用 "policy"，离线 RL 后可用 "q"

    def step_only_inference(self, state):
        return self.agent.select_action(state, train=False, head=self.infer_head)

    def reset(self):
        # 清空 agent 的 embedding 缓存（切换 episode）
        if hasattr(self.agent, "reset_cache"):
            self.agent.reset_cache()


# ======================================================================
# 1) 行为克隆训练器
# ======================================================================
class BCTrainer:
    """行为克隆训练器。

    只调用 agent.train_step_bc()，外层 main.py 负责 epoch 循环、验证、早停。
    这里集中管理类别权重与 label smoothing。
    """

    def __init__(self, agent, class_weights=None, label_smoothing: float = 0.05):
        self.agent = agent
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def train_batch(self, states, actions):
        """一个训练 batch，返回 (loss, accuracy)。"""
        return self.agent.train_step_bc(
            states, actions,
            class_weights=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def eval_batch(self, states, actions):
        """一个验证 batch，返回 (预测, 真实) numpy 数组。"""
        return self.agent.evaluate_bc(states, actions)


# ======================================================================
# 2) 离线 RL 训练器
# ======================================================================
class OfflineRLTrainer:
    """离线 RL 训练器（CQL / AWR）。

    method="cql"：单步保守 Q 学习，抑制未见动作的 Q 高估。
    method="awr"：优势加权行为克隆，对断裂的转移数据更鲁棒（推荐主用）。
    """

    def __init__(self, agent, method: str = "awr", target_update: int = 100):
        self.agent = agent
        self.method = method
        self.target_update = target_update
        self._steps = 0

    def train_batch(self, s1, a, r, s2, done):
        """一个训练 batch，返回 info 字典（loss/td_loss/q_mean/...）。"""
        info = self.agent.train_step_offline(s1, a, r, s2, done, method=self.method)
        self._steps += 1
        # 若未启用 Polyak 软同步，则按间隔硬同步目标网络
        if not self.agent.cfg.use_polyak and self._steps % self.target_update == 0:
            self.agent.sync_target_network()
        return info


# ======================================================================
# 3) 在线交互训练器（实验性，保留）
# ======================================================================
class OnlineTrainer:
    """在线学习训练器：实时交互 + 经验回放。

    【实验性】在线自玩需要真实奖励信号。若无游戏内存读数等外部奖励，其效果有限，
    建议以 BC + 离线 RL 为主。此处保留完整回放/PER 逻辑以便后续接入真实奖励。
    """

    def __init__(self, agent, buffer_capacity: int = 100000,
                 min_buffer_size: int = 1000, train_every: int = 4,
                 batch_size: int = 32, use_per: bool = False,
                 per_alpha: float = 0.6, per_beta_start: float = 0.4,
                 per_beta_end: float = 1.0, per_beta_anneal_steps: int = 100000,
                 normalize_reward: bool = True, state_shape: tuple = (10, 84, 84),
                 per_priority_mode: str = "proportional",
                 offline_method: str = "cql"):
        self.agent = agent
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.train_every = train_every
        self.use_per = use_per
        self.offline_method = offline_method

        if use_per:
            self.buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity, alpha=per_alpha,
                beta_start=per_beta_start, beta_end=per_beta_end,
                beta_anneal_steps=per_beta_anneal_steps,
                normalize_reward=normalize_reward, priority_mode=per_priority_mode)
        else:
            self.buffer = ReplayBuffer(capacity=buffer_capacity,
                                       normalize_reward=normalize_reward,
                                       state_shape=state_shape)

        self._current_state = None
        self._current_action = None
        self._env_steps = 0
        self._train_steps = 0
        self._episode_reward = 0.0
        self._episode_count = 0

    def act(self, state, train: bool = True) -> int:
        self._current_state = state
        action = self.agent.select_action(state, train=train, head="q")
        self._current_action = action
        return action

    def observe(self, next_state, reward: float, done: bool):
        if self._current_state is None or self._current_action is None:
            return
        self.buffer.push(state=self._current_state, action=self._current_action,
                         reward=reward, next_state=next_state, done=done)
        self._env_steps += 1
        self._episode_reward += reward
        if done:
            self._episode_count += 1
            logger.info(f"Episode {self._episode_count} 结束: reward={self._episode_reward:.2f}, "
                        f"buffer={len(self.buffer)}, env_steps={self._env_steps}")
            self._episode_reward = 0.0
        self._current_state = None
        self._current_action = None

    def maybe_train(self) -> dict:
        if not self.buffer.is_ready(self.min_buffer_size):
            return {}
        if self._env_steps % self.train_every != 0:
            return {}
        return self._do_train_step()

    def force_train(self, n_steps: int = 1) -> dict:
        if not self.buffer.is_ready(self.min_buffer_size):
            return {}
        total_loss, total_q = 0.0, 0.0
        for _ in range(n_steps):
            r = self._do_train_step()
            if r:
                total_loss += r["loss"]
                total_q += r["q_mean"]
        if n_steps > 0:
            return {"loss": total_loss / n_steps, "q_mean": total_q / n_steps}
        return {}

    def _do_train_step(self) -> dict:
        self._train_steps += 1
        if self.use_per:
            (s1, a, r, s2, done), indices, weights = self.buffer.sample(self.batch_size)
            info = self.agent.train_step_offline(s1, a, r, s2, done,
                                                 method=self.offline_method, weights=weights)
            self.buffer.update_priorities(indices, info["td_errors"])
        else:
            s1, a, r, s2, done = self.buffer.sample(self.batch_size)
            info = self.agent.train_step_offline(s1, a, r, s2, done, method=self.offline_method)
        if not self.agent.cfg.use_polyak and self._train_steps % self.agent.cfg.target_update == 0:
            self.agent.sync_target_network()
        return {"loss": info["loss"], "q_mean": info["q_mean"]}

    @property
    def env_steps(self) -> int:
        return self._env_steps

    @property
    def train_steps(self) -> int:
        return self._train_steps

    @property
    def buffer_size(self) -> int:
        return len(self.buffer)

    def save_buffer(self, path: str):
        if hasattr(self.buffer, "save"):
            self.buffer.save(path)

    def load_buffer(self, path: str):
        if hasattr(self.buffer, "load"):
            self.buffer.load(path)

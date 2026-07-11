"""
学习型奖励模型（重构版，可选/实验性）
====================================

【重要说明】旧版的奖励模型采用「TD 一致性」自监督：奖励模型去拟合
    r ≈ Q(s,a) - γ·maxQ(s')
而 DQN 又用该奖励模型的输出当奖励来训练 Q。二者互相拟合对方、没有任何外部真实
信号锚定，构成循环自证，理论上会漂移/塌缩，学不出有意义行为。**本次重构已弃用
该循环 TD 更新。**

现版本仅保留「监督式」奖励模型：在离线数据 (state, action) -> 人工 reward_sum 上做
回归。它可作为在线交互模式（实验性）的奖励估计器，但不再自我引用 Q 值。

网络复用新的 GamePolicyNet 主干（逐帧 CNN + 时序编码），额外加一个动作嵌入与回归头。
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.my_logger import logger
from rl.agent import AgentConfig, FrameEncoder, TemporalEncoder


class RewardNet(nn.Module):
    """奖励回归网络：(state 序列, action) -> 标量奖励。

    结构：逐帧 CNN（FrameEncoder）+ 时序编码（TemporalEncoder）得到状态特征，
    与动作嵌入拼接后经回归头输出标量。与 GamePolicyNet 使用相同的视觉/时序模块，
    保证特征质量一致（但参数独立，不共享，避免与 DQN 的梯度冲突）。
    """

    def __init__(self, input_frames: int, num_actions: int, cfg: AgentConfig):
        super().__init__()
        self.model_dim = cfg.model_dim
        self.frame_encoder = FrameEncoder(cfg.model_dim)
        self.temporal_encoder = TemporalEncoder(cfg, input_frames)
        self.action_embed = nn.Embedding(num_actions, cfg.model_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(cfg.model_dim * 2, cfg.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(cfg.model_dim, cfg.model_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.model_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        b, t, h, w = state.shape
        frames = state.reshape(b * t, 1, h, w)
        feat = self.frame_encoder(frames).reshape(b, t, self.model_dim)
        state_feat = self.temporal_encoder(feat)          # (B, model_dim)
        action_feat = self.action_embed(action)           # (B, model_dim)
        combined = torch.cat([state_feat, action_feat], dim=1)
        return self.reward_head(combined).squeeze(1)       # (B,)


class RewardModel:
    """监督式奖励模型封装：pretrain（离线回归）+ predict（在线估计奖励）。"""

    def __init__(self, input_frames: int, num_actions: int, hidden_dim: int = 256,
                 temporal_encoder: str = "gru", gru_layers: int = 1,
                 transformer_layers: int = 2, transformer_heads: int = 4,
                 transformer_dropout: float = 0.1, lr: float = 3e-4,
                 device: Optional[str] = None, reward_mean: float = 0.0,
                 reward_std: float = 1.0, gpu_ids: Optional[list] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.input_frames = input_frames
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.gpu_ids = gpu_ids or []

        if torch.cuda.is_available() and self.gpu_ids:
            gpu_count = torch.cuda.device_count()
            self.gpu_ids = [g for g in self.gpu_ids if 0 <= g < gpu_count]
            if self.gpu_ids:
                self.device = f"cuda:{self.gpu_ids[0]}"

        cfg = AgentConfig(
            model_dim=hidden_dim, temporal_encoder=temporal_encoder,
            gru_layers=gru_layers, transformer_layers=transformer_layers,
            transformer_heads=transformer_heads, transformer_dropout=transformer_dropout,
        )
        self.net = RewardNet(input_frames, num_actions, cfg).to(self.device)
        if len(self.gpu_ids) > 1:
            self.net = nn.DataParallel(self.net, device_ids=self.gpu_ids,
                                       output_device=self.gpu_ids[0])
            logger.info(f"RewardModel: DataParallel on GPUs {self.gpu_ids}")

        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-2)
        self._update_steps = 0

    def _to_state(self, state) -> torch.Tensor:
        t = torch.as_tensor(state)
        t = t.to(self.device)
        if t.dtype == torch.uint8:
            t = t.float().div_(255.0)
        else:
            t = t.float()
        return t

    def predict(self, state: np.ndarray, action: int) -> float:
        """预测单个 (state, action) 的奖励（反归一化回原尺度）。"""
        self.net.eval()
        with torch.no_grad():
            s = self._to_state(state).unsqueeze(0)
            a = torch.tensor([action], dtype=torch.long, device=self.device)
            r_norm = self.net(s, a).item()
        return r_norm * self.reward_std + self.reward_mean

    def pretrain_step(self, states: torch.Tensor, actions: torch.Tensor,
                      rewards: torch.Tensor) -> float:
        """离线监督回归一步：拟合归一化后的人工奖励。"""
        self.net.train()
        s = self._to_state(states)
        a = actions.to(self.device, dtype=torch.long)
        r = rewards.to(self.device, dtype=torch.float32)
        pred = self.net(s, a)
        loss = F.smooth_l1_loss(pred, r)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self._update_steps += 1
        return float(loss.item())

    def _module(self):
        return self.net.module if isinstance(self.net, nn.DataParallel) else self.net

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        torch.save({
            "net": self._module().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "update_steps": self._update_steps,
        }, tmp)
        os.replace(tmp, path)
        logger.info(f"RewardModel 已保存: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self._module().load_state_dict(ckpt["net"])
        except RuntimeError as e:
            logger.warning(f"RewardModel 结构不匹配，跳过加载: {e}")
            return
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        self.reward_mean = ckpt.get("reward_mean", 0.0)
        self.reward_std = ckpt.get("reward_std", 1.0)
        self._update_steps = ckpt.get("update_steps", 0)
        logger.info(f"RewardModel 已加载: {path} (update_steps={self._update_steps})")

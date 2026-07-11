"""
核心网络与智能体（重构版）
============================

本文件定义了「从 640x640 灰度屏幕序列 -> 26 个离散动作」的统一网络 GamePolicyNet，
以及封装了训练/推理/存取逻辑的 RLAgent。

设计要点（相较旧版的关键改动，均为「能真正玩起来」服务）：

1. 单网络三头共享主干
   - policy_head：输出 26 维动作 logits，供「行为克隆 BC」使用。
   - value_stream / adv_stream：Dueling 结构，输出 Q 值，供「离线 RL（CQL/AWR）」使用。
   - 三头共享同一 backbone（逐帧 CNN + 时序编码），因此离线 RL 可以直接从 BC 权重热启动。

2. 逐帧 CNN + 权重共享 + 推理时 embedding 缓存
   - 10 帧的时序窗口每步只前移 1 帧，推理时只需对「新帧」跑一次 CNN，其余 9 帧
     的 embedding 从环形缓存复用。这是 640x640 大分辨率仍能实时的关键。

3. GroupNorm 取代 BatchNorm
   - RL/BC 的 batch 常常很小（在线 batch=2，推理 batch=1），BatchNorm 的滑动统计量
     会失真，且 target 网络与在线网络的 running-stat 会相互漂移。GroupNorm 与 batch
     大小无关，batch=1 也正确。

4. 规模匹配数据量
   - model_dim 从 1024 降到 256，Transformer 8 层替换为单层 GRU（或 2 层轻量
     Transformer，可在 config 切换）。参数量从数千万降到数百万，匹配 ~千级样本。

5. 动作屏蔽（action mask）
   - 数据集中有 15 个动作完全没有样本，网络学不出来。通过 mask 把它们在
     logits / Q 值上强制置为 -inf，避免推理时误选、也避免离线 RL 高估这些 OOD 动作。
"""

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from utils.my_logger import logger


# ======================================================================
# 配置对象
# ======================================================================
@dataclass
class AgentConfig:
    """智能体与网络的超参数集合（从 config.yaml 的 rl 段解析而来）。"""

    # --- 通用 RL ---
    gamma: float = 0.99                 # 折扣因子（离线 RL 单步模式下会被强制为 0）
    lr: float = 1e-4                    # 学习率
    grad_clip: float = 1.0             # 梯度裁剪阈值

    # --- 网络结构 ---
    model_dim: int = 256                # 每帧 embedding 维度 / 时序特征维度
    temporal_encoder: str = "gru"      # 时序编码器类型："gru" 或 "transformer"
    gru_layers: int = 1                 # GRU 层数（temporal_encoder=gru 时生效）
    transformer_layers: int = 2         # Transformer 层数（temporal_encoder=transformer 时生效）
    transformer_heads: int = 4          # Transformer 注意力头数
    transformer_dropout: float = 0.1    # Transformer dropout
    head_dropout: float = 0.3           # 三个输出头的 dropout（正则，防过拟合）

    # --- 探索（仅在线/交互模式用到；离线 BC 不需要）---
    exploration_method: str = "epsilon"
    epsilon_start: float = 0.15
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 200000
    boltzmann_temperature_start: float = 5.0
    boltzmann_temperature_end: float = 0.5
    boltzmann_temperature_decay_steps: int = 50000

    # --- 目标网络 ---
    target_update: int = 1000           # 硬同步间隔（use_polyak=False 时用）
    polyak_tau: float = 0.005           # Polyak 软同步系数
    use_polyak: bool = True             # 是否使用 Polyak 软同步

    # --- 离线 RL 相关 ---
    use_double_dqn: bool = True         # Double DQN
    cql_alpha: float = 1.0              # CQL 保守正则强度
    awr_beta: float = 1.0               # AWR/CRR 优势温度（w = exp(A/beta)）
    awr_weight_clip: float = 20.0       # AWR 权重上限（防爆炸）

    # --- 推理平局打破 ---
    inference_topk: int = 3
    inference_log_every: int = 30
    inference_tie_delta: float = 0.0
    inference_tie_topk: int = 3
    inference_tie_temperature: float = 5.0

    # --- LR 调度 ---
    lr_warmup_steps: int = 1000
    lr_min_ratio: float = 0.01

    # --- 设备 ---
    gpu_ids: Tuple[int, ...] = ()

    # --- 动作屏蔽 ---
    # valid_actions 为空表示不屏蔽任何动作；否则只有列出的动作 id 参与预测/训练。
    valid_actions: Tuple[int, ...] = ()


# ======================================================================
# 基础网络模块
# ======================================================================
def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """构造一个 GroupNorm。

    组数取 min(max_groups, num_channels) 且需能整除通道数，保证任意通道数都能工作。
    """
    groups = min(max_groups, num_channels)
    # 向下寻找能整除的组数，避免 GroupNorm 报错
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class DepthwiseSeparableBlock(nn.Module):
    """深度可分离卷积 + 残差块（带下采样）。

    depthwise（分组=通道数的 3x3 卷积，负责空间信息）+ pointwise（1x1 卷积，负责通道
    混合）的组合，参数量和计算量远小于普通 3x3 卷积，适合在大分辨率上快速下采样。
    当 stride>1 或输入输出通道不一致时，shortcut 分支用 1x1 卷积对齐。
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # --- 主分支：depthwise 3x3（可下采样）---
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.dw_norm = _group_norm(in_ch)
        # --- 主分支：pointwise 1x1（改变通道）---
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_norm = _group_norm(out_ch)
        self.act = nn.ReLU(inplace=True)

        # --- shortcut 分支：需要时用 1x1 对齐空间尺寸与通道数 ---
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                _group_norm(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.dw_norm(self.dw(x)))
        out = self.pw_norm(self.pw(out))
        out = self.act(out + identity)
        return out


class FrameEncoder(nn.Module):
    """逐帧 CNN 主干：单帧 (1, 640, 640) 灰度 -> model_dim 维 embedding。

    通过 5 次 stride 下采样把 640 一路压到 10x10（共 64 倍空间缩减），再全局平均池化
    + 线性投影得到每帧特征。相比旧版只压到 80x80，这里显著降低了后续计算量，是实时
    推理可行的核心。所有归一化使用 GroupNorm。
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

        # stem：7x7、stride=4，快速把 640 -> 160，通道 1 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=4, padding=3, bias=False),
            _group_norm(32),
            nn.ReLU(inplace=True),
        )
        # 逐级下采样：160 -> 80 -> 40 -> 20 -> 10
        self.stage2 = DepthwiseSeparableBlock(32, 64, stride=2)    # 160 -> 80
        self.stage3 = DepthwiseSeparableBlock(64, 96, stride=2)    # 80 -> 40
        self.stage4 = DepthwiseSeparableBlock(96, 128, stride=2)   # 40 -> 20
        self.stage5 = DepthwiseSeparableBlock(128, 192, stride=2)  # 20 -> 10

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) 单帧灰度批（已归一化到 [0,1]）
        Returns:
            (B, out_dim) 每帧特征
        """
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x)
        return self.proj(x)


class TemporalPositionalEncoding(nn.Module):
    """可学习的时序位置编码（仅 Transformer 时序编码器使用）。"""

    def __init__(self, model_dim: int, max_len: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, model_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]


class TemporalEncoder(nn.Module):
    """时序编码器：把 (B, T, model_dim) 的逐帧特征序列聚合为 (B, model_dim) 状态特征。

    支持两种实现（config.temporal_encoder 切换）：
      - "gru"：单/多层单向 GRU，取最后时刻隐状态。因果（不看未来帧）、轻量、参数少，
               在小数据上更稳，是默认推荐。
      - "transformer"：轻量 TransformerEncoder + 注意力池化。表达力更强但更易过拟合。
    """

    def __init__(self, cfg: AgentConfig, input_frames: int):
        super().__init__()
        self.kind = cfg.temporal_encoder
        self.model_dim = cfg.model_dim

        if self.kind == "gru":
            self.gru = nn.GRU(
                input_size=cfg.model_dim,
                hidden_size=cfg.model_dim,
                num_layers=cfg.gru_layers,
                batch_first=True,
            )
        elif self.kind == "transformer":
            self.pos_encoder = TemporalPositionalEncoding(cfg.model_dim, input_frames)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.model_dim,
                nhead=cfg.transformer_heads,
                dim_feedforward=cfg.model_dim * 2,   # 轻量 FFN（旧版是 *4）
                dropout=cfg.transformer_dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
            # 注意力池化：对时序做加权求和，突出关键帧
            self.attn = nn.Sequential(
                nn.Linear(cfg.model_dim, cfg.model_dim // 2),
                nn.Tanh(),
                nn.Linear(cfg.model_dim // 2, 1),
            )
        else:
            raise ValueError(f"未知的 temporal_encoder: {self.kind}（应为 'gru' 或 'transformer'）")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, model_dim) 逐帧特征序列
        Returns:
            (B, model_dim) 聚合后的状态特征
        """
        if self.kind == "gru":
            # GRU 输出 (output, h_n)，h_n 形状 (num_layers, B, hidden)，取最后一层
            _, h_n = self.gru(x)
            return h_n[-1]
        else:
            x = self.pos_encoder(x)
            feat = self.encoder(x)                     # (B, T, D)
            attn_w = torch.softmax(self.attn(feat), dim=1)  # (B, T, 1)
            context = (attn_w * feat).sum(dim=1)       # (B, D) 注意力池化
            return context


# ======================================================================
# 主网络：GamePolicyNet
# ======================================================================
class GamePolicyNet(nn.Module):
    """统一网络：屏幕序列 -> {动作 logits(BC), Q 值(离线RL)}。

    结构：FrameEncoder（逐帧，权重共享）-> TemporalEncoder -> 三个头。
    三头共享 backbone，一个 checkpoint 即可同时用于 BC 与离线 RL。
    """

    def __init__(self, input_frames: int, num_actions: int, cfg: AgentConfig):
        super().__init__()
        self.input_frames = input_frames
        self.num_actions = num_actions
        self.model_dim = cfg.model_dim

        # --- 共享 backbone ---
        self.frame_encoder = FrameEncoder(cfg.model_dim)
        self.temporal_encoder = TemporalEncoder(cfg, input_frames)

        # --- 三个输出头 ---
        # BC 策略头：输出动作 logits
        self.policy_head = nn.Sequential(
            nn.Linear(cfg.model_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(128, num_actions),
        )
        # Dueling 价值头
        self.value_stream = nn.Sequential(
            nn.Linear(cfg.model_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(128, 1),
        )
        # Dueling 优势头
        self.adv_stream = nn.Sequential(
            nn.Linear(cfg.model_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(128, num_actions),
        )

        # --- 动作屏蔽掩码 ---
        # action_mask: (num_actions,) 的布尔缓冲区，True 表示该动作有效。
        # 注册为 buffer 以便随模型一起保存/搬移设备，但不参与梯度。
        mask = torch.ones(num_actions, dtype=torch.bool)
        if cfg.valid_actions:
            mask = torch.zeros(num_actions, dtype=torch.bool)
            for a in cfg.valid_actions:
                if 0 <= a < num_actions:
                    mask[a] = True
        self.register_buffer("action_mask", mask)

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """屏幕序列 -> 状态特征。

        Args:
            x: (B, T, H, W) 已归一化到 [0,1] 的灰度序列
        Returns:
            (B, model_dim) 状态特征
        """
        b, t, h, w = x.shape
        # 逐帧过 CNN：把 (B,T,H,W) 展平成 (B*T,1,H,W)，CNN 权重对每帧共享
        frames = x.reshape(b * t, 1, h, w)
        frame_feat = self.frame_encoder(frames)        # (B*T, model_dim)
        frame_feat = frame_feat.reshape(b, t, self.model_dim)
        return self.temporal_encoder(frame_feat)        # (B, model_dim)

    def encode_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """对单帧编码，供推理时的 embedding 缓存使用。

        Args:
            frame: (B, 1, H, W) 单帧灰度
        Returns:
            (B, model_dim) 单帧 embedding
        """
        return self.frame_encoder(frame)

    def temporal_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """从「已缓存的逐帧 embedding 序列」直接做时序聚合（跳过 CNN）。

        Args:
            embeddings: (B, T, model_dim) 逐帧 embedding
        Returns:
            (B, model_dim) 状态特征
        """
        return self.temporal_encoder(embeddings)

    # ------------------------------------------------------------------
    # 三个输出
    # ------------------------------------------------------------------
    def _apply_mask_to_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """把无效动作的 logits 置为 -inf，使其 softmax 概率与 argmax 永不被选中。"""
        if self.action_mask.all():
            return logits
        mask = self.action_mask.to(logits.device)
        neg_inf = torch.finfo(logits.dtype).min
        return logits.masked_fill(~mask.unsqueeze(0), neg_inf)

    def policy_logits(self, features: torch.Tensor, apply_mask: bool = True) -> torch.Tensor:
        """BC 策略头：状态特征 -> 动作 logits（默认应用动作屏蔽）。"""
        logits = self.policy_head(features)
        return self._apply_mask_to_logits(logits) if apply_mask else logits

    def q_values(self, features: torch.Tensor, apply_mask: bool = True) -> torch.Tensor:
        """Dueling Q：Q = V + (A - mean(A))，默认应用动作屏蔽。"""
        value = self.value_stream(features)
        advantage = self.adv_stream(features)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return self._apply_mask_to_logits(q) if apply_mask else q

    def forward(self, x: torch.Tensor, head: str = "policy",
                apply_mask: bool = True) -> torch.Tensor:
        """统一前向。

        Args:
            x: (B, T, H, W) 屏幕序列
            head: "policy" 返回动作 logits；"q" 返回 Q 值
        """
        feat = self.extract_features(x)
        if head == "policy":
            return self.policy_logits(feat, apply_mask=apply_mask)
        elif head == "q":
            return self.q_values(feat, apply_mask=apply_mask)
        else:
            raise ValueError(f"未知的 head: {head}（应为 'policy' 或 'q'）")


# ======================================================================
# 智能体：RLAgent
# ======================================================================
class RLAgent:
    """封装网络、优化器、训练步（BC/CQL/AWR）、推理与存取。

    - BC：train_step_bc()，交叉熵 + 类别权重 + label smoothing。
    - 离线 RL：train_step_offline()，支持 CQL（单步保守 Q）与 AWR（优势加权 BC）。
    - 推理：select_action()，走 policy 头（BC 后）或 q 头，支持 embedding 缓存。
    """

    def __init__(
        self,
        num_actions: int,
        model_input_dime: int,
        target_update: int = 1000,
        device: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.num_actions = num_actions
        self.input_frames = model_input_dime

        cfg = config or {}
        self.cfg = self._build_config(cfg, target_update)

        # --- 设备解析（支持多卡 DataParallel）---
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_gpu_ids = self._resolve_gpus()

        # --- 构建在线网络与目标网络 ---
        self.online_net = GamePolicyNet(model_input_dime, num_actions, self.cfg).to(self.device)
        self.target_net = GamePolicyNet(model_input_dime, num_actions, self.cfg).to(self.device)

        if len(self.multi_gpu_ids) > 1:
            self.online_net = nn.DataParallel(
                self.online_net, device_ids=list(self.multi_gpu_ids),
                output_device=self.multi_gpu_ids[0])
            self.target_net = nn.DataParallel(
                self.target_net, device_ids=list(self.multi_gpu_ids),
                output_device=self.multi_gpu_ids[0])
            logger.info(f"启用 DataParallel 多卡训练: {self.multi_gpu_ids}")

        self._sync_target_hard()
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.online_net.parameters(), lr=self.cfg.lr,
                                     weight_decay=1e-2)
        self.scheduler = self._build_lr_scheduler()

        self.update_counter = 0
        self.total_env_steps = 0
        self.inference_steps = 0

        # 推理时的 embedding 缓存（环形），加速实时推理
        self._embed_cache: Optional[torch.Tensor] = None  # (T, model_dim)

    # ------------------------------------------------------------------
    # 初始化辅助
    # ------------------------------------------------------------------
    def _build_config(self, cfg: dict, target_update: int) -> AgentConfig:
        """从字典构建 AgentConfig，缺省值向后兼容。"""

        def _norm_ids(ids) -> Tuple[int, ...]:
            if ids is None:
                return ()
            if isinstance(ids, str):
                return tuple(int(x) for x in ids.split(",") if x.strip())
            if isinstance(ids, Sequence):
                return tuple(int(x) for x in ids)
            return ()

        valid_actions = cfg.get("valid_actions", ())
        if valid_actions is None:
            valid_actions = ()

        c = AgentConfig(
            gamma=cfg.get("gamma", 0.99),
            lr=cfg.get("lr", 1e-4),
            grad_clip=cfg.get("grad_clip", 1.0),
            model_dim=cfg.get("model_dim", 256),
            temporal_encoder=cfg.get("temporal_encoder", "gru"),
            gru_layers=cfg.get("gru_layers", 1),
            transformer_layers=cfg.get("transformer_layers", 2),
            transformer_heads=cfg.get("transformer_heads", 4),
            transformer_dropout=cfg.get("transformer_dropout", 0.1),
            head_dropout=cfg.get("head_dropout", 0.3),
            exploration_method=cfg.get("exploration_method", "epsilon"),
            epsilon_start=cfg.get("epsilon_start", 0.15),
            epsilon_end=cfg.get("epsilon_end", 0.02),
            epsilon_decay_steps=cfg.get("epsilon_decay_steps", 200000),
            boltzmann_temperature_start=cfg.get("boltzmann_temperature_start", 5.0),
            boltzmann_temperature_end=cfg.get("boltzmann_temperature_end", 0.5),
            boltzmann_temperature_decay_steps=cfg.get("boltzmann_temperature_decay_steps", 50000),
            target_update=cfg.get("target_update", target_update),
            polyak_tau=cfg.get("polyak_tau", 0.005),
            use_polyak=cfg.get("use_polyak", True),
            use_double_dqn=cfg.get("use_double_dqn", True),
            cql_alpha=cfg.get("cql_alpha", 1.0),
            awr_beta=cfg.get("awr_beta", 1.0),
            awr_weight_clip=cfg.get("awr_weight_clip", 20.0),
            inference_topk=cfg.get("inference_topk", 3),
            inference_log_every=cfg.get("inference_log_every", 30),
            inference_tie_delta=cfg.get("inference_tie_delta", 0.0),
            inference_tie_topk=cfg.get("inference_tie_topk", 3),
            inference_tie_temperature=cfg.get("inference_tie_temperature", 5.0),
            lr_warmup_steps=cfg.get("lr_warmup_steps", 1000),
            lr_min_ratio=cfg.get("lr_min_ratio", 0.01),
            gpu_ids=_norm_ids(cfg.get("gpu_ids", ())),
            valid_actions=tuple(int(a) for a in valid_actions),
        )

        if c.temporal_encoder == "transformer" and c.model_dim % c.transformer_heads != 0:
            raise ValueError("model_dim 必须能被 transformer_heads 整除")
        return c

    def _resolve_gpus(self) -> Tuple[int, ...]:
        """校验并解析可用的 GPU id 列表。"""
        if not (torch.cuda.is_available() and self.cfg.gpu_ids):
            return ()
        gpu_count = torch.cuda.device_count()
        valid = tuple(gid for gid in self.cfg.gpu_ids if 0 <= gid < gpu_count)
        if not valid:
            logger.warning(f"配置的 gpu_ids={self.cfg.gpu_ids} 在本机无效，回退到 {self.device}")
            return ()
        self.device = f"cuda:{valid[0]}"
        return valid

    def _net(self, net: nn.Module) -> nn.Module:
        """解包 DataParallel，拿到真正的 GamePolicyNet。"""
        return net.module if isinstance(net, nn.DataParallel) else net

    # ------------------------------------------------------------------
    # LR 调度：线性 warmup + cosine 衰减
    # ------------------------------------------------------------------
    def _build_lr_scheduler(self, total_train_steps: int = 0) -> LambdaLR:
        warmup = self.cfg.lr_warmup_steps
        min_ratio = self.cfg.lr_min_ratio
        total = total_train_steps if total_train_steps > 0 else self.cfg.target_update * 200

        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total - warmup))
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # 训练/推理模式
    # ------------------------------------------------------------------
    def set_train_mode(self):
        self.online_net.train()

    def set_eval_mode(self):
        self.online_net.eval()

    def reset_cache(self):
        """清空推理 embedding 缓存（切换 episode / F6 关闭时调用）。"""
        self._embed_cache = None

    # ------------------------------------------------------------------
    # 数据搬运辅助
    # ------------------------------------------------------------------
    def _to_float_state(self, state: torch.Tensor) -> torch.Tensor:
        """把 state 搬到设备并转 float。

        支持 uint8 输入（[0,255]），会在 GPU 上归一化到 [0,1]，以节省内存/带宽。
        """
        state = state.to(self.device, non_blocking=True)
        if state.dtype == torch.uint8:
            state = state.float().div_(255.0)
        else:
            state = state.float()
        return state

    # ==================================================================
    # 训练步 1：行为克隆 BC
    # ==================================================================
    def train_step_bc(self, states: torch.Tensor, actions: torch.Tensor,
                      class_weights: Optional[torch.Tensor] = None,
                      label_smoothing: float = 0.05) -> Tuple[float, float]:
        """行为克隆：监督学习 state -> 人类动作。

        Args:
            states: (B, T, H, W) 屏幕序列（uint8 或 float）
            actions: (B,) 人类动作 id
            class_weights: (num_actions,) 类别权重（处理不均衡），可选
            label_smoothing: 标签平滑系数
        Returns:
            (loss, accuracy)
        """
        states = self._to_float_state(states)
        actions = actions.to(self.device, dtype=torch.long).view(-1)

        # 走 policy 头，但训练时不 mask（保留完整 logits 让交叉熵正常；无效动作靠
        # 类别权重=0 / 数据中不出现来自然抑制）。推理时才 mask。
        # 统一通过 self.online_net(...) 调用，DataParallel 才能正确切分到多卡。
        logits = self.online_net(states, head="policy", apply_mask=False)

        if class_weights is not None:
            class_weights = class_weights.to(self.device, dtype=torch.float32)
        loss = F.cross_entropy(logits, actions, weight=class_weights,
                               label_smoothing=label_smoothing)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.update_counter += 1

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == actions).float().mean().item()
        return float(loss.item()), float(acc)

    @torch.no_grad()
    def evaluate_bc(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """BC 验证：返回 (预测动作, 真实动作) 的 numpy 数组，供外部算 macro-F1。"""
        self.online_net.eval()
        states = self._to_float_state(states)
        actions = actions.to(self.device, dtype=torch.long).view(-1)
        logits = self.online_net(states, head="policy", apply_mask=True)
        pred = logits.argmax(dim=1)
        return pred.cpu().numpy(), actions.cpu().numpy()

    # ==================================================================
    # 训练步 2：离线 RL（CQL 单步保守 Q + 可选 AWR 优势加权 BC）
    # ==================================================================
    def train_step_offline(self, states: torch.Tensor, actions: torch.Tensor,
                           rewards: torch.Tensor, next_states: torch.Tensor,
                           dones: torch.Tensor, method: str = "cql",
                           weights: Optional[torch.Tensor] = None) -> dict:
        """离线 RL 训练步。

        由于本项目数据没有真正的 episode 结构、next_state 是同段录像里稍后的几帧，
        因此采用「单步」设定：dones 应全为 1、gamma 内部按需处理，避免通过伪造的
        next_state 做不可靠的 bootstrap。

        Args:
            method: "cql" 使用保守 Q 学习；"awr" 使用优势加权行为克隆。
        """
        states = self._to_float_state(states)
        next_states = self._to_float_state(next_states)
        actions = actions.to(self.device, dtype=torch.long).view(-1)
        rewards = rewards.to(self.device, dtype=torch.float32).view(-1)
        dones = dones.to(self.device, dtype=torch.float32).view(-1)

        # 离线 RL 需要复用同一份状态特征 feat 给 q 头与 policy 头（AWR），因此直接
        # 在解包后的网络上调用；多卡时退化为主卡单卡计算（功能正确，只是不加速）。
        # 离线批量小、单卡足够，这里以正确性与代码清晰度优先。
        net = self._net(self.online_net)
        tgt = self._net(self.target_net)

        feat = net.extract_features(states)
        q_all = net.q_values(feat, apply_mask=True)          # (B, A)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- TD 目标（单步：done=1 时退化为纯 reward 回归，不 bootstrap）---
        with torch.no_grad():
            gamma = self.cfg.gamma
            if self.cfg.use_double_dqn:
                next_a = net.q_values(net.extract_features(next_states)).argmax(dim=1, keepdim=True)
                next_q = tgt.q_values(tgt.extract_features(next_states)).gather(1, next_a).squeeze(1)
            else:
                next_q = tgt.q_values(tgt.extract_features(next_states)).max(dim=1)[0]
            td_target = rewards + gamma * next_q * (1.0 - dones)

        td_loss_elem = F.smooth_l1_loss(q_sa, td_target, reduction="none")
        if weights is not None:
            weights = weights.to(self.device, dtype=torch.float32).view(-1)
            td_loss = (td_loss_elem * weights).mean()
        else:
            td_loss = td_loss_elem.mean()

        info = {}
        if method == "cql":
            # CQL 保守正则：logsumexp_a Q(s,a) - Q(s,a_data)
            # 压低所有动作（尤其 OOD 动作）的 Q，抬高数据里出现过的动作 Q，
            # 直接修复「离线 DQN 高估未见动作」的问题。
            logsumexp_q = torch.logsumexp(q_all, dim=1)
            cql_loss = (logsumexp_q - q_sa).mean()
            loss = td_loss + self.cfg.cql_alpha * cql_loss
            info["cql_loss"] = float(cql_loss.item())
        elif method == "awr":
            # AWR/CRR：用优势对 BC 交叉熵加权，w = exp(A/beta) 截断。
            with torch.no_grad():
                value = q_all.max(dim=1)[0]           # 用 max_a Q 近似 V(s)
                advantage = q_sa - value
                w = torch.exp(advantage / self.cfg.awr_beta).clamp(max=self.cfg.awr_weight_clip)
            logits = net.policy_logits(feat, apply_mask=False)
            policy_loss = (F.cross_entropy(logits, actions, reduction="none") * w).mean()
            loss = td_loss + policy_loss
            info["policy_loss"] = float(policy_loss.item())
        else:
            raise ValueError(f"未知的离线 RL method: {method}（应为 'cql' 或 'awr'）")

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.update_counter += 1

        if self.cfg.use_polyak:
            self._polyak_update()

        with torch.no_grad():
            per_sample_td = (td_target - q_sa).abs().detach().cpu().numpy()
        info.update({
            "loss": float(loss.item()),
            "td_loss": float(td_loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "td_errors": per_sample_td,
        })
        return info

    # ------------------------------------------------------------------
    # 目标网络同步
    # ------------------------------------------------------------------
    def _sync_target_hard(self):
        self._net(self.target_net).load_state_dict(self._net(self.online_net).state_dict())

    def sync_target_network(self):
        """硬同步 online -> target。"""
        self._sync_target_hard()
        logger.info(f"目标网络硬同步完成 @ update_counter={self.update_counter}")

    def _polyak_update(self):
        """软同步：target = tau*online + (1-tau)*target。"""
        tau = self.cfg.polyak_tau
        online = self._net(self.online_net)
        target = self._net(self.target_net)
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)

    # ==================================================================
    # 推理：选动作
    # ==================================================================
    def _compute_epsilon(self) -> float:
        progress = min(1.0, self.total_env_steps / float(self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + (self.cfg.epsilon_end - self.cfg.epsilon_start) * progress

    @torch.inference_mode()
    def select_action(self, state: np.ndarray, train: bool = False,
                      head: str = "policy") -> int:
        """根据当前 state 选动作。

        Args:
            state: (T, H, W) numpy，float[0,1] 或 uint8
            train: 交互训练时是否加探索（BC 离线训练不经过这里）
            head: "policy"（BC 策略，推荐）或 "q"（Q 值 argmax）
        Returns:
            动作 id
        """
        if state is None:
            return 0

        if train:
            self.total_env_steps += 1

        state_t = torch.as_tensor(state)
        state_t = self._to_float_state(state_t).unsqueeze(0)  # (1, T, H, W)

        if head == "policy":
            logits = self.online_net(state_t, head="policy", apply_mask=True).squeeze(0)
            scores = logits
        else:
            scores = self.online_net(state_t, head="q", apply_mask=True).squeeze(0)

        # 交互训练时的 epsilon 探索（只在有效动作里随机）
        if train and self.cfg.exploration_method == "epsilon":
            import random
            if random.random() < self._compute_epsilon():
                valid_idx = torch.nonzero(self._net(self.online_net).action_mask, as_tuple=False).view(-1)
                return int(valid_idx[random.randrange(len(valid_idx))].item())

        action = int(scores.argmax(dim=0).item())

        # 推理日志
        self.inference_steps += 1
        if (not train and self.cfg.inference_log_every > 0
                and self.inference_steps % self.cfg.inference_log_every == 0):
            topk = max(1, min(self.cfg.inference_topk, self.num_actions))
            top_vals, top_idx = torch.topk(scores, k=topk, dim=0)
            items = ", ".join(f"{int(i)}:{float(v):.3f}" for i, v in zip(top_idx.tolist(), top_vals.tolist()))
            logger.info(f"[推理] head={head} top{topk} -> {items}; 选择={action}")

        return action

    # ------------------------------------------------------------------
    # 存取
    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        online = self._net(self.online_net)
        target = self._net(self.target_net)
        tmp_path = path + ".tmp"
        torch.save({
            "online_net": online.state_dict(),
            "target_net": target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "update_counter": self.update_counter,
            "total_env_steps": self.total_env_steps,
            "agent_config": self.cfg.__dict__,
        }, tmp_path)
        os.replace(tmp_path, path)
        logger.info(f"Agent 已保存: {path}")

    def load(self, path: str, strict: bool = True):
        """加载 checkpoint。

        strict=False 时做「部分加载」：只加载 shape 匹配的权重，用于 BC checkpoint
        热启动到离线 RL（backbone/时序/policy 头全部匹配，只有优化器状态可能不同）。
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        def _load(model: nn.Module, sd: dict, name: str):
            target = self._net(model)
            if not sd:
                return
            # 处理 module. 前缀差异（DataParallel）
            ckpt_mod = next(iter(sd)).startswith("module.")
            model_mod = next(iter(target.state_dict())).startswith("module.")
            if ckpt_mod and not model_mod:
                sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            elif model_mod and not ckpt_mod:
                sd = {f"module.{k}": v for k, v in sd.items()}
            try:
                target.load_state_dict(sd)
            except RuntimeError as err:
                if strict:
                    raise RuntimeError(
                        f"{name} 结构与 checkpoint 不匹配（{path}）。请用当前网络定义重新训练。"
                    ) from err
                cur = target.state_dict()
                matched = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
                if not matched:
                    logger.warning(f"{name} 无兼容权重（{path}），从零开始。")
                    return
                cur.update(matched)
                target.load_state_dict(cur)
                logger.warning(f"{name} 部分加载 {len(matched)}/{len(cur)} 个张量（{path}）。")

        if "online_net" in ckpt:
            _load(self.online_net, ckpt["online_net"], "online_net")
            _load(self.target_net, ckpt.get("target_net", ckpt["online_net"]), "target_net")
        else:
            raise KeyError("无效的 checkpoint：缺少 'online_net'。")

        # 优化器/调度器状态仅在结构一致时恢复（热启动到离线 RL 时通常跳过）
        if strict and "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                logger.warning(f"优化器/调度器状态未恢复: {e}")

        self.update_counter = int(ckpt.get("update_counter", 0))
        self.total_env_steps = int(ckpt.get("total_env_steps", 0))
        self.online_net.eval()
        self.target_net.eval()
        logger.info(f"Agent 已加载: {path} (strict={strict})")

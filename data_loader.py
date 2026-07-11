"""
数据管线（重构版）
==================

面向两种离线训练：
  1. 行为克隆 BC —— BCDataset：屏幕序列 state -> 人类动作 action。
  2. 离线 RL —— TransitionDataset：单步转移 (s, a, r, s', done=1)。

关键设计：
  - 帧优先加载预处理好的 .npy（float32 [0,1]），否则回退读 .jpg 实时 letterbox。
  - state 为连续 continue_num 帧，取自标注动作发生帧 ctrl_a_frame 之前，符合「先观察
    再决策」的因果关系。
  - BC 增强对同一 10 帧栈施加「一致」的变换（亮度/对比度/平移/噪声/cutout），严禁
    水平翻转（会翻转左右转向语义、污染标签）。
  - 提供「按录像划分 train/val」的工具，避免同一段录像的帧同时出现在训练集与验证集
    造成信息泄漏、虚高指标。
  - 提供类别权重（sqrt-tempered 逆频率）与采样权重，缓解动作严重不均衡。
"""

import os
import ast
import csv
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from core.vision_engine import VisionEngine
from utils.my_logger import logger


# ======================================================================
# 帧加载
# ======================================================================
def _load_frame(video_dir: str, frame_idx: int, img_size: int) -> Optional[np.ndarray]:
    """加载单帧，返回 (H, W) float32 [0,1] 或 None。

    优先读预处理的 .npy；不存在则回退读 .jpg 并实时 letterbox + 归一化。
    """
    npy_path = os.path.join(video_dir, f"{frame_idx}.npy")
    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        # 兼容历史上可能存成 uint8 的情况
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        return arr

    img_path = os.path.join(video_dir, f"{frame_idx}.jpg")
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = VisionEngine._letterbox(img, img_size)
    return resized.astype(np.float32) / 255.0


def _load_state_frames(video_dir: str, end_frame: int, continue_num: int,
                       img_size: int) -> Optional[np.ndarray]:
    """加载 [end_frame-continue_num, end_frame) 的连续帧，堆叠为 (T, H, W)。

    任一帧缺失则返回 None（由上层过滤掉该样本）。
    """
    frames = []
    for i in range(end_frame - continue_num, end_frame):
        f = _load_frame(video_dir, i, img_size)
        if f is None:
            return None
        frames.append(f)
    if len(frames) < continue_num:
        return None
    return np.stack(frames, axis=0).astype(np.float32)


# ======================================================================
# CSV 解析
# ======================================================================
@dataclass
class TrainSample:
    """一条标注样本（一个「动作时刻」拆出的单动作样本）。"""
    video_dir: str          # 录像帧目录
    ctrl_a_frame: int       # 标注动作发生的帧号
    action: int             # 动作 id
    reward_ids: list        # 原始奖励标签列表
    reward_sum: int         # 奖励总和（离线 RL 的奖励信号）
    done: int               # 终止标记（本数据基本无意义，离线 RL 单步模式会强制为 1）


def load_csv_to_cache(csv_path: str) -> List[TrainSample]:
    """读取 records.csv，把每行的多动作展开成多条 TrainSample。"""
    cache: List[TrainSample] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards = ast.literal_eval(row["rewards"])
            actions = ast.literal_eval(row["actions"])
            for action in actions:
                cache.append(TrainSample(
                    video_dir=row["video_dir"],
                    ctrl_a_frame=int(row["ctrl_a_frame"]),
                    action=int(action),
                    reward_ids=rewards,
                    reward_sum=int(row["reward_sum"]),
                    done=int(row["done"]),
                ))
    return cache


def compute_reward_stats(cache: List[TrainSample]) -> Tuple[float, float]:
    """计算 reward_sum 的均值/标准差，用于离线 RL 的奖励归一化。"""
    if not cache:
        return 0.0, 1.0
    rewards = np.array([float(s.reward_sum) for s in cache], dtype=np.float32)
    mean = float(rewards.mean())
    std = float(rewards.std())
    if std < 1e-8:
        std = 1.0
    logger.info(f"奖励归一化统计: mean={mean:.4f}, std={std:.4f}")
    return mean, std


# ======================================================================
# 动作统计 / 类别均衡
# ======================================================================
def compute_action_stats(cache: List[TrainSample], num_actions: int) -> np.ndarray:
    """统计每个动作 id 的样本数，返回 (num_actions,) 计数数组。"""
    counts = np.zeros(num_actions, dtype=np.int64)
    for s in cache:
        if 0 <= s.action < num_actions:
            counts[s.action] += 1
    return counts


def get_valid_actions(cache: List[TrainSample], num_actions: int,
                      min_count: int = 1) -> List[int]:
    """返回样本数 >= min_count 的动作 id 列表（有效动作空间）。

    无数据的动作学不出来，应从预测/训练中屏蔽，并提示补采集。
    """
    counts = compute_action_stats(cache, num_actions)
    valid = [a for a in range(num_actions) if counts[a] >= min_count]
    missing = [a for a in range(num_actions) if counts[a] == 0]
    if missing:
        logger.warning(f"以下 {len(missing)} 个动作没有任何样本，将被屏蔽（需补采集数据）: {missing}")
    logger.info(f"有效动作 {len(valid)}/{num_actions}: {valid}")
    return valid


def compute_class_weights(cache: List[TrainSample], num_actions: int,
                          temper: float = 0.5, max_weight: float = 10.0) -> torch.Tensor:
    """计算类别权重（sqrt-tempered 逆频率），用于加权交叉熵。

    weight_a ∝ (1/count_a)^temper，再归一化并封顶。temper=0.5 即 1/sqrt(freq)，
    比纯逆频率温和，避免仅 2 个样本的稀有动作权重爆炸、主导训练。无样本动作权重=0。
    """
    counts = compute_action_stats(cache, num_actions).astype(np.float64)
    weights = np.zeros(num_actions, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = (1.0 / counts[nonzero]) ** temper
    # 归一化到均值 1（在有样本的类别上）
    if weights[nonzero].mean() > 0:
        weights[nonzero] /= weights[nonzero].mean()
    weights = np.clip(weights, 0.0, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(cache: List[TrainSample], num_actions: int,
                           temper: float = 0.5) -> List[float]:
    """计算每个样本的采样权重（供 WeightedRandomSampler），∝ (1/count_action)^temper。"""
    counts = compute_action_stats(cache, num_actions).astype(np.float64)
    weights = []
    for s in cache:
        c = counts[s.action] if 0 <= s.action < num_actions else 0
        weights.append((1.0 / c) ** temper if c > 0 else 0.0)
    return weights


def split_by_recording(cache: List[TrainSample], val_ratio: float = 0.2,
                       seed: int = 42) -> Tuple[List[TrainSample], List[TrainSample]]:
    """按录像目录划分 train/val，避免同一录像的帧跨集造成泄漏。"""
    video_dirs = sorted({s.video_dir for s in cache})
    rng = random.Random(seed)
    rng.shuffle(video_dirs)
    n_val = max(1, int(len(video_dirs) * val_ratio))
    val_dirs = set(video_dirs[:n_val])
    train = [s for s in cache if s.video_dir not in val_dirs]
    val = [s for s in cache if s.video_dir in val_dirs]
    logger.info(f"按录像划分: 训练 {len(train)} 样本 / 验证 {len(val)} 样本 "
                f"(录像 {len(video_dirs)-n_val}/{n_val})")
    return train, val


# ======================================================================
# 数据增强（BC 用；对整段 10 帧栈施加一致变换）
# ======================================================================
def augment_state(state: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """对 (T, H, W) float32 [0,1] 序列施加一致的随机增强。

    禁止水平翻转（会污染左右转向标签）。所有帧共用同一组随机参数，保持时序一致。
    """
    out = state.copy()

    # 1) 亮度 + 对比度：out = (x - 0.5) * contrast + 0.5 + brightness
    if rng.rand() < 0.8:
        contrast = 1.0 + rng.uniform(-0.2, 0.2)
        brightness = rng.uniform(-0.1, 0.1)
        out = (out - 0.5) * contrast + 0.5 + brightness

    # 2) gamma 校正
    if rng.rand() < 0.5:
        gamma = rng.uniform(0.8, 1.25)
        out = np.clip(out, 0.0, 1.0) ** gamma

    # 3) 平移抖动（±dx, ±dy，整段一致），用 0 padding
    if rng.rand() < 0.5:
        max_shift = 8
        dx = rng.randint(-max_shift, max_shift + 1)
        dy = rng.randint(-max_shift, max_shift + 1)
        out = np.roll(out, shift=(dy, dx), axis=(1, 2))
        if dy > 0:
            out[:, :dy, :] = 0
        elif dy < 0:
            out[:, dy:, :] = 0
        if dx > 0:
            out[:, :, :dx] = 0
        elif dx < 0:
            out[:, :, dx:] = 0

    # 4) 高斯噪声
    if rng.rand() < 0.4:
        out = out + rng.randn(*out.shape).astype(np.float32) * 0.02

    # 5) cutout：随机遮挡一个方块（整段同位置）
    if rng.rand() < 0.3:
        h, w = out.shape[1], out.shape[2]
        ch, cw = h // 8, w // 8
        cy = rng.randint(0, h - ch)
        cx = rng.randint(0, w - cw)
        out[:, cy:cy + ch, cx:cx + cw] = 0

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ======================================================================
# BC 数据集
# ======================================================================
class BCDataset(Dataset):
    """行为克隆数据集：state (T, H, W) -> action。"""

    def __init__(self, cache: List[TrainSample], img_size: int = 640,
                 continue_num: int = 10, gap_num: int = 1,
                 augment: bool = False, seed: int = 0):
        """
        Args:
            cache: TrainSample 列表
            img_size: 帧尺寸（letterbox 目标）
            continue_num: state 帧数
            gap_num: state 末帧与 ctrl_a_frame 之间的间隔帧数
            augment: 是否启用数据增强（训练集 True，验证集 False）
        """
        self.cache = cache
        self.img_size = img_size
        self.continue_num = continue_num
        self.gap_num = gap_num
        self.augment = augment
        # 每个 worker 独立的随机源，保证增强可复现且不同 worker 不同步
        self._base_seed = seed

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        s = self.cache[idx]
        # state 末帧 = ctrl_a_frame - gap_num（动作发生前 gap 帧），符合因果
        end_frame = s.ctrl_a_frame - self.gap_num
        state = _load_state_frames(s.video_dir, end_frame, self.continue_num, self.img_size)
        if state is None:
            return None, None  # 由 collate 过滤

        if self.augment:
            rng = np.random.RandomState((self._base_seed + idx) % (2 ** 31))
            state = augment_state(state, rng)

        return torch.from_numpy(state), torch.tensor(s.action, dtype=torch.long)


def bc_collate_fn(batch):
    """过滤掉加载失败的 None 样本并堆叠。"""
    valid = [(s, a) for s, a in batch if s is not None]
    if not valid:
        return None, None
    states, actions = zip(*valid)
    return torch.stack(states), torch.stack(actions)


# ======================================================================
# 离线 RL 转移数据集
# ======================================================================
class TransitionDataset(Dataset):
    """离线 RL 数据集：单步转移 (s, a, r, s', done)。

    由于本数据无真正 episode 结构，next_state 仅取自同段录像里动作之后的几帧，
    因此固定 done=1、按「单步奖励回归 + 保守正则」处理，不做跨步 bootstrap。
    """

    def __init__(self, cache: List[TrainSample], img_size: int = 640,
                 continue_num: int = 10, gap_num: int = 1,
                 reward_mean: float = 0.0, reward_std: float = 1.0):
        self.cache = cache
        self.img_size = img_size
        self.continue_num = continue_num
        self.gap_num = gap_num
        self.reward_mean = reward_mean
        self.reward_std = reward_std

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        s = self.cache[idx]
        # s1：动作前的连续帧
        s1_end = s.ctrl_a_frame - self.gap_num
        s1 = _load_state_frames(s.video_dir, s1_end, self.continue_num, self.img_size)
        # s2：动作后的连续帧
        s2_start = s.ctrl_a_frame + self.gap_num + 1
        s2 = _load_state_frames(s.video_dir, s2_start + self.continue_num,
                                self.continue_num, self.img_size)
        if s1 is None or s2 is None:
            return None, None, None, None, None

        reward_norm = (float(s.reward_sum) - self.reward_mean) / self.reward_std
        return (
            torch.from_numpy(s1),
            torch.tensor(s.action, dtype=torch.long),
            torch.tensor(reward_norm, dtype=torch.float32),
            torch.from_numpy(s2),
            torch.tensor(1.0, dtype=torch.float32),  # 单步：done 恒为 1
        )


def transition_collate_fn(batch):
    """过滤 None 并堆叠转移样本。"""
    valid = [b for b in batch if b[0] is not None]
    if not valid:
        return None, None, None, None, None
    s1, a, r, s2, done = zip(*valid)
    return (torch.stack(s1), torch.stack(a), torch.stack(r),
            torch.stack(s2), torch.stack(done))

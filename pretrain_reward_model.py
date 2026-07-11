"""
预训练奖励模型（可选，供在线实验模式使用）
==========================================

在离线数据上做监督回归：(state, action) -> 归一化人工奖励。
训练好的奖励模型可在 `--mode online` 时提供奖励估计。

注意：若你只做 BC + 离线 RL（推荐主线），并不需要奖励模型。

用法:
    python pretrain_reward_model.py --epochs 200 --batch-size 16 --gpus 0
"""

import os
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import load_csv_to_cache, compute_reward_stats, _load_state_frames
from rl.reward_model import RewardModel
from utils.config_loader import load_config
from utils.my_logger import logger


ACTION_NUM = 26


class RewardDataset(Dataset):
    """奖励回归数据集：(state, action) -> 归一化奖励。"""

    def __init__(self, cache, img_size=640, continue_num=10, gap_num=1,
                 reward_mean=0.0, reward_std=1.0):
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
        end_frame = s.ctrl_a_frame - self.gap_num
        state = _load_state_frames(s.video_dir, end_frame, self.continue_num, self.img_size)
        if state is None:
            return None, None, None
        r = (float(s.reward_sum) - self.reward_mean) / self.reward_std
        return (torch.from_numpy(state), torch.tensor(s.action, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32))


def reward_collate_fn(batch):
    valid = [(s, a, r) for s, a, r in batch if s is not None]
    if not valid:
        return None, None, None
    s, a, r = zip(*valid)
    return torch.stack(s), torch.stack(a), torch.stack(r)


def _parse_gpu_ids(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def pretrain_reward_model(config_path, epochs=200, batch_size=16, lr=3e-4,
                          save_path="models/reward_model.pth", gpu_ids=None):
    config = load_config(config_path)
    ai = config.get("ai", {})
    paths = config.get("paths", {})
    train = config.get("train", {})
    rl = config.get("rl", {})

    state_frames = ai.get("continue_frames_num", 10)
    frame_size = ai.get("frams_resize", 640)
    records_csv = paths.get("records_csv", "train_data/records.csv")

    if gpu_ids is None:
        gpu_ids = _parse_gpu_ids(train.get("gpu_ids", []))

    cache = load_csv_to_cache(records_csv)
    reward_mean, reward_std = compute_reward_stats(cache)
    logger.info(f"样本={len(cache)}, reward_mean={reward_mean:.4f}, reward_std={reward_std:.4f}")

    dataset = RewardDataset(cache, img_size=frame_size, continue_num=state_frames,
                            gap_num=1, reward_mean=reward_mean, reward_std=reward_std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=train.get("num_workers", 4), pin_memory=True,
                        collate_fn=reward_collate_fn)

    reward_model = RewardModel(
        input_frames=state_frames, num_actions=ACTION_NUM,
        hidden_dim=rl.get("model_dim", 256),
        temporal_encoder=rl.get("temporal_encoder", "gru"),
        gru_layers=rl.get("gru_layers", 1),
        transformer_layers=rl.get("transformer_layers", 2),
        transformer_heads=rl.get("transformer_heads", 4),
        transformer_dropout=rl.get("transformer_dropout", 0.1),
        lr=lr, reward_mean=reward_mean, reward_std=reward_std, gpu_ids=gpu_ids)

    writer = SummaryWriter("runs/reward_model")
    step = 0
    logger.info(f"开始预训练奖励模型: epochs={epochs}, batch={batch_size}, lr={lr}")
    for epoch in range(1, epochs + 1):
        ep_loss, n = 0.0, 0
        for states, actions, rewards in loader:
            if states is None:
                continue
            loss = reward_model.pretrain_step(states, actions, rewards)
            ep_loss += loss
            n += 1
            step += 1
            if step % 100 == 0:
                writer.add_scalar("reward_model/loss", loss, step)
        if n > 0:
            logger.info(f"Epoch {epoch}/{epochs} avg_loss={ep_loss/n:.6f}")
            writer.add_scalar("reward_model/epoch_loss", ep_loss / n, epoch)
        if epoch % 20 == 0 or epoch == epochs:
            reward_model.save(save_path)

    writer.close()
    logger.info(f"奖励模型预训练完成: {save_path}")
    return reward_model


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="预训练监督式奖励模型")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-path", default="models/reward_model.pth")
    p.add_argument("--gpus", default=None)
    args = p.parse_args()
    pretrain_reward_model(args.config, args.epochs, args.batch_size, args.lr,
                          args.save_path, _parse_gpu_ids(args.gpus) if args.gpus else None)

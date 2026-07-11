import yaml
import os
from utils.my_logger import logger


# 默认配置：与 config.yaml 结构保持一致，缺省时兜底（deep-merge）。
DEFAULT_CONFIG = {
    "app": {"recorder_fps": 60, "train_fps": 10},
    "screen": {"monitor_id": 1, "width": 1920, "height": 1080},
    "ai": {"continue_frames_num": 10, "frams_resize": 640, "enable": True},
    "paths": {
        "model_path": "models/dueling_dqn.pth",
        "records_csv": "train_data/records.csv",
        "tensorboard_dir": "runs/dueling_dqn",
    },
    "train": {
        "batch_size": 16,
        "epochs": 60,
        "save_every": 10,
        "num_workers": 4,
        "pin_memory": True,
        "gpu_ids": [0],
    },
    "inference": {"sleep_ms_when_empty": 100},
    "rl": {
        "gamma": 0.99,
        "lr": 3e-4,
        "grad_clip": 1.0,
        # 网络结构
        "model_dim": 256,
        "temporal_encoder": "gru",
        "gru_layers": 1,
        "transformer_layers": 2,
        "transformer_heads": 4,
        "transformer_dropout": 0.1,
        "head_dropout": 0.3,
        # 目标网络
        "target_update": 100,
        "polyak_tau": 0.005,
        "use_polyak": True,
        # 离线 RL
        "use_double_dqn": True,
        "cql_alpha": 1.0,
        "awr_beta": 1.0,
        "awr_weight_clip": 20.0,
        # LR 调度
        "lr_warmup_steps": 500,
        "lr_min_ratio": 0.01,
        # 探索
        "exploration_method": "epsilon",
        "epsilon_start": 0.15,
        "epsilon_end": 0.02,
        "epsilon_decay_steps": 50000,
        # 推理
        "inference_head": "policy",
        "inference_topk": 3,
        "inference_log_every": 30,
    },
    "bc": {
        "val_ratio": 0.2,
        "augment": True,
        "balanced_sampler": True,
        "class_weight_temper": 0.5,
        "label_smoothing": 0.05,
        "early_stop_patience": 15,
    },
    "offline_rl": {
        "method": "awr",
        "val_ratio": 0.15,
        "warmstart_path": "",
    },
    "record": {"enable": True, "output_dir": "./videos"},
    "online": {"enable": False},
    "log": {"level": "INFO", "file": "./logs/app.log"},
}


def _deep_merge(base: dict, override: dict):
    merged = dict(base)
    for key, value in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path="config/config.yaml"):
    """
    加载 YAML 配置文件
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = _deep_merge(DEFAULT_CONFIG, raw)

    if config["ai"]["continue_frames_num"] <= 0:
        raise ValueError("ai.continue_frames_num must be > 0")
    if config["ai"]["frams_resize"] <= 0:
        raise ValueError("ai.frams_resize must be > 0")
    if config["app"]["train_fps"] <= 0 or config["app"]["recorder_fps"] <= 0:
        raise ValueError("app.train_fps and app.recorder_fps must be > 0")

    logger.info(f"Config loaded from {config_path}")
    return config

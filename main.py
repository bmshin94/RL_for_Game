"""
统一入口（重构版）
==================

训练/使用一个能玩 GTA5 的智能体的完整流程：

    1) preprocess  预处理帧为 .npy（消除训练 I/O 瓶颈）
    2) bc          行为克隆：从人类录像 + 动作标注学「像人一样操作」（主力，先跑通）
    3) offline_rl  离线 RL：从 BC 热启动，用人工奖励标签做 CQL/AWR 精炼
    4) inference   实时推理：截屏 -> 模型决策 -> 键鼠执行（F5 开 / F6 关）
    5) online      在线交互学习（实验性）

命令示例：
    python main.py --mode preprocess
    python main.py --mode bc --epochs 60 --gpus 0
    python main.py --mode offline_rl --offline-method awr --epochs 40 --gpus 0
    python main.py --mode inference
"""

import os
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from rl.agent import RLAgent
from rl.trainer import BCTrainer, OfflineRLTrainer, Trainer
from data_loader import (
    load_csv_to_cache, compute_reward_stats, compute_action_stats,
    compute_class_weights, compute_sample_weights, get_valid_actions,
    split_by_recording, BCDataset, bc_collate_fn,
    TransitionDataset, transition_collate_fn,
)
from utils.config_loader import load_config
from utils.my_logger import logger


# 完整动作空间大小（logic/decision.py 的 ACTIONS 有 26 个）
ACTION_NUM = 26


# ======================================================================
# 配置解析
# ======================================================================
def _parse_gpu_ids(raw):
    """把 '0,1,2' 或 [0,1,2] 解析成 int 列表。"""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    text = str(raw).strip()
    return [int(x.strip()) for x in text.split(",") if x.strip()] if text else []


def _resolve_runtime(config: dict, args) -> dict:
    """把 config + CLI 参数汇总成运行时配置字典。"""
    app = config.get("app", {})
    ai = config.get("ai", {})
    train = config.get("train", {})
    infer = config.get("inference", {})
    paths = config.get("paths", {})
    rl = config.get("rl", {})

    return {
        "mode": args.mode,
        "train_fps": app.get("train_fps", 10),
        "recorder_fps": app.get("recorder_fps", 60),
        "state_frames": ai.get("continue_frames_num", 10),
        "frame_size": ai.get("frams_resize", 640),
        "model_path": args.model_path or paths.get("model_path", "models/dueling_dqn.pth"),
        "records_csv": args.data_csv or paths.get("records_csv", "train_data/records.csv"),
        "log_dir": args.log_dir or paths.get("tensorboard_dir", "runs/dueling_dqn"),
        "batch_size": args.batch_size or train.get("batch_size", 16),
        "epochs": args.epochs or train.get("epochs", 60),
        "save_every": train.get("save_every", 10),
        "num_workers": train.get("num_workers", 4),
        "pin_memory": train.get("pin_memory", True),
        "target_update": rl.get("target_update", 100),
        "train_gpu_ids": (_parse_gpu_ids(args.gpus) if args.gpus is not None
                          else _parse_gpu_ids(train.get("gpu_ids", []))),
        "rl_cfg": rl,
        "bc_cfg": config.get("bc", {}),
        "offline_cfg": config.get("offline_rl", {}),
        "record_enable": config.get("record", {}).get("enable", True),
        "record_output_dir": config.get("record", {}).get("output_dir", "./videos"),
        "inference_sleep_ms": infer.get("sleep_ms_when_empty", 100),
        "online": config.get("online", {}),
    }


def _build_agent(runtime, valid_actions):
    """根据 runtime 与有效动作列表构建 RLAgent。"""
    agent_cfg = dict(runtime["rl_cfg"])
    agent_cfg["gpu_ids"] = runtime["train_gpu_ids"]
    agent_cfg["valid_actions"] = valid_actions
    agent = RLAgent(
        num_actions=ACTION_NUM,
        model_input_dime=runtime["state_frames"],
        target_update=runtime["target_update"],
        config=agent_cfg,
    )
    return agent


def add_suffix(model_path: str, tag) -> str:
    """给 checkpoint 文件名加后缀，如 dueling_dqn_bc_epoch10.pth。"""
    if model_path.endswith(".pth"):
        return f"{model_path[:-4]}_{tag}.pth"
    return f"{model_path}_{tag}"


# ======================================================================
# 模式 1：行为克隆 BC
# ======================================================================
def _macro_f1(y_true, y_pred, labels):
    """计算 macro-F1；若无 sklearn 则用纯 numpy 实现回退。"""
    try:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    except Exception:
        # 纯 numpy 回退：对每个 label 算 F1 再平均
        f1s = []
        for c in labels:
            tp = int(np.sum((y_pred == c) & (y_true == c)))
            fp = int(np.sum((y_pred == c) & (y_true != c)))
            fn = int(np.sum((y_pred != c) & (y_true == c)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            f1s.append(f1)
        return float(np.mean(f1s)) if f1s else 0.0


def run_bc(runtime):
    """行为克隆训练主循环。"""
    bc_cfg = runtime["bc_cfg"]
    cache = load_csv_to_cache(runtime["records_csv"])
    if not cache:
        logger.error("训练缓存为空，请先采集/检查数据。")
        return

    # 有效动作（无样本动作被屏蔽）
    valid_actions = get_valid_actions(cache, ACTION_NUM, min_count=1)

    # 打印动作分布
    counts = compute_action_stats(cache, ACTION_NUM)
    total = int(counts.sum())
    logger.info(f"样本总数={total}")
    for a in range(ACTION_NUM):
        if counts[a] > 0:
            logger.info(f"  动作 {a:02d}: {counts[a]} ({counts[a]/total*100:.1f}%)")

    # 按录像划分 train/val，防泄漏
    val_ratio = bc_cfg.get("val_ratio", 0.2)
    train_cache, val_cache = split_by_recording(cache, val_ratio=val_ratio)

    # 类别权重 & 采样权重（缓解不均衡）
    temper = bc_cfg.get("class_weight_temper", 0.5)
    class_weights = compute_class_weights(train_cache, ACTION_NUM, temper=temper)
    sample_weights = compute_sample_weights(train_cache, ACTION_NUM, temper=temper)
    use_sampler = bc_cfg.get("balanced_sampler", True)
    sampler = (WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                     replacement=True) if use_sampler else None)

    # 数据集/加载器
    train_ds = BCDataset(train_cache, img_size=runtime["frame_size"],
                         continue_num=runtime["state_frames"], gap_num=1,
                         augment=bc_cfg.get("augment", True), seed=0)
    val_ds = BCDataset(val_cache, img_size=runtime["frame_size"],
                       continue_num=runtime["state_frames"], gap_num=1, augment=False)
    train_loader = DataLoader(train_ds, batch_size=runtime["batch_size"],
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=runtime["num_workers"], pin_memory=runtime["pin_memory"],
                              collate_fn=bc_collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=runtime["batch_size"], shuffle=False,
                            num_workers=runtime["num_workers"], pin_memory=runtime["pin_memory"],
                            collate_fn=bc_collate_fn)

    # 构建 agent
    agent = _build_agent(runtime, valid_actions)
    # 若存在同名 checkpoint，非严格加载以继续训练
    if os.path.exists(runtime["model_path"]):
        logger.info(f"继续训练，加载: {runtime['model_path']}")
        agent.load(runtime["model_path"], strict=False)
    agent.set_train_mode()

    # 重建 LR 调度以覆盖总步数（warmup + cosine）
    steps_per_epoch = max(1, len(train_cache) // runtime["batch_size"])
    total_steps = steps_per_epoch * runtime["epochs"]
    agent.scheduler = agent._build_lr_scheduler(total_steps)

    trainer = BCTrainer(agent, class_weights=class_weights,
                        label_smoothing=bc_cfg.get("label_smoothing", 0.05))
    writer = SummaryWriter(runtime["log_dir"])

    best_f1 = -1.0
    patience = bc_cfg.get("early_stop_patience", 15)
    no_improve = 0

    logger.info(f"开始 BC 训练: epochs={runtime['epochs']}, batch={runtime['batch_size']}, "
                f"train={len(train_cache)}, val={len(val_cache)}")

    for epoch in range(1, runtime["epochs"] + 1):
        agent.set_train_mode()
        ep_loss, ep_acc, n = 0.0, 0.0, 0
        for states, actions in train_loader:
            if states is None:
                continue
            loss, acc = trainer.train_batch(states, actions)
            ep_loss += loss
            ep_acc += acc
            n += 1
        if n == 0:
            logger.warning(f"epoch {epoch}: 无有效 batch。")
            continue
        avg_loss, avg_acc = ep_loss / n, ep_acc / n

        # --- 验证：算 macro-F1（对不均衡更有意义）---
        agent.set_eval_mode()
        all_pred, all_true = [], []
        for states, actions in val_loader:
            if states is None:
                continue
            pred, true = trainer.eval_batch(states, actions)
            all_pred.append(pred)
            all_true.append(true)
        if all_pred:
            all_pred = np.concatenate(all_pred)
            all_true = np.concatenate(all_true)
            val_acc = float((all_pred == all_true).mean())
            macro_f1 = _macro_f1(all_true, all_pred, valid_actions)
        else:
            val_acc, macro_f1 = 0.0, 0.0

        writer.add_scalar("bc/train_loss", avg_loss, epoch)
        writer.add_scalar("bc/train_acc", avg_acc, epoch)
        writer.add_scalar("bc/val_acc", val_acc, epoch)
        writer.add_scalar("bc/val_macro_f1", macro_f1, epoch)
        writer.add_scalar("bc/lr", agent.optimizer.param_groups[0]["lr"], epoch)
        logger.info(f"epoch={epoch} loss={avg_loss:.4f} train_acc={avg_acc:.3f} "
                    f"val_acc={val_acc:.3f} val_macroF1={macro_f1:.3f}")

        # --- 保存最优 + 早停 ---
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            no_improve = 0
            agent.save(runtime["model_path"])
            logger.info(f"  新最优 macroF1={best_f1:.3f}，已保存最优模型。")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"早停：{patience} 轮无提升，best_macroF1={best_f1:.3f}")
                break

        if epoch % runtime["save_every"] == 0:
            agent.save(add_suffix(runtime["model_path"], f"bc_epoch{epoch}"))

    writer.close()
    logger.info(f"BC 训练结束，最优 macroF1={best_f1:.3f}，最优模型: {runtime['model_path']}")


# ======================================================================
# 模式 2：离线 RL（CQL / AWR）
# ======================================================================
def run_offline_rl(runtime, args):
    """离线 RL 精炼主循环，从 BC 热启动。"""
    off_cfg = runtime["offline_cfg"]
    method = args.offline_method or off_cfg.get("method", "awr")

    cache = load_csv_to_cache(runtime["records_csv"])
    if not cache:
        logger.error("训练缓存为空。")
        return
    valid_actions = get_valid_actions(cache, ACTION_NUM, min_count=1)
    reward_mean, reward_std = compute_reward_stats(cache)

    train_cache, val_cache = split_by_recording(cache, val_ratio=off_cfg.get("val_ratio", 0.15))

    ds = TransitionDataset(train_cache, img_size=runtime["frame_size"],
                           continue_num=runtime["state_frames"], gap_num=1,
                           reward_mean=reward_mean, reward_std=reward_std)
    loader = DataLoader(ds, batch_size=runtime["batch_size"], shuffle=True,
                        num_workers=runtime["num_workers"], pin_memory=runtime["pin_memory"],
                        collate_fn=transition_collate_fn)

    agent = _build_agent(runtime, valid_actions)

    # 热启动：优先从 BC 最优模型加载（非严格，只对结构一致部分加载）
    warmstart = args.warmstart or off_cfg.get("warmstart_path", runtime["model_path"])
    if warmstart and os.path.exists(warmstart):
        logger.info(f"离线 RL 从 BC 热启动: {warmstart}")
        agent.load(warmstart, strict=False)
    else:
        logger.warning("未找到 BC 热启动权重，将从零开始（不推荐，建议先跑 --mode bc）。")
    agent.set_train_mode()

    steps_per_epoch = max(1, len(train_cache) // runtime["batch_size"])
    total_steps = steps_per_epoch * runtime["epochs"]
    agent.scheduler = agent._build_lr_scheduler(total_steps)

    trainer = OfflineRLTrainer(agent, method=method, target_update=runtime["target_update"])
    writer = SummaryWriter(runtime["log_dir"] + f"_offline_{method}")

    out_path = add_suffix(runtime["model_path"], f"offline_{method}")
    logger.info(f"开始离线 RL（{method}）: epochs={runtime['epochs']}, batch={runtime['batch_size']}")

    global_step = 0
    for epoch in range(1, runtime["epochs"] + 1):
        agent.set_train_mode()
        agg = defaultdict(float)
        n = 0
        for s1, a, r, s2, done in loader:
            if s1 is None:
                continue
            info = trainer.train_batch(s1, a, r, s2, done)
            agg["loss"] += info["loss"]
            agg["td_loss"] += info["td_loss"]
            agg["q_mean"] += info["q_mean"]
            n += 1
            global_step += 1
        if n == 0:
            continue
        for k in agg:
            writer.add_scalar(f"offline/{k}", agg[k] / n, epoch)
        logger.info(f"epoch={epoch} loss={agg['loss']/n:.4f} td={agg['td_loss']/n:.4f} "
                    f"q_mean={agg['q_mean']/n:.4f}")

        if epoch % runtime["save_every"] == 0:
            agent.save(out_path)
            logger.info(f"  已保存: {out_path}")

    agent.save(out_path)
    writer.close()
    logger.info(f"离线 RL 结束，模型: {out_path}")


# ======================================================================
# 模式 3：实时推理（AI 玩游戏）
# ======================================================================
def run_inference(runtime, config, args):
    """实时推理循环：截屏 -> 模型决策 -> 键鼠执行。"""
    from core.bot_controller import BotController
    from core.screen_capture import ScreenCaptureThread
    from core.frame_buffer import FrameBuffer
    from core.recorder import ScreenRecorder
    from core.vision_engine import VisionEngine
    from input.keyboard_controller import KeyboardController
    from input.mouse_controller import MouseControllerGame
    from logic.decision import DecisionEngine

    if not os.path.exists(runtime["model_path"]):
        raise FileNotFoundError(f"模型不存在: {runtime['model_path']}（请先训练）")

    # 推理不需要有效动作列表来构建（mask 会随 checkpoint 的 buffer 一起加载）；
    # 但为保证 mask 一致，这里从数据重建有效动作集。
    valid_actions = ()
    if os.path.exists(runtime["records_csv"]):
        cache = load_csv_to_cache(runtime["records_csv"])
        valid_actions = tuple(get_valid_actions(cache, ACTION_NUM, min_count=1))

    agent = _build_agent(runtime, valid_actions)
    agent.load(runtime["model_path"], strict=False)
    agent.set_eval_mode()

    # 推理用哪个头：BC 权重用 policy，离线 RL 后可用 q
    infer_head = args.infer_head or runtime["rl_cfg"].get("inference_head", "policy")
    trainer = Trainer(agent, infer_head=infer_head)
    logger.info(f"推理头: {infer_head}")

    keyboard = KeyboardController()
    mouse = MouseControllerGame()
    vision = VisionEngine(enable=True, history_len=runtime["state_frames"],
                          out_size=runtime["frame_size"])
    decision = DecisionEngine(keyboard, mouse, agent)
    bot = BotController(vision, decision, trainer)
    bot.start()

    buffer = FrameBuffer()
    capture_thread = ScreenCaptureThread(buffer, config, capture_fps=runtime["train_fps"])
    capture_thread.start()

    recorder = ScreenRecorder(runtime["record_enable"], runtime["record_output_dir"],
                              runtime["recorder_fps"],
                              (config["screen"]["width"], config["screen"]["height"]))

    # 固定决策频率：按 train_fps 节流，决策间隔内保持动作
    decision_interval = 1.0 / max(runtime["train_fps"], 1)
    logger.info(f"推理启动，决策频率={runtime['train_fps']}Hz。按 F5 开启 AI，F6 关闭。")

    last_decision_time = 0.0
    try:
        while True:
            frame = buffer.get()
            if frame is None:
                time.sleep(runtime["inference_sleep_ms"] / 1000.0)
                continue

            now = time.time()
            if now - last_decision_time >= decision_interval:
                bot.handle_frame_for_inference(frame)
                last_decision_time = now

            recorder.write(frame)
    finally:
        capture_thread.stop()
        recorder.release()


# ======================================================================
# 模式 4：预处理
# ======================================================================
def run_preprocess(runtime, args):
    """预处理所有帧为 .npy，消除训练 I/O 瓶颈。"""
    import csv
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from preprocess_frames import preprocess_single_video

    csv_path = runtime["records_csv"]
    video_dirs = set()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            video_dirs.add(os.path.normpath(row["video_dir"]))

    img_size = runtime["frame_size"]
    workers = args.preprocess_workers
    force = args.preprocess_force
    logger.info(f"预处理 {len(video_dirs)} 个目录, img_size={img_size}, workers={workers}, "
                f"模式={'强制覆盖' if force else '增量(跳过已有)'}")

    total_p, total_s = 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(preprocess_single_video, v, img_size, force): v for v in video_dirs}
        for fut in as_completed(futures):
            _, p, s = fut.result()
            total_p += p
            total_s += s
    logger.info(f"预处理完成。processed={total_p}, skipped={total_s}")


# ======================================================================
# 命令行
# ======================================================================
def build_parser():
    p = argparse.ArgumentParser(description="GTA5 强化学习智能体（BC + 离线RL）")
    p.add_argument("--mode", choices=["preprocess", "bc", "offline_rl", "inference", "online"],
                   default="inference", help="运行模式")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--model-path", default=None)
    p.add_argument("--data-csv", default=None)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--gpus", default=None, help="训练 GPU id，如 '0,1'")
    # 离线 RL
    p.add_argument("--offline-method", choices=["cql", "awr"], default=None, help="离线 RL 方法")
    p.add_argument("--warmstart", default=None, help="离线 RL 热启动 checkpoint 路径")
    # 推理
    p.add_argument("--infer-head", choices=["policy", "q"], default=None, help="推理使用的头")
    # 预处理
    p.add_argument("--preprocess-workers", type=int, default=8)
    p.add_argument("--preprocess-force", action="store_true", default=False)
    return p


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    runtime = _resolve_runtime(config, args)

    if runtime["mode"] == "preprocess":
        run_preprocess(runtime, args)
    elif runtime["mode"] == "bc":
        run_bc(runtime)
    elif runtime["mode"] == "offline_rl":
        run_offline_rl(runtime, args)
    elif runtime["mode"] == "online":
        from online_train import run_online
        run_online(runtime, config)
    else:
        run_inference(runtime, config, args)


if __name__ == "__main__":
    main()

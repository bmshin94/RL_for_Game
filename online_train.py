"""
在线交互学习（实验性，精简版）
==============================

【重要】在线自玩需要「真实奖励信号」。本项目没有游戏内存读数等外部奖励，只能用一个
预训练的监督式奖励模型来估计奖励。这在原理上弱于 BC + 离线 RL，仅作实验保留。

流程：
    截屏 -> 构建 state -> 选动作(带探索) -> 执行 -> 用奖励模型估计奖励
    -> 存入回放池 -> 周期性从回放池训练 DQN（CQL/AWR）

与旧版的关键区别：**移除了「奖励模型用 Q 值做 TD 一致性自更新」的循环逻辑**，
奖励模型此处仅作只读的奖励估计器（如需更新请离线用 pretrain_reward_model.py）。

用法:
    python main.py --mode online
"""

import os
import time

import keyboard as kb_hotkey
from torch.utils.tensorboard import SummaryWriter

from rl.agent import RLAgent
from rl.trainer import OnlineTrainer
from rl.reward_model import RewardModel
from data_loader import load_csv_to_cache, compute_reward_stats, get_valid_actions
from utils.my_logger import logger


ACTION_NUM = 26


def run_online(runtime, config):
    from core.screen_capture import ScreenCaptureThread
    from core.frame_buffer import FrameBuffer
    from core.vision_engine import VisionEngine
    from core.recorder import ScreenRecorder
    from input.keyboard_controller import KeyboardController
    from input.mouse_controller import MouseControllerGame
    from logic.decision import DecisionEngine
    from logic.decision import ACTIONS as ACTION_MAP

    online_cfg = runtime["online"]
    rl_cfg = runtime["rl_cfg"]

    # --- 有效动作（用于 mask）---
    valid_actions = ()
    reward_mean, reward_std = 0.0, 1.0
    if os.path.exists(runtime["records_csv"]):
        cache = load_csv_to_cache(runtime["records_csv"])
        valid_actions = tuple(get_valid_actions(cache, ACTION_NUM, min_count=1))
        reward_mean, reward_std = compute_reward_stats(cache)

    # --- 构建 agent（从 BC/离线 RL 模型热启动）---
    agent_cfg = dict(rl_cfg)
    agent_cfg["gpu_ids"] = runtime["train_gpu_ids"]
    agent_cfg["valid_actions"] = valid_actions
    agent = RLAgent(num_actions=ACTION_NUM, model_input_dime=runtime["state_frames"],
                    target_update=runtime["target_update"], config=agent_cfg)
    if os.path.exists(runtime["model_path"]):
        logger.info(f"在线学习从已有模型热启动: {runtime['model_path']}")
        agent.load(runtime["model_path"], strict=False)

    # --- 奖励模型（只读估计器）---
    rm_path = "models/reward_model.pth"
    reward_model = RewardModel(
        input_frames=runtime["state_frames"], num_actions=ACTION_NUM,
        hidden_dim=rl_cfg.get("model_dim", 256),
        temporal_encoder=rl_cfg.get("temporal_encoder", "gru"),
        gru_layers=rl_cfg.get("gru_layers", 1),
        transformer_layers=rl_cfg.get("transformer_layers", 2),
        transformer_heads=rl_cfg.get("transformer_heads", 4),
        reward_mean=reward_mean, reward_std=reward_std)
    if os.path.exists(rm_path):
        reward_model.load(rm_path)
    else:
        logger.warning(f"未找到奖励模型 {rm_path}，奖励估计将不可靠。"
                       "建议先运行 python pretrain_reward_model.py")

    # --- 回放训练器 ---
    state_shape = (runtime["state_frames"], runtime["frame_size"], runtime["frame_size"])
    trainer = OnlineTrainer(
        agent=agent, buffer_capacity=online_cfg.get("buffer_capacity", 2000),
        min_buffer_size=online_cfg.get("min_buffer_size", 64),
        train_every=online_cfg.get("train_every", 8),
        batch_size=online_cfg.get("batch_size", 8),
        use_per=online_cfg.get("use_per", True),
        per_alpha=online_cfg.get("per_alpha", 0.6),
        per_beta_start=online_cfg.get("per_beta_start", 0.4),
        per_beta_end=online_cfg.get("per_beta_end", 1.0),
        per_beta_anneal_steps=online_cfg.get("per_beta_anneal_steps", 100000),
        normalize_reward=False, state_shape=state_shape,
        per_priority_mode=online_cfg.get("per_priority_mode", "proportional"),
        offline_method=online_cfg.get("offline_method", "cql"))

    buffer_path = online_cfg.get("save_buffer_path", "")
    if buffer_path and os.path.exists(buffer_path):
        trainer.load_buffer(buffer_path)

    # --- 游戏交互组件 ---
    keyboard = KeyboardController()
    mouse = MouseControllerGame()
    vision = VisionEngine(enable=True, history_len=runtime["state_frames"],
                          out_size=runtime["frame_size"])
    decision = DecisionEngine(keyboard, mouse, agent)

    buffer = FrameBuffer()
    capture_thread = ScreenCaptureThread(buffer, config, capture_fps=runtime["train_fps"])
    capture_thread.start()
    recorder = ScreenRecorder(runtime["record_enable"], runtime["record_output_dir"],
                              runtime["train_fps"],
                              (config["screen"]["width"], config["screen"]["height"]))
    writer = SummaryWriter(runtime["log_dir"] + "_online")

    # --- F5/F6 热键 ---
    state = {"ai": False, "save": False}

    def enable_ai():
        state["ai"] = True
        logger.info("F5 → AI 开启")

    def disable_ai():
        state["ai"] = False
        state["save"] = True
        logger.info("F6 → AI 关闭，将保存模型")

    kb_hotkey.add_hotkey("f5", enable_ai)
    kb_hotkey.add_hotkey("f6", disable_ai)
    logger.info("在线学习启动。F5 开启 / F6 关闭。")

    action_interval = online_cfg.get("action_interval_seconds", 0.2)
    last_action_time = 0.0
    prev_state, prev_action = None, None

    try:
        while True:
            if state["save"]:
                state["save"] = False
                agent.save(runtime["model_path"])
                logger.info(f"模型已保存: {runtime['model_path']}")

            frame = buffer.get()
            if frame is None:
                time.sleep(runtime["inference_sleep_ms"] / 1000.0)
                continue

            current_state = vision.process(frame)
            if current_state is None:
                recorder.write(frame)
                continue

            if action_interval <= 0 or time.time() - last_action_time >= action_interval:
                # 用上一步 (s,a) 与当前 s' 构成转移，奖励由只读奖励模型估计
                if prev_state is not None and prev_action is not None:
                    reward = reward_model.predict(prev_state, prev_action)
                    trainer.observe(current_state, reward, done=False)
                    result = trainer.maybe_train()
                    if result:
                        writer.add_scalar("online/loss", result["loss"], trainer.train_steps)
                        writer.add_scalar("online/q_mean", result["q_mean"], trainer.train_steps)
                        extra = online_cfg.get("train_steps_per_trigger", 1) - 1
                        if extra > 0:
                            trainer.force_train(extra)
                    writer.add_scalar("online/predicted_reward", reward, trainer.env_steps)

                if state["ai"]:
                    action = trainer.act(current_state, train=True)
                    detail = ACTION_MAP.get(action, {}).get("detail", "")
                    logger.info(f"Agent 动作: {action} ({detail})")
                    decision.execute_action(action)
                else:
                    action = 0

                prev_state, prev_action = current_state, action
                last_action_time = time.time()

            writer.add_scalar("online/buffer_size", trainer.buffer_size, trainer.env_steps)
            recorder.write(frame)
    except KeyboardInterrupt:
        logger.info("在线学习被用户中断。")
    finally:
        agent.save(runtime["model_path"])
        if buffer_path:
            trainer.save_buffer(buffer_path)
        capture_thread.stop()
        recorder.release()
        writer.close()
        logger.info(f"在线学习结束。env_steps={trainer.env_steps}, train_steps={trainer.train_steps}")

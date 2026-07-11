import threading
import keyboard
from utils.my_logger import logger
from logic.decision import ACTIONS


class BotController:
    """
    负责：
    - 运行态快捷键监听
    - AI 决策开关状态
    - 每帧是否执行 vision + decision
    """

    def __init__(self, vision_engine, decision_engine, trainer):
        self.vision = vision_engine
        self.decision = decision_engine
        self.trainer = trainer

        self._ai_enabled = False
        self._lock = threading.Lock()
        self._running = False

    # ---------- 状态控制 ----------

    def enable_ai(self):
        with self._lock:
            self._ai_enabled = True
        logger.info("F5 pressed → AI 决策开启")

    def disable_ai(self):
        with self._lock:
            self._ai_enabled = False
            self.trainer.reset()
        logger.info("F6 pressed → AI 决策关闭")

    def is_ai_enabled(self):
        with self._lock:
            return self._ai_enabled

    # ---------- 热键监听 ----------

    def _hotkey_loop(self):
        keyboard.add_hotkey("f5", self.enable_ai)
        keyboard.add_hotkey("f6", self.disable_ai)

        logger.info("BotController hotkeys ready: [F5=ON, F6=OFF]")
        keyboard.wait()  # 阻塞在子线程

    # ---------- 生命周期 ----------

    def start(self):
        if self._running:
            return

        self._running = True
        t = threading.Thread(
            target=self._hotkey_loop,
            daemon=True
        )
        t.start()

    # ---------- 主循环接口 ----------

    def handle_frame_for_inference(self, frame):
        """处理一帧用于推理：仅在 AI 开启时，构建时序 state、模型选动作、后台执行。

        - vision.process(frame) 累积历史帧并返回 (T, H, W) 的归一化时序 state。
        - trainer.step_only_inference 走 agent 的 policy/q 头选出动作 id。
        - decision.execute_action 把动作放入后台队列异步执行，不阻塞采集循环。
        """
        if not self.is_ai_enabled():
            return
        vision_result = self.vision.process(frame)

        action = self.trainer.step_only_inference(vision_result)
        key_detail = ACTIONS.get(action, {}).get("detail")
        logger.info(f"选择动作 action_id={action}, 详情={key_detail}")
        self.decision.execute_action(action)





# RL for Game （重构版：行为克隆 BC + 离线强化学习）

基于**行为克隆 + 离线强化学习**的 GTA5 自动游玩框架。智能体通过观察屏幕像素（连续多帧灰度图）
输出键盘/鼠标动作来学习玩游戏。核心网络为「逐帧 CNN + 时序编码（GRU/Transformer）+ 三头（策略头 / Dueling Q 双头）」。

![image](imgs/title.png)

## AI（RL）玩 GTA5
![AI（RL）玩GTA5](imgs/output.gif)

---

## 🔧 重构说明（务必先读）

本项目已从「循环奖励模型 + 在线 DQN」重构为更可靠的 **行为克隆(BC) → 离线强化学习(CQL/AWR)**，
并对网络结构做了大幅优化。

**为什么改训练范式**：旧的在线模式中，奖励模型去拟合 `Q(s,a) - γ·maxQ(s')`，而 DQN 又用奖励模型的
输出当奖励来训练 Q。二者互相拟合对方、没有任何外部真实信号锚定，构成循环自证，理论上会漂移/塌缩，
**学不出有意义的游戏行为**。新方案先用人类动作做行为克隆（监督学习），再用人工奖励标签做保守的离线 RL 精炼。

**网络结构优化**（详见文末「模型架构」）：

| 项目 | 旧版 | 新版 | 原因 |
|---|---|---|---|
| 训练范式 | 循环奖励模型 + 在线 DQN | **BC → 离线RL(CQL/AWR)** | 循环自证学不出东西 |
| 归一化 | BatchNorm | **GroupNorm** | RL 小 batch / batch=1 推理时 BN 统计量失真 |
| 特征维度 | model_dim=1024 | **256** | 匹配数据量，防过拟合 |
| 时序编码 | 8 层 Transformer + 1D-CNN 双路 | **单层 GRU**（可切 Transformer） | 过度设计，10 帧序列不需要 |
| 参数量 | 数千万 | **~0.65M** | 匹配千级样本、保证实时 |
| 大分辨率实时 | 640→80 全量重算 | **逐帧 CNN + 推理 embedding 缓存** | 每步只算 1 新帧 |
| 无数据动作 | 26 个全参与 | **屏蔽无样本动作** | 避免误选与 Q 值高估 |

**⚠️ 重要**：旧的 `models/*.pth` 与新网络结构不兼容。加载时会以 `strict=False` 跳过失配权重、
从零开始训练——这是预期行为，**你需要用新代码重新训练**。

---

## 快速开始（推荐完整流程）

```bash
# 1) 预处理：把 .jpg 帧转成归一化 .npy，消除训练 I/O 瓶颈（增量，已处理的自动跳过）
python main.py --mode preprocess

# 2) 行为克隆 BC（主力）：从人类录像 + 动作标注学「像人一样操作」
python main.py --mode bc --epochs 60 --gpus 0

# 3) 离线 RL 精炼（可选）：从 BC 权重热启动，用人工奖励标签做 AWR/CQL 精炼
python main.py --mode offline_rl --offline-method awr --gpus 0

# 4) 实时游玩：截屏 → 模型决策 → 键鼠执行（F5 开启 AI / F6 关闭）
python main.py --mode inference
```

训练过程可用 TensorBoard 观察：
```bash
tensorboard --logdir runs
```
BC 重点看 `bc/val_macro_f1`（对不均衡动作比准确率更有意义），离线 RL 看 `offline/loss` 与 `offline/q_mean`。

---

## 环境要求

Python 3.8+，核心依赖（详见 [requirements.txt](requirements.txt)）：

| 包 | 用途 |
|---------|---------|
| torch / torchvision | 网络、优化器、AMP |
| opencv-python | 帧读写、缩放、letterbox |
| mss | 屏幕捕获 |
| pynput | 键盘模拟 |
| pyyaml | 配置加载 |
| tensorboard | 训练指标日志 |
| termcolor | 彩色控制台输出 |
| keyboard | 全局热键（F5/F6、Ctrl+Shift） |
| scikit-learn | （可选）BC 验证的 macro-F1；缺失时自动用内置 numpy 回退 |

---

## 运行模式详解

`main.py` 通过 `--mode` 选择模式，命令行参数会覆盖 `config/config.yaml` 中的对应值。

### 模式 1：`preprocess` — 帧预处理

将每段录像下的 `.jpg` 灰度化、letterbox resize 到 640×640、归一化后存为 `.npy`，
训练时直接加载 `.npy`，避免反复解码/缩放的 I/O 开销。

```bash
python main.py --mode preprocess
python main.py --mode preprocess --preprocess-force      # 强制覆盖已有 .npy
python main.py --mode preprocess --preprocess-workers 8  # 并行进程数
```

### 模式 2：`bc` — 行为克隆（主力）

监督学习 `state（连续 10 帧）→ 人类动作`。这是让智能体「先动起来、像人操作」的关键一步。

```bash
python main.py --mode bc --epochs 60 --gpus 0
python main.py --mode bc --epochs 100 --batch-size 32 --gpus 0,1   # 多卡
```

内部处理（对应 `bc` 配置段）：
- **按录像划分** train/val（同一段录像的帧不会同时进训练集和验证集，避免信息泄漏、指标虚高）。
- **类别不均衡缓解**：`WeightedRandomSampler` 逆频率加权采样 + 加权交叉熵，权重用 `1/√freq`（sqrt-tempered，
  避免仅几个样本的稀有动作权重爆炸）。
- **数据增强**（对同一 10 帧栈施加一致变换）：亮度/对比度/gamma 抖动、平移、高斯噪声、cutout。
  **不做水平翻转**（会翻转左右转向语义、污染标签）。
- **正则**：AdamW + weight_decay、输出头 dropout、label smoothing。
- **早停**：按验证集 `macro-F1` 保存最优模型（存到 `paths.model_path`），连续 `early_stop_patience` 轮无提升则停。
- **无样本动作**：日志会列出并屏蔽（如「以下 N 个动作没有样本，将被屏蔽」）。

### 模式 3：`offline_rl` — 离线强化学习精炼（可选）

从 BC 权重**热启动**，用人工奖励标签（`reward_sum`）做保守的离线 RL 精炼。

```bash
python main.py --mode offline_rl --offline-method awr --gpus 0   # 推荐 AWR
python main.py --mode offline_rl --offline-method cql --gpus 0   # 或 CQL
python main.py --mode offline_rl --offline-method awr --warmstart models/dueling_dqn.pth
```

两种方法：
- **`awr`（优势加权行为克隆，推荐）**：用优势 `A(s,a)` 对 BC 交叉熵加权（`w = exp(A/β)` 截断）。
  对「无真正 episode 结构」的断裂数据更鲁棒。
- **`cql`（保守 Q 学习）**：在 TD 误差上加 `α·(logsumexp_a Q(s,a) − Q(s,a_data))`，压低未见动作的 Q 值，
  直接修复「离线 DQN 高估 OOD 动作」的问题。

> 说明：本数据没有真正的 episode 结构，`next_state` 仅取自同段录像里动作之后的几帧，因此离线 RL 采用
> **单步设定**（`done=1`），不通过伪造的 `next_state` 做跨步 bootstrap。输出模型带 `_offline_awr` / `_offline_cql` 后缀。

### 模式 4：`inference` — 实时游玩

截屏 → 构建时序 state → 模型选动作 → 后台线程执行键鼠，按 `train_fps` 固定频率决策。

```bash
python main.py --mode inference
python main.py --mode inference --infer-head q   # 用 Q 头推理（离线RL后）；默认 policy 头（BC后）
```

| 热键 | 功能 |
|--------|--------|
| **F5** | 启用 AI 智能体 |
| **F6** | 禁用 AI，清空推理缓存 |

实时性优化：`torch.inference_mode()`、逐帧 CNN embedding 缓存、GroupNorm（保证 batch=1 正确）、
决策与截屏线程解耦。

### 模式 5：`online` — 在线交互学习（⚠️ 实验性）

```bash
python main.py --mode online
```

需要预训练的监督式奖励模型（`models/reward_model.pth`）作为**只读**奖励估计器。
智能体一边玩一边把转移存入回放池并训练 DQN。

> **实验性且不推荐作为主线**：在线自玩需要真实奖励信号，而本项目没有游戏内存读数等外部奖励，
> 只能用离线拟合的奖励模型估计，原理上弱于 BC + 离线 RL。**已彻底移除旧版的循环 TD 自更新**，
> 奖励模型此处仅做只读估计，不再自我引用 Q 值。

预训练奖励模型（仅在使用 online 模式时需要）：
```bash
python pretrain_reward_model.py --epochs 200 --batch-size 16 --gpus 0
```

---

## 数据采集与下载

### 采集
```bash
python data_collection.py
```
以 `train_fps` 录制屏幕：
- **Ctrl+Shift** — 标记当前时刻，继续录制 N 秒后弹出标注对话框
- **标注对话框** — 选择执行的动作与观察到的奖励，**Ctrl+S** 保存

输出：`train_data/saved_videos/<时间戳>/` 帧目录 + `train_data/records.csv` 索引条目。

### 下载现成数据集（扩充训练数据）
```
https://huggingface.co/datasets/zhirui001/RL_for_Game_Dataset
```

---

## 项目结构

```
rl_for_game/
├── main.py                  # 统一入口（preprocess / bc / offline_rl / inference / online）
├── data_collection.py       # 屏幕录制 + Tkinter 标注工具
├── data_loader.py           # BCDataset / TransitionDataset、类别均衡、按录像划分、数据增强
├── online_train.py          # 在线交互学习循环（实验性，只读奖励模型）
├── pretrain_reward_model.py # 监督式奖励模型预训练（仅 online 模式需要）
├── preprocess_frames.py     # .jpg -> .npy 预处理
├── config/
│   └── config.yaml          # 统一配置（app/screen/ai/rl/bc/offline_rl/online/...）
├── core/
│   ├── bot_controller.py    # 热键监听 + AI 决策循环
│   ├── frame_buffer.py      # 线程安全帧缓冲
│   ├── recorder.py          # 推理视频录制（XVID avi）
│   ├── screen_capture.py    # 后台屏幕捕获线程
│   └── vision_engine.py     # 单帧处理 + 时序 state 组合
├── input/
│   ├── keyboard_controller.py  # pynput 键盘输入
│   └── mouse_controller.py     # Win32 SendInput 鼠标输入
├── logic/
│   └── decision.py          # 动作 ID -> 键鼠执行映射（ACTIONS 共 26 个）
├── rl/
│   ├── agent.py             # GamePolicyNet（逐帧CNN+时序+三头）+ RLAgent（BC/CQL/AWR）
│   ├── trainer.py           # BCTrainer / OfflineRLTrainer / OnlineTrainer
│   ├── replay_buffer.py     # 统一回放 + 优先经验回放 PER（SumTree）
│   └── reward_model.py      # 监督式奖励模型（实验性，已停用循环 TD）
├── utils/
│   ├── config_loader.py     # YAML 加载（深度合并 + 默认值 + 校验）
│   └── my_logger.py         # 彩色控制台 + 每日文件日志
├── models/                  # 模型检查点
├── train_data/              # 训练数据（records.csv + saved_videos/）
└── videos/                  # 推理录制视频
```

---

## 配置说明

所有配置集中在 [config/config.yaml](config/config.yaml)，命令行参数覆盖对应项。

| 配置段 | 关键参数 |
|---------|---------------|
| `app` | `recorder_fps`、`train_fps`（决策频率） |
| `screen` | `monitor_id`、`width`、`height` |
| `ai` | `continue_frames_num`（状态帧数=10）、`frams_resize`（=640） |
| `paths` | `model_path`、`records_csv`、`tensorboard_dir` |
| `train` | `batch_size`、`epochs`、`save_every`、`num_workers`、`gpu_ids` |
| `rl` | `lr`、`model_dim`、`temporal_encoder`(gru/transformer)、`use_double_dqn`、`cql_alpha`、`awr_beta`、`inference_head` |
| `bc` | `val_ratio`、`augment`、`balanced_sampler`、`class_weight_temper`、`label_smoothing`、`early_stop_patience` |
| `offline_rl` | `method`(awr/cql)、`val_ratio`、`warmstart_path` |
| `inference` | `sleep_ms_when_empty` |
| `online` | `buffer_capacity`、`use_per`、`offline_method`（实验性） |
| `record` | `enable`、`output_dir` |
| `log` | `level`、`file` |

---

## 模型架构

**GamePolicyNet**：单网络、三头共享主干，一个 checkpoint 同时用于 BC 与离线 RL。

```
输入: (B, T=10, 640, 640) 灰度序列, 归一化到 [0,1]
  │
  ├─ 逐帧 CNN 主干 FrameEncoder（每帧权重共享；GroupNorm）
  │    stem 7x7 s4        → 160×160 (32ch)
  │    深度可分离+残差 s2 → 80×80   (64ch)
  │    深度可分离+残差 s2 → 40×40   (96ch)
  │    深度可分离+残差 s2 → 20×20   (128ch)
  │    深度可分离+残差 s2 → 10×10   (192ch)
  │    AdaptiveAvgPool → Linear → 256   # 每帧 embedding（空间下采样 64×，实时关键）
  │
  ├─ 时序编码 TemporalEncoder（10 个 256-d token）
  │    默认: 单层单向 GRU(256)，取最后隐状态（因果、轻量）
  │    可选: 2 层轻量 Transformer + 注意力池化（config 切换）
  │
  └─ 256-d 状态特征 → 三个头:
       ├─ policy_head : 256→128→26 logits     # BC 用（交叉熵）
       ├─ value_stream: 256→128→1             # Dueling
       └─ adv_stream  : 256→128→26            # Dueling
          Q(s,a) = V(s) + A(s,a) − mean(A)     # 离线 RL 用
```

**关键技术**：
- **逐帧 CNN + 推理 embedding 缓存**：10 帧窗口每步只前移 1 帧，推理时只对新帧跑一次 CNN，其余 9 帧
  embedding 从环形缓存复用 —— 这是 640×640 大分辨率仍能实时的核心。
- **GroupNorm 替代 BatchNorm**：与 batch 大小无关，batch=1 推理、小 batch 训练、target 网络均正确。
- **动作屏蔽**：无样本动作的 logits/Q 强制置 `-inf`，softmax 概率与 argmax 永不选中它们。
- **三头共享主干 + 热启动**：离线 RL 直接从 BC checkpoint 加载 backbone/时序/policy 头。
- **Dueling + Double DQN + Polyak 软同步**：离线 RL 端沿用，减小 Q 值过估计。

---

## 完整训练流程回顾

1. **录制与标注** — `data_collection.py` 捕获帧并保存动作/奖励标注到 `records.csv`
2. **预处理** — `.jpg → .npy`（灰度 + letterbox + 归一化）
3. **行为克隆 BC** — 监督学习 state→动作，按录像划分 + 类别均衡 + 增强 + 早停（主力）
4. **离线 RL 精炼**（可选）— 从 BC 热启动，AWR/CQL + 单步保守 Q
5. **实时游玩** — 加载模型，截屏→决策→键鼠执行（F5/F6）
6. **在线学习**（可选，实验性）— 只读奖励模型 + 回放池训练

---

## 已知限制（决定「能否真正玩好」）

这次重构把**方法学与网络**从「学不出东西」修到了「能正确学习并实时运行」。但要让智能体在 GTA5 里
表现好，仍受以下现实条件制约：

1. **数据量是根本瓶颈**：当前约 563 条标注、26 个动作里仅约 11 个有样本。BC 能让智能体动起来，
   但要玩得好需要更多数据。**最高杠杆的下一步**：改造采集工具在录制时逐帧记录真实键鼠状态自动打标，
   把每段录像从 1 个样本变成数十个样本。
2. **26 个离散动作对 GTA 偏粗**：鼠标视角被离散化得过粗，长期需要**混合动作空间**（离散键 + 连续/细分
   鼠标 dx/dy）与**按键时长语义**（开车需「按住 W」而非点按）。
3. **BC 协变量漂移**：BC 会漂进没见过的状态且难恢复，需更多数据 / DAgger 纠偏 / 离线 RL 的奖励加权来缓解。
4. **评估**：验证集 macro-F1 ≠ 会玩。真正的评估需要游戏内 rollout 指标（存活时长 / 行驶距离 / 任务完成）。

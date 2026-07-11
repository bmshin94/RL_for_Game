# RL for Game (Behavioral Cloning + Offline RL)

A framework for learning to play GTA5 via **Behavioral Cloning + Offline Reinforcement Learning**.
The agent observes screen pixels (a stack of grayscale frames) and outputs keyboard/mouse actions.
The core network is "per-frame CNN + temporal encoder (GRU/Transformer) + three heads (policy head / dueling Q streams)".

![image](imgs/title.png)

## AI (RL) plays GTA5
<div align="center">
  <img src="imgs/output.gif" alt="AI (RL) plays GTA5" width="640">
</div>

---

## 🔧 Refactor Notes (read first)

This project was refactored from "circular reward model + online DQN" to the more reliable
**Behavioral Cloning (BC) → Offline RL (CQL/AWR)**, with a heavily optimized network.

**Why change the paradigm**: In the old online mode, the reward model was fit to `Q(s,a) - γ·maxQ(s')`,
while the DQN used the reward model's output as the reward to train Q. The two fit each other with no
external grounding signal — a self-referential loop that drifts/collapses and **cannot learn meaningful
gameplay**. The new approach first clones human actions via supervised learning, then refines with
conservative offline RL using human reward labels.

**Network optimizations** (see "Model Architecture" below):

| Item | Old | New | Reason |
|---|---|---|---|
| Paradigm | Circular reward + online DQN | **BC → Offline RL (CQL/AWR)** | Circular self-reference learns nothing |
| Normalization | BatchNorm | **GroupNorm** | BN stats break at small batch / batch=1 inference |
| Feature dim | model_dim=1024 | **256** | Match data scale, prevent overfitting |
| Temporal enc. | 8-layer Transformer + 1D-CNN dual path | **1-layer GRU** (Transformer optional) | Over-engineered for a 10-frame sequence |
| Params | tens of millions | **~0.65M** | Match ~1k samples, ensure real-time |
| Real-time @640 | full recompute 640→80 | **per-frame CNN + inference embedding cache** | Only 1 new frame per step |
| Zero-data actions | all 26 active | **masked out** | Avoid misfire & Q overestimation |

**⚠️ Important**: Old `models/*.pth` are incompatible with the new architecture. Loading uses
`strict=False` and skips mismatched weights (training from scratch) — this is expected; **retrain with the new code**.

---

## Quick Start (recommended full pipeline)

```bash
# 1) Preprocess: convert .jpg frames to normalized .npy (kills training I/O bottleneck; incremental)
python main.py --mode preprocess

# 2) Behavioral Cloning (primary): learn human-like control from recordings + action labels
python main.py --mode bc --epochs 60 --gpus 0

# 3) Offline RL refinement (optional): warm-start from BC, refine with human reward labels via AWR/CQL
python main.py --mode offline_rl --offline-method awr --gpus 0

# 4) Play in real time: capture → model decision → keyboard/mouse (F5 enable AI / F6 disable)
python main.py --mode inference
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir runs
```
For BC watch `bc/val_macro_f1` (more meaningful than accuracy under imbalance); for offline RL watch
`offline/loss` and `offline/q_mean`.

---

## Requirements

Python 3.8+ (see [requirements.txt](requirements.txt)):

| Package | Purpose |
|---------|---------|
| torch / torchvision | Network, optimizers, AMP |
| opencv-python | Frame I/O, resize, letterbox |
| mss | Screen capture |
| pynput | Keyboard emulation |
| pyyaml | Config loading |
| tensorboard | Training metric logging |
| termcolor | Colored console output |
| keyboard | Global hotkeys (F5/F6, Ctrl+Shift) |
| scikit-learn | (Optional) macro-F1 for BC validation; falls back to a built-in numpy impl if missing |

---

## Modes in Detail

`main.py` selects a mode via `--mode`. CLI args override the matching values in `config/config.yaml`.

### Mode 1: `preprocess` — Frame preprocessing

Grayscale + letterbox to 640×640 + normalize each `.jpg` and save as `.npy`, so training loads `.npy`
directly and avoids repeated decode/resize I/O.

```bash
python main.py --mode preprocess
python main.py --mode preprocess --preprocess-force      # overwrite existing .npy
python main.py --mode preprocess --preprocess-workers 8  # parallel workers
```

### Mode 2: `bc` — Behavioral Cloning (primary)

Supervised learning of `state (10 stacked frames) → human action`. This is the key step to make the
agent "move and act like a human" first.

```bash
python main.py --mode bc --epochs 60 --gpus 0
python main.py --mode bc --epochs 100 --batch-size 32 --gpus 0,1   # multi-GPU
```

Internals (from the `bc` config section):
- **Split by recording** into train/val (frames from one clip never straddle train and val — prevents
  leakage and inflated metrics).
- **Imbalance handling**: `WeightedRandomSampler` inverse-frequency sampling + weighted cross-entropy,
  weights ∝ `1/√freq` (sqrt-tempered so a rare 2-sample action doesn't explode the weights).
- **Augmentation** (consistent across all 10 frames in a stack): brightness/contrast/gamma jitter,
  translation, Gaussian noise, cutout. **No horizontal flip** (would corrupt left/right turn labels).
- **Regularization**: AdamW + weight decay, head dropout, label smoothing.
- **Early stopping**: saves the best model (to `paths.model_path`) by validation `macro-F1`; stops after
  `early_stop_patience` epochs without improvement.
- **Zero-data actions**: logged and masked (e.g. "N actions have no samples, masked out").

### Mode 3: `offline_rl` — Offline RL refinement (optional)

**Warm-starts** from BC weights and refines using human reward labels (`reward_sum`) with conservative offline RL.

```bash
python main.py --mode offline_rl --offline-method awr --gpus 0   # AWR (recommended)
python main.py --mode offline_rl --offline-method cql --gpus 0   # or CQL
python main.py --mode offline_rl --offline-method awr --warmstart models/dueling_dqn.pth
```

Two methods:
- **`awr` (Advantage-Weighted Behavioral Cloning, recommended)**: weights BC cross-entropy by advantage
  `A(s,a)` (`w = exp(A/β)`, clipped). More robust to data with no proper episode structure.
- **`cql` (Conservative Q-Learning)**: adds `α·(logsumexp_a Q(s,a) − Q(s,a_data))` to the TD error,
  pushing down Q on unseen actions — directly fixing offline DQN's OOD overestimation.

> Note: the data has no real episode structure; `next_state` is just a few frames later in the same clip.
> Offline RL therefore uses a **single-step setting** (`done=1`) and does not bootstrap through the
> fabricated `next_state`. Output models get a `_offline_awr` / `_offline_cql` suffix.

### Mode 4: `inference` — Play in real time

Capture → build temporal state → model picks action → background thread executes keyboard/mouse, at a
fixed decision rate (`train_fps`).

```bash
python main.py --mode inference
python main.py --mode inference --infer-head q   # use Q head (after offline RL); default is policy head (after BC)
```

| Hotkey | Action |
|--------|--------|
| **F5** | Enable AI agent |
| **F6** | Disable AI, clear inference cache |

Real-time optimizations: `torch.inference_mode()`, per-frame CNN embedding cache, GroupNorm (correct at
batch=1), decoupled capture/decision threads.

### Mode 5: `online` — Online interactive learning (⚠️ experimental)

```bash
python main.py --mode online
```

Requires a pretrained supervised reward model (`models/reward_model.pth`) as a **read-only** reward
estimator. The agent plays while pushing transitions into a replay buffer and training the DQN.

> **Experimental, not the main path**: online self-play needs a real reward signal, but this project has
> no external reward (e.g. game-memory reads), so it can only estimate reward via an offline-fit model —
> weaker in principle than BC + offline RL. **The old circular TD self-update has been removed**; the
> reward model here is read-only and no longer references Q values.

Pretrain the reward model (only needed for online mode):
```bash
python pretrain_reward_model.py --epochs 200 --batch-size 16 --gpus 0
```

---

## Data Collection & Download

### Collect
```bash
python data_collection.py
```
Records the screen at `train_fps`:
- **Ctrl+Shift** — mark the moment, keep recording N seconds, then open the annotation dialog
- **Annotation dialog** — select performed actions and observed rewards, then **Ctrl+S** to save

Output: `train_data/saved_videos/<timestamp>/` frame dirs + entries in `train_data/records.csv`.

### Download the dataset (to expand training data)
```
https://huggingface.co/datasets/zhirui001/RL_for_Game_Dataset
```

---

## Project Structure

```
rl_for_game/
├── main.py                  # Entry point (preprocess / bc / offline_rl / inference / online)
├── data_collection.py       # Screen recording + Tkinter annotation tool
├── data_loader.py           # BCDataset / TransitionDataset, class balancing, split-by-recording, augmentation
├── online_train.py          # Online interactive loop (experimental, read-only reward model)
├── pretrain_reward_model.py # Supervised reward model pretraining (only for online mode)
├── preprocess_frames.py     # .jpg -> .npy preprocessing
├── config/
│   └── config.yaml          # Unified config (app/screen/ai/rl/bc/offline_rl/online/...)
├── core/
│   ├── bot_controller.py    # Hotkey listener + AI decision loop
│   ├── frame_buffer.py      # Thread-safe frame buffer
│   ├── recorder.py          # Inference video recording (XVID avi)
│   ├── screen_capture.py    # Background screen capture thread
│   └── vision_engine.py     # Single-frame processing + temporal state assembly
├── input/
│   ├── keyboard_controller.py  # Keyboard via pynput
│   └── mouse_controller.py     # Mouse via Win32 SendInput
├── logic/
│   └── decision.py          # Action ID -> keyboard/mouse mapping (26 ACTIONS)
├── rl/
│   ├── agent.py             # GamePolicyNet (per-frame CNN + temporal + 3 heads) + RLAgent (BC/CQL/AWR)
│   ├── trainer.py           # BCTrainer / OfflineRLTrainer / OnlineTrainer
│   ├── replay_buffer.py     # Uniform replay + Prioritized Experience Replay (SumTree)
│   └── reward_model.py      # Supervised reward model (experimental; circular TD removed)
├── utils/
│   ├── config_loader.py     # YAML loader (deep-merge + defaults + validation)
│   └── my_logger.py         # Colored console + daily file logger
├── models/                  # Model checkpoints
├── train_data/              # Training data (records.csv + saved_videos/)
└── videos/                  # Inference recordings
```

---

## Configuration

All settings live in [config/config.yaml](config/config.yaml); CLI args override matching values.

| Section | Key Parameters |
|---------|---------------|
| `app` | `recorder_fps`, `train_fps` (decision rate) |
| `screen` | `monitor_id`, `width`, `height` |
| `ai` | `continue_frames_num` (state frames = 10), `frams_resize` (= 640) |
| `paths` | `model_path`, `records_csv`, `tensorboard_dir` |
| `train` | `batch_size`, `epochs`, `save_every`, `num_workers`, `gpu_ids` |
| `rl` | `lr`, `model_dim`, `temporal_encoder` (gru/transformer), `use_double_dqn`, `cql_alpha`, `awr_beta`, `inference_head` |
| `bc` | `val_ratio`, `augment`, `balanced_sampler`, `class_weight_temper`, `label_smoothing`, `early_stop_patience` |
| `offline_rl` | `method` (awr/cql), `val_ratio`, `warmstart_path` |
| `inference` | `sleep_ms_when_empty` |
| `online` | `buffer_capacity`, `use_per`, `offline_method` (experimental) |
| `record` | `enable`, `output_dir` |
| `log` | `level`, `file` |

---

## Model Architecture

**GamePolicyNet**: a single network with three heads over a shared backbone; one checkpoint serves both
BC and offline RL.

```
Input: (B, T=10, 640, 640) grayscale sequence, normalized to [0,1]
  │
  ├─ Per-frame CNN backbone FrameEncoder (weights shared across frames; GroupNorm)
  │    stem 7x7 s4          → 160×160 (32ch)
  │    depthwise-sep+res s2 → 80×80   (64ch)
  │    depthwise-sep+res s2 → 40×40   (96ch)
  │    depthwise-sep+res s2 → 20×20   (128ch)
  │    depthwise-sep+res s2 → 10×10   (192ch)
  │    AdaptiveAvgPool → Linear → 256   # per-frame embedding (64× spatial reduction — key to real-time)
  │
  ├─ Temporal encoder TemporalEncoder (10 tokens of dim 256)
  │    default: 1-layer unidirectional GRU(256), take last hidden (causal, lightweight)
  │    optional: 2-layer lightweight Transformer + attention pooling (config switch)
  │
  └─ 256-d state feature → three heads:
       ├─ policy_head : 256→128→26 logits     # BC (cross-entropy)
       ├─ value_stream: 256→128→1             # Dueling
       └─ adv_stream  : 256→128→26            # Dueling
          Q(s,a) = V(s) + A(s,a) − mean(A)     # offline RL
```

**Key techniques**:
- **Per-frame CNN + inference embedding cache**: the 10-frame window shifts by one frame per step, so at
  inference only the new frame runs through the CNN while the other 9 embeddings are reused from a ring
  cache — the core reason 640×640 stays real-time.
- **GroupNorm instead of BatchNorm**: independent of batch size — correct at batch=1 inference, small-batch
  training, and for the target network.
- **Action masking**: zero-data actions have their logits/Q forced to `-inf`, so softmax prob and argmax
  never select them.
- **Shared backbone + warm-start**: offline RL loads backbone/temporal/policy head directly from the BC checkpoint.
- **Dueling + Double DQN + Polyak soft update**: retained on the offline-RL side to reduce Q overestimation.

---

## Full Training Pipeline

1. **Record & annotate** — `data_collection.py` captures frames and saves action/reward labels to `records.csv`
2. **Preprocess** — `.jpg → .npy` (gray + letterbox + normalize)
3. **Behavioral Cloning** — supervised state→action, split-by-recording + class balancing + augmentation + early stop (primary)
4. **Offline RL refinement** (optional) — warm-start from BC, AWR/CQL + single-step conservative Q
5. **Play** — load model, capture→decide→execute (F5/F6)
6. **Online learning** (optional, experimental) — read-only reward model + replay training

---

## Known Limitations (what actually gates "playing well")

This refactor fixed the **methodology and network** from "learns nothing" to "learns correctly and runs
in real time". But good in-game performance is still bounded by:

1. **Data volume is the fundamental bottleneck**: ~563 labeled samples, and only ~11 of 26 actions have
   any data. BC gets the agent moving, but playing well needs more data. **Highest-leverage next step**:
   modify the collection tool to log real keyboard/mouse state per frame at record time (auto-labeling),
   turning each recording from 1 sample into dozens.
2. **26 discrete actions are too coarse for GTA**: mouse-look is discretized too coarsely; long term you
   need a **hybrid action space** (discrete keys + continuous/fine-binned mouse dx/dy) and **key-hold
   duration semantics** (driving needs "hold W", not a tap).
3. **BC covariate shift**: BC drifts into unseen states and struggles to recover; mitigate with more data /
   DAgger-style correction / the offline-RL reward weighting.
4. **Evaluation**: validation macro-F1 ≠ playing well. Real evaluation needs in-game rollout metrics
   (time alive / distance driven / task completion).
```

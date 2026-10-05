# DMAST: Dual-Modality Multi-Stage Adversarial Safety Training

Code for the paper **Dual-Modality Multi-Stage Adversarial Safety Training: Robustifying Multimodal Web Agents Against Cross-Modal Attacks**.

DMAST trains a multimodal web agent against an HTML-injection attacker that corrupts both the screenshot and the accessibility tree. The agent and attacker share one VLM and are co-trained in three stages:

1. **Imitation learning** from a stronger teacher
2. **Oracle-guided SFT** with task-focused (zero-acknowledgment) Chain-of-Thought
3. **Adversarial RL** via GRPO self-play

---

## Environment Setup

### MiniWoB++

```bash
git clone git@github.com:Farama-Foundation/miniwob-plusplus.git
git -C "./miniwob-plusplus" reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838
```

BrowserGym looks up this checkout through environment variables set by `utils/bgym.py`.

### Python environment

```bash
bash setup_env.sh
conda activate dmast
```

The script installs PyTorch, a custom vLLM build, Playwright, and the packages in `requirements.txt`.

### Models

```bash
bash ckpt/download_model.sh
```

Paper defaults:

| Role | Model |
| --- | --- |
| Student (trained policy) | Gemma-3-12B-IT |
| Teacher / oracle / eval attacker | Gemma-3-27B-IT |

Place checkpoints under `~/dmast_ckpt/pretrained/`.

---

## Training Pipeline

`python train.py` defaults to the complete paper pipeline: pretrained Gemma-3-12B-IT → imitation SFT → task-focused oracle-guided SFT → shared-weight GRPO self-play. It does **not** take a `--stage` flag. Settings are defined in [`configs/train_adversarial_args.py`](configs/train_adversarial_args.py).

| Main training setting | Default |
| --- | --- |
| Initial model, both roles | `~/dmast_ckpt/pretrained/gemma-3-12b-it` |
| Stage 1 / Stage 2 SFT | Both enabled |
| RL method | `grpo` |
| Shared policy / roles updated | `self_play=True` / both agent and attacker |
| Historical population | Disabled for both roles |
| ART mutation | Disabled |
| RL learning rate / KL coefficient | `2e-6` / `0.05` |
| Persistent cosine scheduler | 25 iterations; 1,200 optimizer-step budget |

Stage switches for configuring experiments:

| Flag | Effect |
| --- | --- |
| `agent.train_imitation` / `attacker.train_imitation` | Run Stage 1 SFT (skipped if that checkpoint already exists) |
| `enhanced_sft.enabled` | After Stage 1, run Stage 2 SFT on the oracle-guided dataset |
| `agent.train_rl` / `attacker.train_rl` | Update these roles with the configured iterative method (GRPO by default) |

Data collection is separate: prepare the two datasets with the commands below, then run `python train.py`. Their output paths match the training defaults. The entry point walks Stage 1 → Stage 2 → Stage 3 in order. Existing nonempty SFT checkpoint directories are reused, and RL resumes from its progress record.

The default SFT output directories are `~/dmast_ckpt/paper_imitation/` and `~/dmast_ckpt/paper_enhanced_sft_task_focused/`. RL uses the `dmast_paper` experiment group under `~/dmast_ckpt/` and `~/bucket/ckpt/`. Use separate SFT output directories and experiment groups for runs with different data to avoid reusing incompatible checkpoints.

All SFT collectors use the task definitions in [configs/task_splits.py](configs/task_splits.py) and exclude the fixed 28 validation/evaluation tasks. The Stage 1 imitation collector defaults to the 42-task RL training split. The Stage 2 oracle pipeline and its golden/attacker collectors default to all registered non-held-out tasks; use `--no-all-tasks --num-tasks 42` to select the shared training split instead. Collection caches, SFT stage inputs, and the training data loader reject held-out or unidentified task records. See the [task-split guidance](tools/enhanced_sft_pipeline/README.md#training-and-held-out-tasks) for input requirements and task validation.

### Stage 1: Imitation learning

Collect clean and adversarial teacher trajectories:

```bash
python collect_data_for_imitation.py \
  --model-path ~/dmast_ckpt/pretrained/gemma-3-27b-it \
  --output ~/dmast_data/imitation/teacher_demos.json \
  --num-episodes 4000 \
  --all-tasks \
  --on-policy-role both
```

This schedules a nominal 4,000 episodes for each of the clean and attacked modes, using all registered non-held-out tasks. With T selected tasks, each mode schedules `T * (4000 // T)` attempts. Retained training transitions depend on collection and success filtering. The output matches `imitation.raw_data_path`, and imitation is enabled for both roles by default; `train.py` postprocesses the episodes and trains their shared model.

### Stage 2: Oracle-guided SFT

Build the denoising dataset (golden trajectories → HTML injections → oracle CoT → mix with clean data):

```bash
python tools/enhanced_sft_pipeline/run_pipeline.py \
  --model-path ~/dmast_ckpt/pretrained/gemma-3-27b-it \
  --output-dir ~/dmast_data/enhanced_sft \
  --num-episodes 4000 \
  --attacker-episodes 4000 \
  --all-tasks \
  --task-focused
```

The displayed episode and task options are the defaults. Golden collection schedules two independent modes, paired golden and clean-only, each with a nominal budget of 4,000 episodes. A separate attacker collection also has a 4,000-episode budget. With T selected tasks, each mode schedules `T * (4000 // T)` episodes. Final training counts are retained transitions after success filtering, step extraction, and formatting, not episode budgets; the three sample groups are combined without enforcing equal sizes.

`--task-focused` is the paper setting: the oracle receives the attacked view, original task goal, and golden action. Its prompt asks for task-focused CoT without acknowledging the attack; it is not given the clean view in this mode. Zero acknowledgment is a prompt objective: attack-related words only trigger warnings, selected comparison phrases are rejected, and exhausted retries use a template fallback. See the [oracle pipeline documentation](tools/enhanced_sft_pipeline/README.md) for the exact inputs and validation scope.

Stage 2 is enabled by default, and both `enhanced_sft.agent_data_path` and `attacker_data_path` point to `~/dmast_data/enhanced_sft/final_sft_dataset_simplified.json`. Stage 2 requires imitation to be enabled for the corresponding role; the defaults enable both stages. RL uses the resulting Stage 2 checkpoint as its reference model.

### Stage 3: Adversarial RL (GRPO self-play)

Run the complete pipeline after preparing the datasets:

```bash
python train.py
```

Both roles have `train_rl=True`, share one model, and use the current opponent without a historical population. Each iteration collects episodes, assigns each role its own binary success reward, computes GRPO advantages, and updates the shared policy. In this configuration, agent and attacker returns share normalization groups keyed by task, attack mode, and seed. Progress is written to `rl_progress.txt` in the cloud experiment directory; rerunning the command resumes after the last completed iteration.

The 25-iteration setting defines both the training cap and the persistent cosine scheduler horizon: `25 * (48 * 16 * 8 // 128) = 1200` optimizer steps. The paper reports iteration 10, selected through early stopping after later improvements became limited. The entry point does not implement an automatic early-stopping criterion; do not reduce `num_iterations` to 10 to reproduce the paper's 25-iteration schedule.

---

## Evaluation

### Main eval (held-out MiniWoB++)

The command below documents the MiniWoB++ evaluation workflow included in this repository.

```bash
python eval_multimodal_adversarial.py \
  --base-config /path/to/exp/config.json \
  --agent-models /path/to/agent/checkpoint \
  --attacker-models ~/dmast_ckpt/pretrained/gemma-3-27b-it \
  --output-dir ./eval_results \
  --num-episodes-per-env 8 \
  --eval-attacker-only
```

**Use a separate output directory for each evaluation run or setting.** The script reuses an existing `*_episodes.json` based only on its simplified model-combination name; it does not verify that the cached episodes match the requested models or evaluation configuration. Different models can share a name: two checkpoints ending in `iter_10_merged`, or two pretrained agents both labeled `base`, can collide within the same directory. If multiple requested models would have the same label, evaluate them in separate commands with separate output directories.

Choose a new `--output-dir` whenever you change either model/checkpoint, the prompt or defense level, attack goal, task set, seed, episode count, or other evaluation settings. Reuse a directory only to resume the same evaluation with unchanged models and settings. `[CACHED]` means an existing file was reused, not that its configuration was checked; `eval_config.json` is overwritten on each invocation and is not used to validate the cache.

Useful flags:

- `--defense-level {0,1,2}` — prompt-only defense (level 2 is the paper Prompt Defense setting)
- `--attack-goal {leakage,misinformation}` — selects the attack objective on supported MiniWoB++ tasks; leakage is the main threat

The misinformation option includes attack generation and success detection. The dedicated aggregation used for the paper's cross-objective results is not included here and applies a different task-success rule; the standard evaluator's summary is not the complete Table 4 evaluation protocol.

`run_quick_eval.py` is a thin launcher around the same script. Edit the model paths at the top of that file before running it.

### Vulnerability analysis (Table 1)

Compares no-attack, text-only, image-only, and dual HTML injection:

```bash
python study_attack_modes.py \
  --model-path ~/dmast_ckpt/pretrained/gemma-3-27b-it
```

---

## Project Structure

```
.
├── train.py                         # Stage 1 SFT -> Stage 2 oracle SFT -> Stage 3 GRPO
├── collect_data_for_imitation.py    # Teacher trajectory collection
├── eval_multimodal_adversarial.py   # Agent vs attacker evaluation
├── study_attack_modes.py            # Attack-modality ablation
├── agents/
│   ├── multimodal_web_agent.py      # Vision + AXTree agent
│   └── html_injection_attacker.py   # Structured HTML/CSS injection attacker
├── configs/train_adversarial_args.py
├── tools/
│   ├── multimodal_collect_episode.py
│   ├── postprocess_rollout.py
│   ├── assign_credit.py
│   ├── train_model.py
│   └── enhanced_sft_pipeline/       # Stage 2 oracle-guided SFT
└── utils/                           # vLLM, BrowserGym, paths, logging
```

The attacker action is a tuple `(target, position, html, css)` compiled to JavaScript and executed in the page. Because the screenshot and AXTree are both derived from the DOM, one injection updates both modalities.

---

## Citation

If you use this code, please cite the paper.

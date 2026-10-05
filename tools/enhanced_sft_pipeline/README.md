# Oracle-Guided SFT Pipeline (DMAST Stage 2)

Builds the oracle-guided SFT dataset: golden teacher trajectories and browser-captured HTML injections, oracle task-focused CoT, then a mix of augmented agent, clean agent, and attacker demonstrations.

The default CoT mode is **task-focused**. The oracle receives the attacked AXTree, original task goal, and golden action, plus the attacked screenshot when images are enabled and available. It does **not** receive the clean observation in this mode. The optional `--defense-focused` mode also supplies the clean observation as a reference and asks for explicit analysis of the attack.

Zero acknowledgment is a **prompt objective**, not a strict property enforced for every retained sample. In task-focused mode, `validate_cot()` checks minimum length, task-related keywords, and selected literal phrases such as "clean version" and "clean screenshot". Mentions of "attack", "injection", or "suspicious" trigger warnings rather than rejection by themselves. If retries produce no accepted reasoning, the pipeline substitutes a task-focused template and retains the step without revalidating that template. Other processing exceptions can cause a step to be skipped. These checks do not establish semantic correctness or guarantee the requested reasoning format.

## Stages

1. **Golden trajectories** (`stage1_golden_trajectory.py`): paired golden and independent clean-only episodes, retaining successful teacher trajectories. A separate `stage1b_attacker_collection.py` call collects adversarial episodes for attacker demonstrations.
2. **Attack factory** (`stage2_attack_factory.py`): normally formats and reuses attacked observations captured during golden collection; the golden action is kept as the label. The offline-generation fallback is not recommended: augmented samples whose golden records lack attack screenshots are skipped by the formatter. Prefer browser-paired collection and reuse of its attacked views.
3. **Oracle CoT** (`stage3_oracle_cot.py`): the oracle generates task-focused reasoning from the attacked observation, goal, and golden action, with the validation and fallback behavior described above.
4. **Format and mix** (`stage4_format_mix.py`): construct augmented agent, clean agent, and successful-episode attacker samples separately, then combine them and count the retained transitions in each category.

```bash
python tools/enhanced_sft_pipeline/run_pipeline.py \
  --model-path ~/dmast_ckpt/pretrained/gemma-3-27b-it \
  --output-dir ~/dmast_data/enhanced_sft \
  --num-episodes 4000 \
  --attacker-episodes 4000 \
  --all-tasks \
  --task-focused
```

The command uses the default episode budgets and selects all registered non-held-out tasks.

Resume from a later stage:

```bash
python tools/enhanced_sft_pipeline/run_pipeline.py --stages 2,3,4 --output-dir ~/dmast_data/enhanced_sft
```

The final `*_simplified.json` file is the SFT input used by `train.py`.

## Collection budgets and sample counts

`--num-episodes 4000` is a nominal budget **per golden collection mode**: paired golden and independent clean-only. `--attacker-episodes 4000` is the nominal budget for a third, independent adversarial collection. If T tasks are selected, each category schedules `T * (4000 // T)` episodes, since the collector rounds down to equal repetitions per task. For example, T=97 gives 3,977 episodes per category and 11,931 in total; T=42 gives 3,990 per category and 11,970 in total. These are planned raw episodes, not successful episodes or final training samples.

The formatter creates augmented agent samples from attacked observations with oracle reasoning and golden actions, clean agent samples from the teacher's clean observations and demonstrations, and attacker samples from successful attacker episodes. Clean samples are selected from successful golden trajectories whose steps lack an attacked screenshot; they can include steps from attacker-enabled episodes that did not yield an attack screenshot, as well as independently collected clean-only episodes. The clean label therefore describes the training demonstration, not a guarantee that the original episode had no attacker. Each usable step produces a training sample. The formatter combines these three groups and records their counts; it does not force their final sizes to match. The paper reports 2,704 augmented agent, 2,863 clean agent, and 2,997 attacker transitions, totaling 8,564. Actual retained counts depend on collection outcomes and filtering.

## Training and held-out tasks

Task definitions are shared with RL in [configs/task_splits.py](../../configs/task_splits.py). The oracle pipeline and its two standalone collectors default to all registered non-held-out tasks. The ordinary imitation collector defaults to the 42-task RL training split. The fixed 28 held-out tasks are the union of the 12 validation tasks and 16 evaluation tasks; they are excluded from SFT collection as well as the default RL training set.

For the oracle pipeline and its standalone collectors, `--no-all-tasks --num-tasks N` selects the first N tasks of the shared training list (1–42); omitting N in this mode uses all 42. `--all-tasks` is enabled by default and ignores `--num-tasks`, selecting all registered MiniWoB++ tasks **except** the 28 held-out tasks. Python callers can select the shared subset with `all_tasks=False`. Episode allocations use the number of selected tasks.

Existing collection caches, Stage 2–4 inputs, external attacker data, and reused final datasets are checked using each record's `task_name`. The training data loader also checks JSON/JSONL records before building the training dataset. A held-out task, missing task identity, or unsupported input schema stops processing with an error; samples are not silently removed. This applies when resuming later stages or calling them directly. Validation and evaluation collection remain allowed on held-out tasks.

`study_attack_modes.py` uses the imitation collector's `--evaluation` mode with a default 56-task study pool; `--all-tasks` selects the full benchmark. This evaluation path does not select SFT training data.

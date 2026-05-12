# DP Centralized Training — Troubleshooting Plan

Notes for the stuck `val_bal_acc ≈ 0.5` observed across the four DP-Adam sweeps in
`centralized_runs/dp_adam_experiments_20260511_173118.log`. Companion to
[`DIFFERENTIAL_PRIVACY.md`](DIFFERENTIAL_PRIVACY.md).

## 1. What the runs are doing

All four sweeps fail in the same way: balanced accuracy never rises above the
"predict-everything-as-benign" baseline.

| Run | batch | lr | C | σ | train_loss trend | train_bal_acc | val_bal_acc |
|---|---|---|---|---|---|---|---|
| `lr1e3_b4_norm1_noise1`     | 4  | 1e-3 | 1.0 | 1.0 | 0.97 → **1.50** (diverging) | flat ~0.50 | stuck 0.5000 |
| `lr5e4_b8_norm1_noise1`     | 8  | 5e-4 | 1.0 | 1.0 | flat ~1.02 | flat ~0.50 | stuck 0.5000 |
| `lr1e3_b8_norm12_noise1`    | 8  | 1e-3 | 1.2 | 1.0 | 1.00 → ~1.18 (drifting up) | flat ~0.50 | stuck 0.5000 |
| `lr5e4_b16_norm12_noise08`  | 16 | 5e-4 | 1.2 | 0.8 | flat ~1.06 | flat ~0.50 | stuck 0.5000 |

`metrics.json` (run 1) confirms the model never escapes the random init:

- `best_epoch: 1` (best checkpoint is the first eval pass)
- `best_validation.confusion_matrix: [[424, 0], [154, 0]]` — every validation
  sample predicted as benign
- `module_fixed: true`, `stochastic_depth_disabled: 17` — Opacus' `ModuleValidator.fix`
  swapped `BatchNorm1d`→`GroupNorm` in the head and disabled 17 stochastic-depth
  blocks
- `final_epsilon ≈ 0.88` — privacy budget is fine; the model just isn't learning

For reference, the archived non-DP run on the same data
(`centralized_runs/archive/convnext_tiny_20260511_150847`) reaches
`val_bal_acc ≈ 0.84` with `batch=128, lr_backbone=1e-5, lr_finetune=1e-4`.

Crucially, **`train_bal_acc` is also flat at 0.50** in every DP run. This is not
an eval-mode or validation-distribution bug; the model is genuinely not moving
off its initialization.

## 2. Root cause

### 2.1 Primary — signal-to-noise ratio is hopeless

For DP-SGD the noise on the averaged gradient has per-coordinate std

```
σ_eff = σ · C / B
```

| Run | σ_eff |
|---|---|
| b=4,  σ=1.0, C=1.0 | **0.25** |
| b=8,  σ=1.0, C=1.0 | 0.125 |
| b=8,  σ=1.0, C=1.2 | 0.150 |
| b=16, σ=0.8, C=1.2 | 0.060 |

ConvNeXt-Tiny here has ~28M trainable parameters and the new `head = nn.Linear`
is randomly initialized. After flat per-sample clipping at `C`, the *signal* per
coordinate is on the order of `C / √P ≈ 2e-4`. Noise is **3–4 orders of magnitude
larger than signal per coordinate**. The non-DP baseline used `batch=128`; these
DP runs cut the batch by 8–32×, which is the wrong direction — DP wants bigger
batches, not smaller ones.

### 2.2 Secondary — Adam + DP-SGD at `lr=1e-3` amplifies the noise

`prepare_private_training` passes `torch.optim.Adam` straight to Opacus. Adam
normalizes each step by `1/√v_t`. Once DP noise dominates the gradient, `v_t`
becomes the variance of pure Gaussian noise (uniform across coordinates), so
Adam's adaptive scaling collapses to a near-uniform step of size `≈ lr · sign(noise)`.
With `lr=1e-3` and ~500 steps/epoch × 50 epochs ≈ 25,000 steps, every parameter
takes ~25k random ±1e-3 walks. That is exactly the "train_loss climbing 0.97 → 1.50"
pattern in run 1 — the network is being annealed *away* from its pretrained
weights.

### 2.3 Amplifiers

1. **Random head, no LR split.** `MyConvNeXtTiny.head = nn.Linear(...)` has no
   pretrained weights, yet `lr_backbone == lr_finetune == 1e-3` (or `5e-4`) in
   every DP config. The new head must learn from scratch under DP noise while
   the 28M-parameter backbone is being shaken loose at the same LR.
2. **`class_weights: true` + flat clipping.** Class weights are `[0.66, 2.09]`,
   so per-sample CE gradients for malignant samples are ~3× larger and therefore
   get clipped harder under flat `C=1`. Minority-class signal is suppressed more
   than majority — biasing the model toward "always benign", which matches the
   `[[424, 0], [154, 0]]` confusion matrix.
3. **`poisson_sampling: false`.** Disables the variance-reduction benefit of
   Poisson averaging and prevents pairing with `BatchMemoryManager` to grow the
   logical batch. PRV accounting still works in both modes, so this is a
   performance issue, not a privacy issue.

### 2.4 Why `val_bal_acc` is *literally* 0.5000

At init the random `head = Linear(768, 2)` produces logits ≈ 0; `argmax` returns
class 0 (benign) deterministically. Validation = 424 benign + 154 malignant ⇒
`(424/424 + 0/154) / 2 = 0.5000` exactly. The occasional dips to 0.4989 / 0.4953
/ 0.4783 are noise-driven prediction flips. The model never leaves that basin
because every gradient step is buried in DP noise.

## 3. Troubleshooting plan

### Step 1 — Pipeline sanity check

Copy `dp_adam_lr5e4_b8_norm1_noise1.yml` and set:

```yaml
differential_privacy:
  enabled: false
```

Keep everything else (batch=8, lr=5e-4). Run 1–2 epochs.

- **If `train_bal_acc` still hovers at 0.5:** the bug is in the model adaptation
  / `ModuleValidator.fix` path, not DP. Investigate the BatchNorm→GroupNorm
  rewrite of the classifier head and whether `head` is receiving any gradient at
  all.
- **If it learns normally:** the problem is purely DP hyperparameters; continue
  to Step 2.

### Step 2 — Make DP-SGD viable

Apply these four changes together. Doing one of them is usually not enough on
ConvNeXt-Tiny.

1. **Freeze the backbone, train only the new head.** Drops trainable parameters
   from ~28M to ~1.5k, improving SNR by roughly four orders of magnitude. Add an
   `optimizer.freeze_backbone: true` flag and set `param.requires_grad = False`
   on everything classified as backbone in `split_parameters`.
2. **Large logical batch via Opacus `BatchMemoryManager`.** Keep the *physical*
   batch at 8–16 (memory bound) but request a *logical* batch of 256–1024. Add a
   `differential_privacy.logical_batch_size` key and wrap the training loop:

   ```python
   from opacus.utils.batch_memory_manager import BatchMemoryManager
   with BatchMemoryManager(
       data_loader=loaders["train"],
       max_physical_batch_size=config["dataloader"]["batch_size"],
       optimizer=optimizer,
   ) as memory_safe_loader:
       train_metrics = train_one_epoch(
           model, memory_safe_loader, criterion, optimizer, device
       )
   ```

   Pass a `DataLoader(batch_size=logical_batch_size)` into `make_private` so
   Opacus uses the logical size as the expected batch.
3. **Use plain DP-SGD (or DP-Adam at a much smaller LR).** SGD+momentum is the
   standard partner for DP-SGD. If keeping Adam, drop `lr_finetune` to `1e-4` or
   `5e-5` and set `lr_backbone: 0` (frozen).
4. **Tighten the clip, drop the noise, then ramp σ back up.** Start with
   `max_grad_norm: 0.1`, `noise_multiplier: 0.5`, `poisson_sampling: true`,
   confirm learning, then sweep σ upward toward the target ε.

### Step 3 — Suggested starter config

A replacement for `dp_adam_lr5e4_b16_norm12_noise08.yml` that should at least
move off 0.5:

```yaml
dataloader:
  batch_size: 16          # physical
  num_workers: 4
  shuffle: true
  pin_memory: true

epochs: 20                # DP fine-tuning needs few epochs, not 50

differential_privacy:
  enabled: true
  accountant: "prv"
  mode: "noise_multiplier"
  noise_multiplier: 0.5
  max_grad_norm: 0.1
  poisson_sampling: true
  clipping: "flat"
  grad_sample_mode: "hooks"
  fix_modules: true
  logical_batch_size: 512  # NEW — routed to BatchMemoryManager

optimizer:
  name: "Adam"             # or "SGD" with momentum 0.9
  lr_backbone: 0.0
  lr_finetune: 0.0001
  freeze_backbone: true    # NEW

loss:
  name: "CrossEntropyLoss"
  label_smoothing: 0.1
  class_weights: false     # replace with WeightedRandomSampler on train loader
```

### Step 4 — Code changes required in `centralized/train_centralized.py`

1. **`optimizer.freeze_backbone`.** In `split_parameters`, when the flag is set,
   call `param.requires_grad = False` on backbone params and return only the
   finetune group. Also reduces what Opacus has to wrap and speeds training.
2. **`differential_privacy.logical_batch_size`.** Plumb the key into
   `prepare_private_training` (use it as the DataLoader batch size handed to
   `make_private`) and wrap the per-epoch training call with
   `BatchMemoryManager` as shown above.
3. **Replace `class_weights=true` with a balanced sampler.** Use
   `torch.utils.data.WeightedRandomSampler` on the *training* loader only. This
   yields the same balancing effect without inflating per-sample gradient norms
   for the minority class, which is what flat clipping currently penalizes.
   Leave validation and test loaders unweighted.

### Step 5 — If the full backbone must stay trainable

For a full-finetune DP run, at minimum:

- raise the **logical batch to ≥ 1024** via `BatchMemoryManager`,
- drop `lr_backbone` to `1e-5` and `lr_finetune` to `1e-4` (match the non-DP run),
- set `max_grad_norm: 0.5` and `noise_multiplier: 0.5`,
- budget **5–10 epochs**, not 50 — long DP runs add drift without adding signal.

Realistically, for a 28M-parameter ConvNeXt-Tiny on 3.8k training samples, DP
fine-tuning of only the head is the regime that actually works.

## 4. Configs added for this iteration (non-coding)

Three configs were added in `centralized/experiment_configs/`, wired into
`scripts/run_dp_adam_experiments.sh` in this order. None require code changes.

1. **`sanity_off_adam_lr5e4_b16.yml`** — Step 1 of the plan. Keeps the failing
   DP runs' hyperparameters (`lr=5e-4`, `batch=16`, `class_weights=true`) but
   disables DP. 5 epochs. Purpose: confirm the pipeline can learn at all under
   those settings.
2. **`dp_adam_lr1e4_b32_norm05_noise025.yml`** — noise-sensitivity probe. DP on,
   `batch=32`, `lr_backbone=1e-5`, `lr_finetune=1e-4`, `C=0.5`, `σ=0.25`,
   `poisson_sampling=true`, 10 epochs. Sacrifices privacy budget to test
   whether the new regime can learn at low noise. If this stalls, full-backbone
   DP is hopeless without a code change (freeze backbone, BatchMemoryManager).
3. **`dp_adam_lr1e4_b32_norm05_noise05.yml`** — main candidate. Same as (2) but
   `σ=0.5`, which keeps a more useful ε. Only worth interpreting if (2) shows
   learning.

What is *not* changed in these configs because each would require a code edit:

- backbone freeze flag (Step 2.1)
- `BatchMemoryManager` for logical batches ≥ 256 (Step 2.2)
- `WeightedRandomSampler` replacement for `class_weights: true` (Step 4 of the
  plan's code changes)

## 5. Signals to verify in each run

Logs are written to `centralized_runs/dp_adam_experiments_<RUN_TS>.log`.
Per-epoch lines look like:

```
Epoch NNN/MMM train_loss=... train_bal_acc=... val_bal_acc=... [epsilon=...] epoch_time=...
```

Plus a final `Best model test metrics: {... "confusion_matrix": [[..],[..]] ...}`
block and the per-run `metrics.json` in the workdir.

### 5.1 `sanity_off_adam_lr5e4_b16` (DP off)

If this run does **not** learn, the failing DP runs were never going to learn
even at σ=0. Stop and revisit the model / data path before touching DP.

Pass signals (within 5 epochs):

- `epsilon=...` should be absent from the log lines (DP disabled).
- `metrics.json` should show `differential_privacy.enabled: false`.
- `train_loss` should drop by ≥ 0.1 between epoch 1 and epoch 5 (baseline went
  from 0.75 → 0.61 in the same span on `batch=128`; with `batch=16` expect
  smaller but still negative slope).
- `train_bal_acc` should rise above 0.55 by epoch 3, above 0.60 by epoch 5.
- `val_bal_acc` should rise above 0.55 by epoch 5 (the non-DP baseline hit 0.61
  by epoch 1 with `batch=128`).
- Final test `confusion_matrix` should have non-zero entries in **both**
  off-diagonal columns (i.e., not collapsed to a single class).
- `best_epoch` in `metrics.json` should be ≥ 2 (not the random init).

Fail signals:

- `val_bal_acc` stays at exactly 0.5000 for 5 epochs → re-check the head
  initialization / `BatchNorm1d` rewrite path, not DP. `train_bal_acc` flat at
  0.50 in the same window confirms it.
- `train_loss` *increases* with DP off → the high LR (5e-4) is wrong for this
  pipeline even without DP. Try `lr=1e-4` first.

### 5.2 `dp_adam_lr1e4_b32_norm05_noise025` (low-noise DP probe)

The strongest "is DP-SGD ever going to work without freezing the backbone?"
diagnostic. Apply the Section 4 acceptance criteria with these specifics:

Pass signals (within 5 epochs):

- `epsilon=...` should be present, monotonically increasing across epochs (PRV
  accountant works as expected).
- `train_loss` slope should be negative or at worst flat, **never increasing
  monotonically across 5 epochs** (the symptom of runs 1 and 3 in the original
  sweep).
- `train_bal_acc` deviates from 0.50 by more than 0.02 on at least 2
  consecutive epochs.
- `val_bal_acc` deviates from 0.5000 by more than 0.01 on at least 2
  consecutive epochs.
- `metrics.json` → `best_epoch ≥ 2`.
- `metrics.json` → `differential_privacy.noise_multiplier == 0.25`,
  `max_grad_norm == 0.5`, `poisson_sampling == true`, `module_fixed: true`.
- `final_epsilon` should be small (≪ 1.0 expected at σ=0.25 with PRV); used as
  an upper bound on the privacy cost of this debug run.

Fail signals:

- `train_loss` climbs across epochs → Adam is still being driven by noise even
  at σ=0.25; this means full-backbone DP fine-tuning is not viable here and we
  must freeze the backbone (code change) before continuing.
- `val_bal_acc` flat at 0.5000 with `train_bal_acc` flat at 0.50 → the model is
  still stuck at random init; same conclusion as above.
- `module_fixed: true` paired with `stochastic_depth_disabled > 0` → Opacus is
  rewriting the model the same way it did in the failing runs. If the sanity
  run (5.1) showed the head-rewrite path is OK, this is expected; if 5.1 also
  failed, this is where to look.

### 5.3 `dp_adam_lr1e4_b32_norm05_noise05` (main DP candidate)

Only interpret this run if 5.2 passed; otherwise it is uninformative.

Pass signals (within 10 epochs):

- All four pass signals from 5.2.
- `final_epsilon` is still useful (i.e., not blown past the project's privacy
  budget). Record it in `metrics.json`.
- `Best model test metrics.balanced_accuracy ≥ 0.55` — a real, non-degenerate
  test metric.

Fail signals:

- 5.2 passed but 5.3 stalls → confirms σ=0.5 is the bottleneck for *this*
  parameter count. Either accept the lower ε at σ=0.25, or apply the code
  changes (freeze backbone, larger logical batch) and re-try σ=0.5.
- `final_epsilon` exceeds the target with `val_bal_acc` still flat → the
  privacy budget is being spent without buying any signal. Stop and switch to
  the code-change plan rather than continuing the sweep.

### 5.4 What to grep for after a sweep finishes

Quick triage commands on the global log:

```bash
# Per-run final test confusion matrices (degenerate = single off-diagonal col)
rg '"confusion_matrix"' centralized_runs/dp_adam_experiments_<TS>.log

# Epsilon trajectory per run
rg 'epsilon=' centralized_runs/dp_adam_experiments_<TS>.log

# val_bal_acc trajectory per run
rg 'val_bal_acc=' centralized_runs/dp_adam_experiments_<TS>.log

# Which runs improved past random init
for d in centralized_runs/sanity_off_adam_lr5e4_b16_*/ \
         centralized_runs/dp_adam_lr1e4_b32_norm05_noise*_*/ ; do
  python -c "import json,sys; m=json.load(open('$d/metrics.json'));
print('$d', 'best_epoch=', m.get('best_epoch'),
'best_val_bal_acc=', m.get('best_validation', {}).get('balanced_accuracy'))"
done
```

## 5.5 Results from iteration 1 (sweep `20260512_121712`)

All three new runs completed; **none of them pass the §5 signals**.

| Run | epoch-1 → epoch-N train_loss | best_epoch | best val_bal_acc | final test cm | epsilon (final) |
|---|---|---|---|---|---|
| `sanity_off_adam_lr5e4_b16` (5 ep) | 0.788 → 0.748 | 1 | 0.5000 (`[[424,0],[154,0]]`) | `[[0,843],[0,324]]` (all malignant) | n/a |
| `dp_adam_lr1e4_b32_norm05_noise025` (10 ep) | 0.997 → 1.070 (↑) | 1 | 0.5000 | `[[843,0],[324,0]]` (all benign) | 98.4 |
| `dp_adam_lr1e4_b32_norm05_noise05` (10 ep) | 0.967 → 1.053 (↑) | 1 | 0.5000 | `[[843,0],[324,0]]` (all benign) | 9.4 |

Key observations:

- **Sanity (DP off) failed too**, in a different way: `train_loss` *did* decrease
  slightly (0.79→0.75) but `val_bal_acc` stayed exactly 0.5000 every epoch. The
  final test prediction flipped from "all benign" (epoch-1 best) to "all
  malignant" (epoch-5 final). The model is moving but only between two
  degenerate modes, never spending any time in a non-degenerate prediction
  regime. `best_epoch=1` (random init) for all three runs.
- **The low-σ DP probe failed.** At σ=0.25, C=0.5, B=32, the per-coordinate
  effective noise on the averaged gradient is σ_eff = σ·C/B = **0.0039**, and
  final ε=98 (no real privacy). Despite that, the run stalls *and* `train_loss`
  climbs across epochs. This rules out DP noise as the dominant cause — the
  blocker is somewhere in the Opacus wrapping (`module_fixed: true`,
  `stochastic_depth_disabled: 17`) plus the random `nn.Linear` head, or in the
  basic small-batch / weighted-CE combo that all three runs share.
- The σ=0.5 main candidate confirmed the same pattern (ε=9.4 final), so the
  outcome is not a low-σ fluke.

### Decision-tree mapping

The §5 tree had four branches:

1. 5.1 fails → not a DP problem; investigate model / data first.
2. 5.1 passes, 5.2 fails → freeze backbone is required.
3. 5.2 passes, 5.3 fails → σ is the bottleneck.
4. 5.3 passes → ship it.

5.1 *technically* failed, putting us on branch 1. But the sanity config used
`lr=5e-4, batch=16`, which is **not** the same as the DP configs
(`lr=1e-4, batch=32`). The failure of sanity may be a high-LR-scrambles-the-
backbone issue that does not apply to the DP runs at all. We have a
**confounded diagnostic**: we cannot yet tell branch 1 (broken model/data
path) from branch 2 (DP wrapper is the blocker).

### Next non-coding step before declaring the no-code space exhausted

Add two more sanity configs at the *DP runs' actual LRs* with DP off, to
disambiguate:

1. **`sanity_off_lr1e4_b16.yml`** — DP off, `lr_backbone=1e-5,
   lr_finetune=1e-4`, `batch=16`, `class_weights=true`, 10 epochs. Tests
   whether `batch=16` alone is too small for the pipeline.
2. **`sanity_off_lr1e4_b32.yml`** — DP off, same LRs, `batch=32`, 10 epochs.
   Mirrors the DP runs' batch/LR exactly, with DP off.

Interpretation table:

| (1) `b=16` off | (2) `b=32` off | Conclusion |
|---|---|---|
| learns | learns | The Opacus wrapper / module rewrite is the blocker → no-code DP space exhausted; switch to code-change plan (freeze backbone or BatchMemoryManager). |
| stalls | learns | Small batch alone is too small for this pipeline; batch ≥ 32 required. |
| stalls | stalls | Head architecture or `class_weights=true` + small batch is the fundamental issue, independent of DP. Investigate head / loss path before any DP work continues. |
| learns | stalls | Unlikely; would indicate a setup mismatch between the two runs. |

After running these two, the next code-or-no-code decision is:

- If conclusion = "wrapper is the blocker": apply the deferred code changes
  (`optimizer.freeze_backbone`, `differential_privacy.logical_batch_size` +
  `BatchMemoryManager`, optional `WeightedRandomSampler`).
- If conclusion = "batch is the blocker": raise physical batch in the DP
  configs to whatever the GPU can hold (e.g., 64) and re-run §5.2 / §5.3.
- If conclusion = "head/loss is the blocker": fix the model definition before
  proceeding with DP at all.

## 6. Acceptance criteria

A DP run is "moving" once we see, within the first 5 epochs:

- `train_bal_acc` deviating from 0.50 by more than 0.02 on at least 2
  consecutive epochs, and
- `val_bal_acc` deviating from 0.5000 by more than 0.01 on at least 2
  consecutive epochs, and
- `train_loss` trending downward, not upward.

If those three hold, the DP regime is workable and the σ / C / logical-batch
sweep can begin. If not, return to Step 1.

## 7. Open questions / follow-ups

- Does `WeightedRandomSampler` interact correctly with Opacus when
  `poisson_sampling: true`? May need to drop Poisson sampling in favor of
  oversampling, in which case re-verify the PRV accountant numbers.
- Worth comparing `clipping: "flat"` vs `clipping: "per_layer"` once the basic
  regime works — per-layer clipping often helps on transformer-style backbones.
- Long-term: replace the head's `BatchNorm1d` with `GroupNorm` *in the model
  definition* so DP and non-DP runs share an architecture, removing the
  `module_fixed: true` divergence.

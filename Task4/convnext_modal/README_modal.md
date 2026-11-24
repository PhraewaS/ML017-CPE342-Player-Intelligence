# Task 4 on Modal

`app.py` runs the Task 4 image experiments on Modal GPUs. You run it from your machine:
- **Containers:** each experiment and each cross-validation fold gets its own GPU container (L4, falling back to A10 or T4).
- **Experiment choice:** Optuna picks the next experiments from the scores of earlier ones, so you never edit configs by hand.
- **Resumable:** finished experiments and folds are cached on a Volume, and re-running a command skips them.

## Setup (once)
```bash
.venv/bin/modal setup          # log in to Modal
```
Training uploads `data_task4/task4_images.zip` (626 MB) to the Volume `cpe342-task4-data` if the archive is absent on that Volume. Run the commands from the repository root.

## Workflow
```bash
# 0. Cheap end-to-end check (tiny subset, a few training steps)
.venv/bin/modal run Task4/convnext_modal/app.py --mode screen --trials 2 --parallel 2 --smoke

# 1. Automated screening: Optuna picks configs, 4 trials at a time on 4 GPUs
.venv/bin/modal run Task4/convnext_modal/app.py --mode screen --trials 16 --parallel 4

# 2. See the leaderboard (local, free)
.venv/bin/modal run Task4/convnext_modal/app.py --mode leaderboard

# 3. Final: best 2 configs x 5 folds (10 GPUs in parallel) -> work/out/task4.csv + metrics
.venv/bin/modal run Task4/convnext_modal/app.py --mode final --top 2
#    or choose trial numbers yourself:  --trials-to-use 0,5,9
```
Add `--detach` after `modal run` to keep a long job running after you close the terminal.

The Modal pipeline writes its standalone CSV in `work/out/`. To assemble the released five-task v10 file, follow the repository [README](../../README.md). The ConvNeXt scores used in that file are saved in `Task4/predictions/`.

## What Optuna searches (`suggest()` in app.py)
| Parameter | Choices |
|---|---|
| Backbone | ConvNeXt-tiny, ConvNeXt-small (both ImageNet-22k), EfficientNetV2-S |
| Resolution (h×w) | 128×240, 160×288, 192×352 |
| Imbalance fix | logit adjustment / class-balanced sampler / none |
| Augmentation | CutMix 0 / 0.3 / 0.6, MixUp 0 / 0.2, RandAugment on/off (always on top of the corruption augmentations) |
| Training | lr ×0.4–2.5, 8 or 12 epochs, label smoothing 0 or 0.1, drop-path 0.1 or 0.2 |

The first trial is always the baseline config, so every later trial is compared against it.

## Reading the scores honestly
- Screening scores one fold, made of scenes held out from training. After many trials, the top score is partly luck on that fold, so expect the winner's real score to be a little lower.
- The `final` step's **5-fold out-of-fold macro F1** is the number to trust. Check it before submitting.
- Rules: the test set is only predicted, never trained or tuned on. Pretrained models are allowed for Task 4. Optuna counts as allowed hyper-parameter optimisation, not AutoML.

## Rough cost/time
A 12-epoch ConvNeXt-tiny trial on an L4 should take about 30–40 minutes; I haven't measured it yet. A batch of 4 trials runs in parallel, so 16 trials take about 4 batches' worth of time. Check prices in Modal's dashboard before large runs.

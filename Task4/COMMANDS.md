# Task 4: how each model was trained

`predict.py` combines four models' test log-probabilities (saved in `predictions/`) and the kernel model used in v10. The four team models were trained on GPUs
(Modal, a teammate's RTX 5060 laptop, an Apple-silicon Mac). GPU training is not bit-for-bit reproducible, so
retraining gives the same recipe and very similar, but not identical, predictions. The saved predictions reproduce the
submitted file exactly.

For these training recipes, unpack the official `task4_images.zip` in `data_task4/` at the repository root.
The directory contains `train.csv`, `test.csv`, `train/`, and `test/`. Run commands from the repository root.
Only train images and labels are used for training. Test images are only predicted.

| Weight | File in `predictions/` | Model | Code |
|---|---|---|---|
| 0.4 | `convnext_modal_ensemble.npy` | ConvNeXt-Small (ImageNet-22k), 2 configs × 5 folds | `convnext_modal/` |
| 0.4 | `clip_convnext_base.npy` | CLIP ConvNeXt-Base (LAION-2B → IN-12k), all train images | `clip_convnext/` |
| 0.1 | `dinov2_denoised.npy` | DINOv2-Small fine-tune, predicted on Gaussian-denoised test images | `dino/t4_local_finetune.py` |
| 0.1 | `dinov3_duplicates_logits.npy` (raw logits) | DINOv3-Small scene model, near-duplicates down-weighted | `dino/t4_scene_finetune.py` |

## 1. ConvNeXt-Small ensemble (Modal, L4 GPUs)
Optuna screened backbones, resolution, imbalance handling and augmentation on a scene-held-out split
(`convnext_modal/README_modal.md`). The final run trains the best two configs on 5 folds each:
```bash
modal run Task4/convnext_modal/app.py --mode screen --trials 16 --parallel 4
modal run Task4/convnext_modal/app.py --mode final --top 2          # used trials 4 and 5
CPE342_T4_DATA=data_task4 python -m Task4.convnext_modal.build_ensemble
```
`runs/<id>/cfg.json` holds each config:
- **128×240 run:** class-balanced sampler, label smoothing 0.1.
- **160×288 run:** logit adjustment.
- **Both:** CutMix 0.6, 12 epochs.

Out-of-fold macro F1 is 0.990 for each.

## 2. CLIP ConvNeXt-Base (teammate's RTX 5060 laptop, 58 min)
```bash
python Task4/clip_convnext/train_t4.py --data data_task4 --full     # writes out/clip_full_s0/test_logprob.npy
```
`convnext_base.clip_laion2b_augreg_ft_in12k` at 160×288, batch 16, lr 4e-5, 12 epochs, CutMix 0.6,
logit adjustment, label smoothing 0.1, drop-path 0.2, corruption augmentations.

## 3. DINOv2-Small, denoised inference (Apple-silicon Mac)
These research scripts use the repository root as the project directory and `work/review/` for weights and intermediate files.
```bash
python Task4/dino/t4_local_finetune.py . work/review/dinov2_small.safetensors --out work/review/t4_dino_finetune_fold3 --epochs 4
python Task4/dino/t4_local_finetune.py . work/review/dinov2_small.safetensors --weights work/review/t4_dino_finetune_fold3/weights.pt \
       --predictions work/review/t4_dino_finetune_fold3/test_denoised_logprob.npy --batch-size 64 --denoise-input
```
Test images get a 7×7 Gaussian blur (sigma 0.92) before prediction, which removes the sensor-style noise added to the
screenshots.

## 4. DINOv3-Small scene model (Apple-silicon Mac)
```bash
python Task4/dino/t4_scene_retrieval_audit.py . work/review          # scene fingerprints of train images
python Task4/dino/t4_independent_split.py . --work work/review --strict      # strict scene-held-out split (t4_independent_split_strict.csv)
python Task4/dino/t4_backbone_views.py . --work work/review --backbone dinov3 # cached DINOv3 views of train images
python Task4/dino/t4_scene_finetune.py . --split work/review/t4_independent_split_strict.csv --output work/review/t4_scene_dinov3_strict --epochs 5
python Task4/dino/t4_patch_head.py . --checkpoint work/review/t4_scene_dinov3_strict/best.pt --cache work/review/t4_dinov3_strict_patches.npy --extract
python Task4/dino/rare_group_weights.py .                           # up-weight rare visual groups
python Task4/dino/duplicate_weights.py data_task4                 # down-weight near-duplicate screenshots
python Task4/dino/t4_scene_finetune.py . --split work/review/t4_independent_split_strict.csv --backbone dinov3 \
       --warm-start work/review/t4_scene_dinov3_strict/best.pt --sample-weights work/review/t4_scene_duplicate_weights.npz \
       --learning-rate-scale 0.45 --epochs 1 --seed 342 --output work/review/t4_scene_dinov3_duplicates
python Task4/dino/t4_scene_finetune.py . --weights work/review/t4_scene_dinov3_duplicates/best.pt --test-output work/review/t4_scene_dinov3_duplicates/test.npz
```
The blend uses the `clean` (denoised-input) logits from `test.npz`.

## Blend weights
The weights 0.4 / 0.4 / 0.1 / 0.1 were set step by step:
1. Averaging the ConvNeXt ensemble and the CLIP model raised the leaderboard by +0.0003.
2. Adding the two DINO models (denoised and scene-robust) raised it by a further +0.0007.

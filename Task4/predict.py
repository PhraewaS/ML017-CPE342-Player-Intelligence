"""Combine the original team ensemble and the kernel model used in v10."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import log_softmax

from shared.config import IMAGE_CLASSES, ROWS


PREDICTIONS = Path(__file__).resolve().parent / "predictions"
MODELS = (
    "convnext_modal_ensemble", "clip_convnext_base", "dinov2_denoised", "dinov3_duplicates_logits",
)


def read_scores(path):
    values = np.load(path, allow_pickle=False)
    if values.shape != (ROWS, IMAGE_CLASSES) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite scores for {ROWS} images and {IMAGE_CLASSES} classes in {path}")
    return values


def predict(probabilities=PREDICTIONS / "kernel_probabilities.npy", weight=0.08):
    if not 0 <= weight <= 1:
        raise ValueError("Kernel weight must be between 0 and 1")
    convnext, clip, dinov2, dinov3 = (read_scores(PREDICTIONS / f"{name}.npy") for name in MODELS)
    team_scores = 0.40 * convnext + 0.40 * clip + 0.10 * dinov2 + 0.10 * log_softmax(dinov3, axis=1)
    image_prob = read_scores(probabilities)
    if ((image_prob < 0) | (image_prob > 1)).any() or not np.allclose(image_prob.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Kernel probabilities must be between zero and one and sum to one per image")

    team_log = log_softmax(team_scores, axis=1)
    kernel_log = np.log(image_prob.clip(1e-10))
    team_log -= team_log.max(axis=1, keepdims=True)
    kernel_log -= kernel_log.max(axis=1, keepdims=True)
    return ((1 - weight) * team_log + weight * kernel_log).argmax(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probabilities", type=Path, default=PREDICTIONS / "kernel_probabilities.npy")
    parser.add_argument("--weight", type=float, default=0.08)
    parser.add_argument("--out", type=Path, default=Path("trained/task4.npy"))
    args = parser.parse_args()
    labels = predict(args.probabilities, args.weight)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, labels, allow_pickle=False)
    print(f"Wrote {args.out} with {len(labels):,} image labels")


if __name__ == "__main__":
    main()

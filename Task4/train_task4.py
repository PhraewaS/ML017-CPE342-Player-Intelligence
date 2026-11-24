"""Train the kernel and nearest-neighbor model on frozen image embeddings."""

import argparse
import os
import zipfile
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="1", MKL_NUM_THREADS="4")

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from shared.config import IMAGE_CLASSES, ROWS, sample_ids


def load_views(folder, views, split, rows):
    features = []
    for view in views:
        matrix = np.load(folder / f"{view}-{split}.npy", allow_pickle=False)
        if matrix.ndim != 2 or len(matrix) != rows:
            raise ValueError(f"Unexpected embedding shape for {view}-{split}: {matrix.shape}")
        features.append(normalize(matrix))
    return normalize(np.column_stack(features))


def kernel(a, b):
    dots = a @ b.T
    np.subtract(dots, 1, out=dots)
    dots *= 3.0
    return np.exp(dots, out=dots)


def predict(features, labels, query):
    centers_idx, _ = train_test_split(
        np.arange(len(features)), train_size=2304, stratify=labels, random_state=530,
    )
    centers = features[centers_idx]
    basis = kernel(features, centers)
    weights = (np.bincount(labels)[labels] / len(labels) * IMAGE_CLASSES) ** -0.35
    sqrt_weights = np.sqrt(weights.astype(np.float32))[:, None]
    basis *= sqrt_weights
    basis = basis.astype(np.float64)
    gram = basis.T @ basis + 0.3 * kernel(centers, centers)
    gram.flat[::len(gram) + 1] += 0.004
    target = np.eye(IMAGE_CLASSES, dtype=np.float32)[labels] * sqrt_weights
    coefficients = solve(gram, basis.T @ target, assume_a="pos")

    output = []
    for start in range(0, len(query), 128):
        batch = query[start:start + 128]
        regression = softmax(8 * (kernel(batch, centers) @ coefficients), axis=1)
        similarities = batch @ features.T
        top = np.argpartition(similarities, -16, axis=1)[:, -16:]
        neighbors = np.take_along_axis(similarities, top, axis=1)
        neighbor_weights = softmax(neighbors / 0.027, axis=1)
        local = np.stack(
            [(neighbor_weights * (labels[top] == k)).sum(axis=1) for k in range(IMAGE_CLASSES)],
            axis=1,
        )
        output.append(0.75 * regression + 0.25 * local)
    return np.vstack(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Path to task4_images.zip")
    parser.add_argument("--embeddings", type=Path, default=Path("embeddings"))
    parser.add_argument("--out", type=Path, default=Path("trained/task4.npy"))
    parser.add_argument("--views", nargs="+", default=["dino", "upright", "clip"])
    parser.add_argument("--audit", action="store_true", help="Also evaluate on an 18% stratified holdout")
    args = parser.parse_args()

    with zipfile.ZipFile(args.images) as archive:
        labels = pd.read_csv(archive.open("train.csv")).label.to_numpy()
        test_ids = pd.read_csv(archive.open("test.csv"), usecols=["id"]).id.tolist()
    if test_ids != sample_ids():
        raise ValueError("Task 4 test IDs do not match the sample submission order")
    train = load_views(args.embeddings, args.views, "train", len(labels))
    if args.audit:
        fit, holdout = train_test_split(
            np.arange(len(train)), test_size=0.18, stratify=labels, random_state=243,
        )
        validation = predict(train[fit], labels[fit], train[holdout])
        score = f1_score(labels[holdout], validation.argmax(axis=1), average="macro")
        print(f"Holdout macro F1: {score:.6f}", flush=True)
    test = load_views(args.embeddings, args.views, "test", ROWS)
    estimates = predict(train, labels, test)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, estimates)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

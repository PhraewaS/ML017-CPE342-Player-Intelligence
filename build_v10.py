"""Assemble the original v10 submission or replace predictions for individual tasks."""

import argparse
import csv
import hashlib
import io
from pathlib import Path

import numpy as np

from shared.config import ROOT, ROWS, sample_ids
from Task4.predict import predict as predict_images


COLUMNS = ("id", "task1", "task2", "task3", "task4", "task5")
CLASS_COUNTS = {1: 2, 2: 4, 4: 5, 5: 2}


def predictions(task, path):
    if task == 4 and path is None:
        return predict_images()
    source = path if path is not None else ROOT / f"Task{task}/cache/release.npy"
    values = np.load(source, allow_pickle=False)
    if values.shape != (ROWS,) or values.dtype.kind not in "biuf" or not np.isfinite(values).all():
        raise ValueError(f"Task {task} requires {ROWS:,} finite numeric predictions in {source}")
    if task == 3:
        if not ((0 <= values) & (values <= 300_000)).all():
            raise ValueError("Task 3 predictions must be between 0 and 300,000")
        return [f"{value:.2f}" for value in values]
    if not np.isin(values, range(CLASS_COUNTS[task])).all():
        raise ValueError(f"Unexpected class labels in Task {task} in {source}")
    return values.astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for task in range(1, 6):
        parser.add_argument(f"--task{task}", type=Path, help=f"Override Task {task} with a 1D NumPy array")
    parser.add_argument("--out", type=Path, default=Path("outputs/ML017_submission_v10.csv"))
    parser.add_argument("--check", action="store_true", help="Require an exact match to the original v10")
    args = parser.parse_args()

    arrays = [predictions(task, getattr(args, f"task{task}")) for task in range(1, 6)]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(zip(sample_ids(), *arrays, strict=True))
    content = buffer.getvalue().encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    if args.check and digest != (ROOT / "shared/v10.sha256").read_text(encoding="ascii").strip():
        raise SystemExit(f"CSV differs from the original v10. SHA-256: {digest}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(content)
    print(f"Wrote {args.out} with {ROWS:,} rows. SHA-256: {digest}")


if __name__ == "__main__":
    main()

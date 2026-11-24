"""Apply the original, refined, and v3 anomaly rules carried forward to v10."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from shared.competition_data import read_task
from .signature_rules import detect_signatures
from .stability import refine
from .task5_rules import detect_anomalies


TEST_ID_HASH = "366495677590a62c1b75e73c6a7eb53ab1b49e61b6f376c0091b4f482fa7e604"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Official competition ZIP or data directory")
    parser.add_argument("--odds-cutoff", type=float, help="Override the v3 missing-stability odds threshold")
    parser.add_argument("--out", type=Path, default=Path("trained/task5.npy"))
    args = parser.parse_args()
    data = read_task(args.data, "05_anomaly_detection", "test.csv")
    digest = hashlib.sha256("\n".join(data.player_id).encode("utf-8")).hexdigest()
    if digest != TEST_ID_HASH:
        raise ValueError("Task 5 player IDs must follow the official test order")
    baseline = detect_signatures(data)
    updated = detect_anomalies(data, baseline.astype(bool))
    labels = refine(data, updated.astype(bool), args.odds_cutoff)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, labels, allow_pickle=False)
    print(f"Wrote {args.out} with {labels.sum():,} anomalies across {len(labels):,} rows")


if __name__ == "__main__":
    main()

"""Separate two report-count distributions when learning combat-statistic distributions."""
import argparse
import os
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
from shared.official_data import read_aligned
from . import generator_blend as gb


def report_ratio(fit, y, query):
    # Extreme report counts identify the generating band. The sample weights
    # undo their unequal inclusion probabilities within those two bands.
    reports = fit[gb.REPORTS]
    known = (y == 1) & ((reports <= 5) | (reports >= 16))
    columns = gb.COMBAT + ["weapon_switch_speed", "first_blood_rate", "damage_per_round"]
    center = fit.loc[known, columns].median()
    def design(frame):
        values = frame[columns]
        return np.column_stack([values.fillna(center), values.isna()])
    scaler = StandardScaler()
    x = scaler.fit_transform(design(fit.loc[known]))
    high = (reports[known] >= 16).to_numpy()
    model = LogisticRegression(C=.15, max_iter=250)
    model.fit(x, high, sample_weight=np.where(high, 14/4, 12/2))
    prob = model.predict_proba(scaler.transform(design(query)))[:, 1]
    r = query[gb.REPORTS].to_numpy()
    lower = gb.uniform(r, (4, 15), discrete=True)
    upper = gb.uniform(r, (6, 19), discrete=True)
    base = .69 * lower + .31 * upper
    return np.divide((1-prob)*lower + prob*upper, base, out=np.ones(len(r)), where=base>0)


def estimate(fit, y, query, boosted):
    scores, _ = gb.mixture(fit, query, y, gb.population_features(fit),
                           gb.population_features(query), gb.RULE)
    ratio = report_ratio(fit, y, query)
    updated = scores * ratio / (1 - scores + scores * ratio)
    def combine(p):
        return .25 * boosted + .75 * np.where(np.isfinite(p), p, boosted)
    return combine(scores), combine(updated)


def run(source, output):
    folder = "01_binary_classification"
    raw = read_aligned(source, folder, "train.csv").dropna(subset=["is_cheater"])
    labels = raw.pop("is_cheater").to_numpy(int)
    features = raw.drop(columns="player_id")
    boosted = np.load(ROOT / "cache/task1_oof_prob.npy")
    old = np.empty(len(labels))
    new = np.empty(len(labels))
    folds = np.empty(len(labels), dtype=int)
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(features, labels)):
        args = (features.iloc[a], labels[a], features.iloc[b], boosted[b])
        old[b], new[b] = estimate(*args)
        folds[b] = k
        print("Completed combat fold", k, flush=True)

    def metric(weight, mask):
        prob = (1 - weight) * old[mask] + weight * new[mask]
        return fbeta_score(labels[mask], prob > gb.THRESHOLD, beta=2)

    weight = max([0, .5, 1], key=lambda w: metric(w, folds < 2))
    audit = folds >= 2
    for k in range(5):
        print("Combat F2 fold", k, metric(0, folds == k), metric(weight, folds == k))
    print("Combat weight, audit before and after", weight, metric(0, audit), metric(weight, audit), flush=True)
    ref = np.load(ROOT / "cache/baseline_v3.npy", allow_pickle=False)
    if weight and metric(weight, audit) > metric(0, audit):
        test = read_aligned(source, folder, "test.csv").drop(columns="player_id")
        boosted_test = np.load(ROOT / "cache/task1_test_prob.npy")
        before, after = estimate(features, labels, test, boosted_test)
        if not np.array_equal(before > gb.THRESHOLD, ref):
            raise ValueError("Reconstructed Task 1 baseline differs from v3.")
        ref = ((1 - weight) * before + weight * after > gb.THRESHOLD).astype(int)
    output.parent.mkdir(exist_ok=True, parents=True)
    np.save(output, ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "trained/combat.npy")
    args = parser.parse_args()
    run(args.data, args.out)

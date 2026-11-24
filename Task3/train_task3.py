"""Fit the additive spending model and blend it with the team's model."""

import argparse
import os
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="2")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from shared.config import ROWS, task_artifacts


def prepare(frame):
    data = frame.drop(columns="player_id").copy()
    transactions = data.total_transactions.fillna(
        (data.purchases_on_discount / data.discount_rate_used.replace(0, np.nan)).round()
    )
    history = data.historical_spending.fillna(transactions.clip(lower=1) * data.avg_transaction_value)
    previous = 2.857 * data.prev_month_spending
    data["base"] = (0.6 * history + 0.4 * previous).fillna(history).fillna(previous)
    data["season"] = data.seasonal_spending_pattern.map({
        0: 2, 11: 2, 5: 1, 6: 1, **dict.fromkeys([1, 2, 3, 4, 7, 8, 9, 10], 0),
    })
    days = data.days_since_last_purchase.fillna(-1).to_numpy()
    data["recent"] = np.digitize(np.floor(days), [0, 15, 45]) - 1
    data["freq"] = np.digitize(data.purchase_frequency, [5, 8, 10, 15])
    data.loc[data.purchase_frequency.isna(), "freq"] = -1
    data["event"] = np.floor(data.event_participation_rate / 6)
    data["activity"] = np.floor(data.cross_game_activity / 4)
    data["hist_bin"] = np.floor(np.log2(1 + data.base) * 2)
    data["premium"] = data.is_premium_member.fillna(
        (data.vip_status >= 3).astype(float).where(data.vip_status.notna())
    )
    return data


def design(frame):
    categories = ["segment", "vip_status", "premium", "season", "recent", "freq",
                  "event", "games_played", "activity", "hist_bin"]
    basis = frame[categories].fillna(-1).astype(int).astype(str)
    for column in categories[1:]:
        basis[f"seg_{column}"] = basis.segment + "/" + basis[column]
    return basis


def fit_predict(train, target, query):
    train, query = train.copy(), query.copy()
    features = train.columns.drop("segment")
    known = train.segment.notna()
    segment = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=15, early_stopping=False, random_state=14,
    )
    segment.fit(train.loc[known, features], train.loc[known, "segment"])
    for frame in (train, query):
        missing = frame.segment.isna()
        if missing.any():
            frame.loc[missing, "segment"] = segment.predict(frame.loc[missing, features])

    gate = HistGradientBoostingClassifier(
        max_iter=160, max_leaf_nodes=12, learning_rate=0.07,
        early_stopping=False, l2_regularization=6, random_state=15,
    )
    gate.fit(train, target > 0)
    probability = gate.predict_proba(query)[:, 1]

    positive = (target > 0) & (train.base.fillna(0).to_numpy() > 0)
    spenders = train.loc[positive]
    encoder = OneHotEncoder(handle_unknown="ignore")
    fit_design = encoder.fit_transform(design(spenders))
    query_design = encoder.transform(design(query))
    response = np.log(target[positive] / spenders.base.clip(lower=1)).to_numpy()
    weights = np.ones(len(response))
    model = Ridge(alpha=12, solver="lsqr")
    for _ in range(5):
        model.fit(fit_design, response, sample_weight=weights)
        residual = response - model.predict(fit_design)
        weights = np.minimum(1, 0.17 / np.maximum(np.abs(residual), 1e-5))

    log_ratio = model.predict(query_design)
    for group in range(4):
        index = query.segment.to_numpy() == group
        reference = residual[spenders.segment.to_numpy() == group]
        percentile = np.clip(1 - 0.5 / np.maximum(probability[index], 0.0001), 0.012, 0.50)
        log_ratio[index] += np.quantile(reference, percentile)
    forecast = np.where(probability > 0.5, np.exp(log_ratio) * query.base.fillna(0), 0)
    return forecast.clip(0, 300_000)


def original_estimates(suffix=""):
    artifacts = task_artifacts(3)
    mixture = np.load(artifacts / f"task3_mixture{suffix}.npy", allow_pickle=False)
    mlp = np.load(artifacts / f"task3_mlp{suffix}.npy", allow_pickle=False)
    return 0.85 * mixture + 0.15 * mlp


def metric(target, estimates):
    return 1 / (1 + np.abs(target - estimates).sum() / target.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Competition 03_regression directory")
    parser.add_argument("--out", type=Path, default=Path("trained/task3.npy"))
    args = parser.parse_args()
    train = pd.read_csv(args.data / "train.csv")
    target = train.pop("spending_30d").to_numpy()
    test = pd.read_csv(args.data / "test.csv")
    if len(test) != ROWS or not test.player_id.is_unique:
        raise ValueError("Expected the original Task 3 test data with 25,889 unique players")
    train, test = prepare(train), prepare(test)
    original = original_estimates("_oof")
    proposed = np.empty(len(train))
    folds = np.empty(len(train), dtype=int)
    splits = KFold(n_splits=5, shuffle=True, random_state=42).split(train)
    for fold, (fit, validation) in enumerate(splits):
        held_out = train.iloc[validation]
        estimates = fit_predict(train.iloc[fit], target[fit], held_out)
        proposed[validation] = np.where(held_out.base.isna(), original[validation], estimates)
        folds[validation] = fold
        print(f"Fold {fold}: {metric(target[validation], proposed[validation]):.6f}", flush=True)

    selection = folds <= 1
    audit = ~selection
    weight = max(
        (0, 0.25, 0.50, 0.75, 1),
        key=lambda w: metric(target[selection], ((1 - w) * original + w * proposed)[selection]),
    )
    combined = (1 - weight) * original + weight * proposed
    before = metric(target[audit], original[audit])
    after = metric(target[audit], combined[audit])
    print(f"Audit score {before:.8f} -> {after:.8f}, candidate weight {weight}", flush=True)
    test_original = original_estimates()
    if weight > 0 and after > before:
        estimates = fit_predict(train, target, test)
        estimates = np.where(test.base.isna(), test_original, estimates)
        test_original = (1 - weight) * test_original + weight * estimates
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, test_original)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

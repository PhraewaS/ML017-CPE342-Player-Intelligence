"""Task 3 model A: LightGBM hurdle model.
P(spend > 0) from a classifier; if above 0.5, predict the conditional median of the positive amount, read off three
quantile regressors (0.25/0.45/0.50) at the percentile that makes the overall prediction the median of the
zero-inflated distribution (MAE is minimised by the median). Writes artifacts/task3_mixture.npy.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
from shared.config import task_artifacts, ROWS, T3
from .restore_features import blend_history, restore
from .spend_features import ID, TARGET, features, nmae


QUANTILES = (0.25, 0.45, 0.50)
ESTIMATORS = ("absolute_weight", "mixture", "mixture_blend")


def fit_models(frame, y):
    ratio = (frame["historical_spending"] / frame["prev_month_spending"]).median()
    x, scale = features(frame, ratio)
    normalized = y / scale
    positive = y > 0
    config = dict(
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=100,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.95,
        verbosity=-1,
        n_jobs=2,
        random_state=0,
    )
    direct = lgb.LGBMRegressor(**config, n_estimators=650, objective="l1")
    direct.fit(x, normalized, sample_weight=scale)
    positive_rate = lgb.LGBMClassifier(**config, n_estimators=400)
    positive_rate.fit(x, positive, sample_weight=np.sqrt(scale))
    quantiles = []
    for alpha in QUANTILES:
        model = lgb.LGBMRegressor(**config, n_estimators=650, objective="quantile", alpha=alpha)
        model.fit(x.loc[positive], np.log(normalized[positive]), sample_weight=np.sqrt(scale[positive]))
        quantiles.append(model)
    return ratio, direct, positive_rate, quantiles


def predict(models, frame):
    ratio, direct, positive_rate, quantiles = models
    x, scale = features(frame, ratio)
    p = positive_rate.predict_proba(x)[:, 1]
    target_percentile = np.clip(1 - 0.5 / np.maximum(p, 1e-6), 0, 0.5)
    values = np.maximum.accumulate(np.column_stack([np.exp(model.predict(x)) for model in quantiles]), axis=1)
    hurdle = np.fromiter(
        (np.interp(percentile, QUANTILES, row) for percentile, row in zip(target_percentile, values, strict=True)),
        dtype=float,
        count=len(x),
    )
    hurdle = np.where(p > 0.5, hurdle * scale, 0)
    weighted = np.maximum(0, direct.predict(x) * scale)
    return dict(absolute_weight=weighted, mixture=hurdle, mixture_blend=(weighted + hurdle) / 2)


ARTIFACTS = task_artifacts(3)


def main():
    train = pd.read_csv(T3 / "train.csv")
    y = train.pop(TARGET).to_numpy()
    x = restore(train.drop(columns=ID), buckets=True)
    scores = {name: np.empty(len(x)) for name in ESTIMATORS}
    models = []
    for i, (a, b) in enumerate(KFold(5, shuffle=True, random_state=42).split(x), start=1):
        fit_x, val_x = x.iloc[a], x.iloc[b]
        adjustment = (fit_x["historical_spending"] / fit_x["prev_month_spending"]).median()
        fit_x, val_x = blend_history(fit_x, adjustment), blend_history(val_x, adjustment)
        fold = fit_models(fit_x, y[a])
        models.append((fold, adjustment))
        for name, prediction in predict(fold, val_x).items():
            scores[name][b] = prediction
        print(f"Fold {i}", {name: round(nmae(y[b], out[b]), 6) for name, out in scores.items()}, flush=True)
    print("OOF NMAE", {name: round(nmae(y, out), 6) for name, out in scores.items()}, flush=True)
    np.save(ARTIFACTS / "task3_mixture_oof.npy", scores["mixture"])

    test = pd.read_csv(T3 / "test.csv")
    test_ids = test.pop(ID)
    test = restore(test, buckets=True)
    if len(test) != ROWS or not test_ids.is_unique or list(test.columns) != list(x.columns):
        raise ValueError("Unexpected Task 3 test schema.")
    result = np.mean([predict(fold, blend_history(test, adjustment))["mixture"] for fold, adjustment in models], axis=0)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite Task 3 predictions.")
    np.save(ARTIFACTS / "task3_mixture.npy", result)


if __name__ == "__main__":
    main()

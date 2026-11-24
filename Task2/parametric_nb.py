"""Task 2: naive Bayes with smoothed histograms, where informative features get a truncated-normal-regularised
density, plus a macro-F1 decision rule fitted on out-of-fold posteriors.

Why naive Bayes: within each segment the features are independent (max within-class Spearman 0.04), so the
naive Bayes form is the Bayes-optimal classifier for this data. 5-fold OOF macro F1 is about 0.755.
Writes artifacts/task2.npy (integer labels 0-3 in test order).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr, softmax
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
from shared.config import task_artifacts, ROWS, T2
from .density import fit_density, macro_f1_rule
from .hist_nb import HistNB


def gaussian_prob(edges, mean, log_scale):
    cdf = ndtr((edges - mean) / np.exp(log_scale))
    p = np.maximum(np.diff(cdf), 1e-14)
    return p / p.sum()


def regularized_probs(model, name):
    edges = model.edges[name][1:-1]
    if len(edges) < 10:
        return model.logp[name]
    middle = np.median(edges)
    scale = np.quantile(edges, 0.75) - np.quantile(edges, 0.25)
    if scale <= 0:
        return model.logp[name]
    standardized = (edges - middle) / scale
    frequencies = np.exp(model.logp[name])
    regularized = frequencies.copy()
    for cls, row in enumerate(frequencies):
        inner = row[1:-2]
        if not len(inner):
            continue
        result = minimize(
            lambda p: -np.dot(inner, np.log(gaussian_prob(standardized, *p))),
            [0, -0.3],
            method='L-BFGS-B',
            bounds=[(-5, 5), (-2.5, 2)],
        )
        if result.success:
            modeled = gaussian_prob(standardized, *result.x)
            regularized[cls, 1:-2] = 0.5 * inner + 0.5 * modeled * inner.sum()
    return np.log(regularized)


def fit_model(frame, labels):
    model = fit_density(HistNB, frame, labels)
    tuned = np.zeros(len(model.cols), bool)
    for i, name in enumerate(model.cols):
        if name not in model.edges:
            continue
        prior = np.exp(model.prior)
        frequencies = np.exp(model.logp[name])
        info = (prior[:, None] * frequencies * np.log(frequencies / (prior @ frequencies))).sum()
        if info >= 0.006:
            model.logp[name] = regularized_probs(model, name)
            tuned[i] = True
    return model, tuned.sum()


def evaluate(frame, labels, seed):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(frame, labels))
    probs = np.zeros((len(labels), 4))
    models = []
    for train, validate in folds:
        model, tuned = fit_model(frame.iloc[train], labels[train])
        models.append(model)
        probs[validate] = softmax(model.log_lik(frame.iloc[validate]) + model.prior, axis=1)
        print(f'Seed {seed}, {tuned} fitted marginals', flush=True)
    pred = np.empty(len(probs), dtype=int)
    for train, val in folds:
        offset, scale = macro_f1_rule(probs[train])
        pred[val] = ((probs[val] - offset) / scale).argmax(axis=1)
    print(f'Seed {seed}: OOF macro F1 {f1_score(labels, pred, average="macro"):.6f}, log loss {log_loss(labels, probs):.6f}', flush=True)
    return models, probs


ARTIFACTS = task_artifacts(2)


def main():
    train = pd.read_csv(T2 / 'train.csv')
    labels = train.pop('segment').to_numpy(int)
    features = train.drop(columns='player_id')
    models, oof = evaluate(features, labels, 42)

    test = pd.read_csv(T2 / 'test.csv')
    assert len(test) == ROWS and test.player_id.notna().all() and test.player_id.is_unique
    test = test.drop(columns='player_id')
    assert test.columns.equals(features.columns)
    probabilities = np.mean([model.predict_proba(test) for model in models], axis=0)
    offset, scale = macro_f1_rule(oof)
    predictions = ((probabilities - offset) / scale).argmax(axis=1)
    np.save(ARTIFACTS / 'task2.npy', predictions.astype(np.int64))
    print(f'Wrote {len(predictions):,} Task 2 predictions.', flush=True)


if __name__ == '__main__':
    main()

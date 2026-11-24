"""Task 3 model B: small MLP that outputs a zero-inflated log-normal spending distribution (P(spend), location,
spread) plus a direct amount head; the prediction averages the distribution's median and the direct head.
Trained from scratch, CPU, fixed seeds. Writes artifacts/task3_mlp.npy.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
from shared.config import task_artifacts, ROWS, T3
from .restore_features import blend_history, restore
from .spend_features import ID, TARGET, features, nmae


def encode(train, frames):
    data = train.to_numpy(np.float64, copy=True)
    data[~np.isfinite(data)] = np.nan
    transform = QuantileTransformer(n_quantiles=512, output_distribution='normal', random_state=342).fit(data)
    categories = [np.unique(column[np.isfinite(column)]) for column in data.T]
    encoded = []
    for frame in (train, *frames):
        values = frame.to_numpy(np.float64, copy=True)
        values[~np.isfinite(values)] = np.nan
        parts = [np.nan_to_num(transform.transform(values), nan=0), np.isnan(values)]
        for column, choices in enumerate(categories):
            if 2 <= len(choices) <= 24:
                parts.append(values[:, column, None] == choices[None, :])
        encoded.append(torch.from_numpy(np.column_stack(parts).astype(np.float32)))
    return encoded


@torch.inference_mode()
def predict(net, inputs, scale):
    probability, location, raw_sigma, raw_direct = net(inputs).unbind(1)
    p = probability.sigmoid().numpy()
    sigma = (F.softplus(raw_sigma) + 0.05).clamp(max=2).numpy()
    percentile = np.clip(1 - 0.5 / np.maximum(p, 1e-6), 0.005, 0.5)
    mixture = np.where(p > 0.5, np.exp(location.numpy() + sigma * ndtri(percentile)), 0)
    direct = F.softplus(raw_direct).numpy()
    result = (direct + mixture) * 0.5 * scale
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('Expected one finite nonnegative amount for each row.')
    return result


def fit_fold(train, amounts, frames, seed):
    ratio = (train['historical_spending'] / train['prev_month_spending']).median()
    data, scale = features(blend_history(train, ratio), ratio)
    evaluation = [features(blend_history(frame, ratio), ratio) for frame in frames]
    inputs, *other_inputs = encode(data, [frame for frame, _ in evaluation])

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    y = torch.tensor(amounts / scale, dtype=torch.float32)
    positive = (y > 0).float()
    log_amount = y.clamp_min(1e-8).log()
    root = np.sqrt(scale)
    weights = torch.tensor(root / root.mean(), dtype=torch.float32)
    net = nn.Sequential(
        nn.Linear(inputs.shape[1], 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.1),
        nn.Linear(64, 64), nn.SiLU(), nn.Dropout(0.1), nn.Linear(64, 4),
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.012)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=14, eta_min=0.0005)
    for _ in range(14):
        for idx in torch.randperm(len(y), generator=generator).split(1024):
            probability, location, raw_sigma, raw_direct = net(inputs[idx]).unbind(1)
            sigma = (F.softplus(raw_sigma) + 0.05).clamp(max=2)
            likelihood = F.binary_cross_entropy_with_logits(probability, positive[idx], reduction='none')
            likelihood += positive[idx] * (sigma.log() + 0.5 * ((log_amount[idx] - location) / sigma).square())
            direct = F.smooth_l1_loss(F.softplus(raw_direct), y[idx], beta=0.04, reduction='none')
            loss = ((likelihood + 0.80 * direct) * weights[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError('The spending model loss became nonfinite.')
        scheduler.step()
    net.eval()
    return [predict(net, x, scale) for x, (_, scale) in zip(other_inputs, evaluation, strict=True)]


ARTIFACTS = task_artifacts(3)


def main():
    torch.set_num_threads(2)
    train = pd.read_csv(T3 / 'train.csv')
    y = train.pop(TARGET).to_numpy()
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError('Training spending must be finite and nonnegative.')
    train = restore(train.drop(columns=ID), buckets=True)
    test = pd.read_csv(T3 / 'test.csv')
    identifiers = test.pop(ID)
    test = restore(test, buckets=True)
    if len(test) != ROWS or identifiers.isna().any() or not identifiers.is_unique or not test.columns.equals(train.columns):
        raise ValueError('Unexpected Task 3 test schema.')

    oof = np.empty(len(train))
    test_predictions = []
    for fold, (fit, validation) in enumerate(KFold(5, shuffle=True, random_state=42).split(train), 1):
        val, result = fit_fold(train.iloc[fit], y[fit], [train.iloc[validation], test], seed=342 + fold)
        oof[validation] = val
        test_predictions.append(result)
        print(f'Fold {fold} NMAE {nmae(y[validation], val):.6f}', flush=True)
    print(f'Out-of-fold NMAE {nmae(y, oof):.6f}', flush=True)
    np.save(ARTIFACTS / 'task3_mlp_oof.npy', oof)
    np.save(ARTIFACTS / 'task3_mlp.npy', np.mean(test_predictions, axis=0))


if __name__ == '__main__':
    main()

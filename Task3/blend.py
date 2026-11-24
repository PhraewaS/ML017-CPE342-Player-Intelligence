"""Task 3 final: 0.85 * LightGBM hurdle model + 0.15 * distribution MLP (weights chosen on out-of-fold NMAE).
Run Task3/spending_mixture.py and Task3/distribution_mlp.py first. Writes artifacts/task3.npy."""
from pathlib import Path

import numpy as np
import pandas as pd

from shared.config import task_artifacts, T3

W_MIXTURE, W_MLP = 0.85, 0.15


def nmae(y, p):
    return np.abs(y - p).sum() / np.abs(y).sum()


ARTIFACTS = task_artifacts(3)


def main():
    y = pd.read_csv(T3 / 'train.csv', usecols=['spending_30d'])['spending_30d'].to_numpy()
    oof = W_MIXTURE * np.load(ARTIFACTS / 'task3_mixture_oof.npy') + W_MLP * np.load(ARTIFACTS / 'task3_mlp_oof.npy')
    print(f'OOF NMAE {nmae(y, oof):.5f} -> task score 1/(1+NMAE) = {1 / (1 + nmae(y, oof)):.5f}')
    test = W_MIXTURE * np.load(ARTIFACTS / 'task3_mixture.npy') + W_MLP * np.load(ARTIFACTS / 'task3_mlp.npy')
    assert np.isfinite(test).all() and (test >= 0).all()
    np.save(ARTIFACTS / 'task3.npy', test)


if __name__ == '__main__':
    main()

"""Rebuild the ConvNeXt ensemble from the Modal fold outputs in runs/ (2 configs x 5 folds).
For each config: average the 5 fold models' test logits, log-softmax; then average the two configs.
Also prints each config's out-of-fold macro F1. Writes ../predictions/convnext_modal_ensemble.npy if --write."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from shared.config import T4_DATA


def main():
    y = pd.read_csv(T4_DATA / 'train.csv').label.values
    oofs, tests = [], []
    for run in sorted((HERE / 'runs').iterdir()):
        oof = np.zeros((len(y), 5)); test = 0
        for k in range(5):
            oof[np.load(run / f'idx_{k}.npy')] = np.load(run / f'oof_{k}.npy')
            test = test + np.load(run / f'test_{k}.npy') / 5
        cfg = json.load(open(run / 'cfg.json'))
        print(run.name, cfg['model'], f"{cfg['h']}x{cfg['w']}", 'OOF macro F1', round(f1_score(y, oof.argmax(1), average='macro'), 4))
        oofs.append(log_softmax(oof, axis=1)); tests.append(log_softmax(test, axis=1))
    ens = log_softmax(np.mean(tests, 0), axis=1)
    frozen = np.load(HERE.parent / 'predictions/convnext_modal_ensemble.npy')
    print('matches frozen predictions:', np.allclose(ens, frozen, atol=1e-6), '| same labels:', np.array_equal(ens.argmax(1), frozen.argmax(1)))
    if '--write' in sys.argv:
        np.save(HERE.parent / 'predictions/convnext_modal_ensemble.npy', ens)


if __name__ == '__main__':
    main()

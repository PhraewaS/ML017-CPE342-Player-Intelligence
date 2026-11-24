"""Compare cached screenshot encoders on held-out training images."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

from Task4.dino.t4_backbone_views import BACKBONES


def design(cache, indices, view):
    views = [0, 1] if view == 'both' else [0 if view == 'raw' else 1]
    z = np.asarray(cache[indices][:, views], dtype=np.float32)
    rows, _, _, width = z.shape
    return normalize(normalize(z.reshape(-1, width)).reshape(rows, -1))


def benchmark(args):
    frame = pd.read_csv(args.project / 'data_task4/train.csv')
    y = frame.label.to_numpy()
    review = args.project / 'work/review'
    if args.split:
        mask = pd.read_csv(args.split).set_index('id').loc[frame.id, 'is_val'].astype(bool).to_numpy()
        fit, val = np.flatnonzero(~mask), np.flatnonzero(mask)
    else:
        val = np.load(review / 't4_runs/229f19b8de97' / f'idx_{args.fold}.npy', allow_pickle=False)
        fit = np.setdiff1d(np.arange(len(frame)), val)
    baseline = np.load(review / 't4_oof_logprob_ens.npy', allow_pickle=False)[val]
    sorted_logits = np.sort(baseline, axis=1)
    hard = sorted_logits[:, -1] - sorted_logits[:, -2] < 1.75

    def report(name, logits):
        pred = logits.argmax(axis=1)
        full = f1_score(y[val], pred, average='macro', labels=range(5))
        low = f1_score(y[val][hard], pred[hard], average='macro', labels=range(5))
        print(f'{name:28s} full {full:.5f}, low-margin {low:.5f}', flush=True)

    print(f'fit={len(fit):,} validation={len(val):,} low-margin={sum(hard):,}', flush=True)
    if args.split is None:
        report('ConvNeXt reference', baseline)
    saved = []
    for name in BACKBONES:
        file = args.work / f't4_views_{name}_train.npy'
        if not file.exists():
            continue
        cache = np.load(file, allow_pickle=False, mmap_mode='r')
        views = {}
        for view in ('raw', 'canonical', 'both'):
            x = design(cache, fit, view)
            model = LogisticRegression(C=8, max_iter=350).fit(x, y[fit])
            p = model.predict_log_proba(design(cache, val, view))
            views[view] = p
            report(f'{name} {view}', p)
            if args.split is None:
                report(f'{name} {view} 20% blend', 0.8 * baseline + 0.2 * p)
        saved.append(views['both'])
    if len(saved) > 1:
        averaged = np.log(np.mean([softmax(p, axis=1) for p in saved], axis=0))
        report('two encoders, both views', averaged)
        if args.split is None:
            report('two encoders 20% blend', 0.8 * baseline + 0.2 * averaged)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('--work', type=Path, default=Path('work/review'))
    parser.add_argument('--fold', type=int, default=3)
    parser.add_argument('--split', type=Path)
    benchmark(parser.parse_args())

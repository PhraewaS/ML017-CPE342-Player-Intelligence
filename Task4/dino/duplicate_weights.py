"""Step 6 of the DINOv3 scene model: down-weight near-duplicate training screenshots (same scene, same class) by
1/sqrt(group size), floor 0.25, keeping each class's mean weight. Near-duplicates = scene-fingerprint similarity
>= 0.90 (see t4_scene_retrieval_audit.py). Train images only; held-out weights unchanged.
Recorded from the Codex session; run from the original working folder (project root with work/review/)."""
from pathlib import Path
from time import perf_counter
import sys
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

work = Path('work/review')
project = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data_task4')
frame = pd.read_csv(project / 'train.csv')
split = pd.read_csv(work / 't4_independent_split_strict.csv')
original = np.load(work / 't4_scene_rare_group_weights.npz', allow_pickle=False)
fingerprint = np.load(work / 't4_scene_fingerprints.npz', allow_pickle=False)
assert np.array_equal(frame.id, split.id) and np.array_equal(frame.id, fingerprint['id'])
assert np.array_equal(frame.id, original['ids'])
holdout = split.is_val.to_numpy(bool)
assert np.array_equal(holdout, original['holdout'])
y = frame.label.to_numpy()
descriptor = fingerprint['descriptor']
flip = np.tile(np.array([1, -1], dtype=np.float32), 80)
weights = original['weights'].astype(np.float32, copy=True)
assert weights.shape == (len(frame),)
start = perf_counter()
for k in np.unique(y):
    rows = np.flatnonzero((y == k) & ~holdout)
    candidates = np.moveaxis(descriptor[rows], 1, 0).copy()
    graph_i, graph_j = [], []
    for lo in range(0, len(rows), 128):
        raw = descriptor[rows[lo:lo + 128]]
        scores = []
        for q in (raw, raw * flip):
            s = 0.75 * np.abs(q[:, 0] @ candidates[0].T)
            s += 0.15 * (q[:, 1] @ candidates[1].T)
            s += 0.10 * (q[:, 2] @ candidates[2].T)
            scores.append(s)
        a, b = np.where(np.maximum(*scores) >= .90)
        a += lo
        unique = a < b
        graph_i.append(a[unique]); graph_j.append(b[unique])
    left, right = np.concatenate(graph_i), np.concatenate(graph_j)
    graph = coo_matrix((np.ones(len(left), dtype=np.uint8), (left, right)), shape=(len(rows), len(rows))).tocsr()
    count, groups = connected_components(graph, directed=False)
    sizes = np.bincount(groups, minlength=count)[groups]
    prior_weight = weights[rows].mean()
    weights[rows] *= np.maximum(0.25, 1 / np.sqrt(sizes))
    weights[rows] *= prior_weight / weights[rows].mean()
    print(f'Class {k}: {len(left):,} matched pairs, {(sizes > 1).sum():,} of {len(rows):,} rows with a near-match, '
          f'largest group {sizes.max():,}, {perf_counter() - start:.1f}s')
assert np.array_equal(weights[holdout], original['weights'][holdout]) and (weights > 0).all() and np.isfinite(weights).all()
with (work / 't4_scene_duplicate_weights.npz').open('xb') as stream:
    np.savez(stream, ids=original['ids'], holdout=holdout, weights=weights)

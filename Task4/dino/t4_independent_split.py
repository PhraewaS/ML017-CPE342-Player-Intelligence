"""Build a Task 4 concept holdout from labeled images and cached near-match edges."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


GROUPS_PER_CLASS = 18
MATCH_THRESHOLD = 0.90
TARGET_FRACTION = 0.20
SEED = 342


def scene_components(retrieval, size):
    left, ranks = np.where(retrieval['sim'] >= MATCH_THRESHOLD)
    right = retrieval['nearest'][left, ranks]
    if np.any((right < 0) | (right >= size)):
        raise ValueError('Cached near-match row does not exist.')
    graph = coo_matrix((np.ones(len(left), dtype=np.uint8), (left, right)), shape=(size, size)).tocsr()
    count, components = connected_components(graph, directed=False)
    return count, components, left, right


def concept_groups(cache, labels, classes):
    native = np.asarray(cache[:, :, 0], dtype=np.float32)
    size, views, width = native.shape
    embeddings = normalize(normalize(native.reshape(size * views, width)).reshape(size, -1))
    groups = np.empty(size, dtype=int)
    isolation = []
    for index, label in enumerate(classes):
        match = labels == label
        clustering = MiniBatchKMeans(
            n_clusters=GROUPS_PER_CLASS, n_init=3, batch_size=512, max_iter=50, random_state=SEED + index,
        ).fit(embeddings[match])
        groups[match] = index * GROUPS_PER_CLASS + clustering.labels_
        centers = normalize(clustering.cluster_centers_)
        similarity = centers @ centers.T
        np.fill_diagonal(similarity, -np.inf)
        isolation.extend(1 - similarity.max(axis=1))
    return groups, np.asarray(isolation)


def choose_holdout(counts, isolation):
    # Exhausting these 18 groups lets the holdout cover complete concepts while keeping class support.
    choices = np.arange(1, 1 << len(counts), dtype=np.uint32)
    masks = ((choices[:, None] >> np.arange(len(counts), dtype=np.uint32)) & 1).astype(np.float32)
    sizes = masks @ counts
    fraction = sizes / counts.sum()
    permitted = (fraction >= 0.16) & (fraction <= 0.24)
    if not np.any(permitted):
        raise ValueError('Could not reserve 16–24% of a class without splitting its concept groups.')
    score = (masks[permitted] @ (counts * isolation)) / sizes[permitted]
    score -= 0.5 * np.abs(fraction[permitted] - TARGET_FRACTION)
    return masks[permitted][np.argmax(score)].astype(bool)


def build(project, work):
    train = pd.read_csv(project / 'data_task4/train.csv')
    with np.load(work / 't4_scene_retrieval.npz', allow_pickle=False) as retrieval:
        if not np.array_equal(train.id.to_numpy(), retrieval['id']):
            raise ValueError('Near-match cache IDs differ from the training CSV.')
        count, components, left, right = scene_components(retrieval, len(train))
        old_fold = retrieval['fold']
    cache = np.load(work / 't4_views_dinov3_train.npy', mmap_mode='r', allow_pickle=False)
    labels = train.label.to_numpy()
    classes = np.unique(labels)
    if cache.shape[:3] != (len(train), 2, 5) or len(classes) != 5:
        raise ValueError('Unexpected training cache schema.')
    groups, isolation = concept_groups(cache, labels, classes)
    n_groups = len(isolation)
    # Assign each complete near-match component to the most frequent concept among its rows.
    votes = np.bincount(components * n_groups + groups, minlength=count * n_groups).reshape(count, n_groups)
    safe_groups = votes.argmax(axis=1)[components]
    held_out = np.zeros(n_groups, dtype=bool)
    for index, label in enumerate(classes):
        section = slice(index * GROUPS_PER_CLASS, (index + 1) * GROUPS_PER_CLASS)
        counts = np.bincount(safe_groups[labels == label], minlength=n_groups)[section]
        held_out[section] = choose_holdout(counts, isolation[section])
    is_val = held_out[safe_groups]
    if np.any(is_val[left] != is_val[right]):
        raise AssertionError('A cached near-match edge crosses the split.')
    print(f'{int(is_val.sum()):,} validation and {int((~is_val).sum()):,} fit rows.', flush=True)
    for label in classes:
        mask = labels == label
        held_count = int(is_val[mask].sum())
        print(f'class {label}: {held_count:,}/{int(mask.sum()):,} validation ({is_val[mask].mean():.2%})', flush=True)
        if not 0.14 <= is_val[mask].mean() <= 0.26:
            raise AssertionError(f'Class {label} falls outside desired holdout support.')
    print(f'{len(left):,} cached near-match links >= {MATCH_THRESHOLD:.2f}. '
          f'Old folds separated {int((old_fold[left] != old_fold[right]).sum()):,}. '
          f'New split separated {int((is_val[left] != is_val[right]).sum()):,}.', flush=True)
    large = np.bincount(components, minlength=count)
    print(f'{int((large > 1).sum()):,} multi-row near-match components, '
          f'maximum size {large.max()}, {int((groups != safe_groups).sum()):,} rows reassigned.', flush=True)
    result = pd.DataFrame({'id': train.id, 'is_val': is_val.astype(np.uint8)})
    result.to_csv(work / 't4_independent_split.csv', index=False, mode='x')


def boundary_matches(descriptors, frontier, fit):
    candidate = np.moveaxis(descriptors[fit], 1, 0).copy()
    flip = np.tile([1, -1], 80).astype(np.float32)
    matched = np.zeros(len(fit), dtype=bool)
    edges = 0
    for start in range(0, len(frontier), 128):
        queries = descriptors[frontier[start:start + 128]]
        variants = []
        for view in (queries, queries * flip):
            s = 0.75 * np.abs(view[:, 0] @ candidate[0].T)
            s += 0.15 * (view[:, 1] @ candidate[1].T)
            s += 0.10 * (view[:, 2] @ candidate[2].T)
            variants.append(s)
        close = np.maximum(*variants) >= MATCH_THRESHOLD
        matched |= close.any(axis=0)
        edges += int(close.sum())
    return fit[matched], edges


def build_strict(project, work):
    train = pd.read_csv(project / 'data_task4/train.csv')
    split = pd.read_csv(work / 't4_independent_split.csv')
    with np.load(work / 't4_scene_fingerprints.npz', allow_pickle=False) as fingerprint:
        if not np.array_equal(train.id, split.id) or not np.array_equal(train.id, fingerprint['id']):
            raise ValueError('Fingerprint cache or split IDs differ from labeled training IDs.')
        descriptors = fingerprint['descriptor']
    is_val = split.is_val.to_numpy(dtype=bool, copy=True)
    initial = is_val.sum()
    frontier = np.flatnonzero(is_val)
    # The first pass audits every validation/fit pair. Later passes check only the newly added rows.
    while len(frontier):
        frontier, edges = boundary_matches(descriptors, frontier, np.flatnonzero(~is_val))
        print(f'Full-threshold audit found {edges:,} links and promoted {len(frontier):,} rows.', flush=True)
        is_val[frontier] = True
    _, remaining = boundary_matches(descriptors, np.flatnonzero(is_val), np.flatnonzero(~is_val))
    if remaining:
        raise AssertionError(f'{remaining} links still cross the strict split.')
    print(f'Strict split holds out {is_val.sum():,} rows (+{is_val.sum() - initial:,}) with zero descriptor links crossing.')
    for label in sorted(train.label.unique()):
        mask = train.label == label
        print(f'class {label}: {is_val[mask].sum():,}/{mask.sum():,} validation ({is_val[mask].mean():.2%})')
        if not 0.14 <= is_val[mask].mean() <= 0.26:
            raise AssertionError(f'Class {label} falls outside desired holdout support.')
    split.assign(is_val=is_val.astype(np.uint8)).to_csv(work / 't4_independent_split_strict.csv', index=False, mode='x')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('--work', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    if args.strict:
        build_strict(args.project, args.work)
    else:
        build(args.project, args.work)

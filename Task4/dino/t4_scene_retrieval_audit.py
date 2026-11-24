"""Train-only near-duplicate check for the Task 4 image split."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter
from sklearn.metrics import f1_score

from t4_rotation_probe import padding_angle


def descriptor(image):
    angle = padding_angle(image)
    if angle:
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
    # Emphasize the shared center to reduce the effect of missing corners.
    image = image.crop((34, 26, 196, 94)).resize((40, 20), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, np.float32)
    low, high = rgb.min(2), rgb.max(2)
    light = (low + high) * 0.5
    chroma = high - low
    # HSL lightness survives hue remapping. Its sign may flip under RGB inversion.
    blur = gaussian_filter(light, 0.85)
    edge = np.hypot(*np.gradient(blur))
    parts = []
    for channel in (light, chroma, edge):
        descriptor = dctn(channel, norm='ortho')[:10, :16].copy().ravel()
        descriptor[0] = 0
        descriptor /= max(1e-6, np.linalg.norm(descriptor))
        parts.append(descriptor)
    return np.stack(parts), angle


def fingerprints(frame, directory, save):
    if save.exists():
        stored = np.load(save)
        assert np.array_equal(stored['id'], frame.id.to_numpy())
        return stored['descriptor'], stored['angle']
    vectors = np.empty((len(frame), 3, 160), np.float32)
    angles = np.empty(len(frame), np.float32)
    for i, name in enumerate(frame.file_name):
        with Image.open(directory / name) as image:
            vectors[i], angles[i] = descriptor(image.convert('RGB'))
        if i % 4000 == 0:
            print('images', i, flush=True)
    np.savez(save, id=np.asarray(frame.id, dtype='U'), descriptor=vectors, angle=angles)
    return vectors, angles


def cross_fold_neighbors(vectors, fold, top=3):
    n = len(vectors)
    sims = np.full((n, top), np.nan, np.float32)
    refs = np.full((n, top), -1, np.int32)
    flipped = vectors * np.tile([1, -1], 80).astype(np.float32)
    for k in np.unique(fold):
        possible = np.flatnonzero(fold != k)
        queries = np.flatnonzero(fold == k)
        valid = np.moveaxis(vectors[possible], 1, 0).copy()
        for offset in range(0, len(queries), 128):
            ix = queries[offset:offset + 128]
            q, q_flip = vectors[ix], flipped[ix]
            def score(sample):
                light = np.abs(sample[:, 0] @ valid[0].T)
                chroma = sample[:, 1] @ valid[1].T
                edge = sample[:, 2] @ valid[2].T
                return 0.75 * light + 0.15 * chroma + 0.10 * edge
            sim = np.maximum(score(q), score(q_flip))
            nearest = np.argpartition(-sim, top - 1, axis=1)[:, :top]
            available = np.take_along_axis(sim, nearest, axis=1)
            order = np.argsort(-available, axis=1)
            refs[ix] = possible[np.take_along_axis(nearest, order, axis=1)]
            sims[ix] = np.take_along_axis(available, order, axis=1)
        print('fold matched', k, flush=True)
    return sims, refs


def contact(frame, directory, sim, index, output):
    from PIL import ImageDraw
    rng = np.random.default_rng(4)
    rows = []
    for lo, hi in [(0.96, 1), (0.9, 0.96), (0.84, 0.9), (0.75, 0.84)]:
        pool = np.flatnonzero((sim >= lo) & (sim < hi))
        if len(pool):
            rows.extend(rng.choice(pool, min(6, len(pool)), replace=False))
    board = Image.new('RGB', (840, ((len(rows) + 2) // 3) * 112), (0, 0, 0))
    draw = ImageDraw.Draw(board)
    for pos, src in enumerate(rows):
        dst = index[src]
        x, y = pos % 3 * 280, pos // 3 * 112
        for align, j in enumerate([src, dst]):
            row = frame.iloc[j]
            with Image.open(directory / row.file_name) as pic:
                board.paste(pic.resize((135, 70)), (x + align * 138, y))
            draw.text((x + align * 138, y + 70), f'{row.id}: class {row.label}', (255, 255, 255))
        draw.text((x, y + 84), f'Score {sim[src]:.3f}', (255, 220, 160))
    board.save(output)


def run(project, output):
    folder = project / 'data_task4/train'
    frame = pd.read_csv(project / 'data_task4/train.csv')
    reference = project / 'work/review'
    run = sorted((reference / 't4_runs').iterdir())[0]
    fold = np.full(len(frame), -1, np.int8)
    for k in range(5):
        fold[np.load(run / f'idx_{k}.npy')] = k
    assert fold.min() == 0
    vectors, angles = fingerprints(frame, folder, output / 't4_scene_fingerprints.npz')
    print('rotation detected', np.count_nonzero(angles), '/', len(frame), flush=True)
    sim, nearest = cross_fold_neighbors(vectors, fold)
    np.savez(output / 't4_scene_retrieval.npz', id=np.asarray(frame.id, dtype='U'),
             sim=sim, nearest=nearest, angle=angles, fold=fold)
    pred = np.load(reference / 't4_oof_logprob_ens.npy').argmax(1)
    y = frame.label.to_numpy()
    s, match = sim[:, 0], y[nearest[:, 0]]
    bins = [(0, 0.72), (0.72, 0.8), (0.8, 0.86), (0.86, 0.9), (0.9, 0.94), (0.94, 0.97), (0.97, 1.00001)]
    print('Nearest similarity bin: n, nearest accuracy, OOF accuracy, OOF macro-F1, oracle override wins/losses', flush=True)
    for lo, hi in bins:
        ix = (lo <= s) & (s < hi)
        print((lo, hi), ix.sum(),
              round(np.mean(match[ix] == y[ix]), 4),
              round(np.mean(pred[ix] == y[ix]), 4),
              round(f1_score(y[ix], pred[ix], average='macro'), 4),
              (ix & (pred != y) & (match == y)).sum(),
              (ix & (pred == y) & (match != y)).sum(), flush=True)
    contact(frame, folder, s, nearest[:, 0], output / 't4_scene_match_contact.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    run(options.project, options.output)

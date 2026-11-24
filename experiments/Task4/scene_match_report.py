"""Report held-out training errors by similarity to the nearest scene."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score


def contact(frame, directory, sim, index, output):
    from PIL import ImageDraw
    rng = np.random.default_rng(4)
    rows = []
    for lo, hi in [(0.96, 1), (0.9, 0.96), (0.84, 0.9), (0.75, 0.84)]:
        pool = np.flatnonzero((sim >= lo) & (sim < hi))
        if len(pool):
            rows.extend(rng.choice(pool, min(6, len(pool)), replace=False))
    if not rows:
        return
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
    with np.load(output / 't4_scene_retrieval.npz', allow_pickle=False) as cache:
        if not np.array_equal(cache['id'], frame.id.to_numpy()):
            raise ValueError('Scene matches do not follow the training image order.')
        sim, nearest = cache['sim'], cache['nearest']
    pred = np.load(reference / 't4_oof_logprob_ens.npy').argmax(1)
    y = frame.label.to_numpy()
    s, match = sim[:, 0], y[nearest[:, 0]]
    bins = [(0, 0.72), (0.72, 0.8), (0.8, 0.86), (0.86, 0.9), (0.9, 0.94), (0.94, 0.97), (0.97, 1.00001)]
    print('Nearest similarity bin: n, nearest accuracy, OOF accuracy, OOF macro-F1, oracle override wins/losses', flush=True)
    for lo, hi in bins:
        ix = (lo <= s) & (s < hi)
        if not ix.any():
            print((lo, hi), 0, flush=True)
            continue
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

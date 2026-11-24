"""Extract cached image patches for the DINOv3 training weights."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image


def extract(args):
    import torch

    from t4_scene_finetune import DEVICE, SIZE, denoise, encoder, smoother, to_device

    if not torch.backends.mps.is_available():
        raise RuntimeError('The patch extractor requires the Apple GPU.')
    torch.set_num_threads(2)
    frame = pd.read_csv(args.project / 'data_task4/train.csv')
    source = args.project / 'data_task4/train'
    model = encoder(args, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()
    mean = torch.tensor(model.pretrained_cfg['mean'], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(model.pretrained_cfg['std'], device=DEVICE).view(1, 3, 1, 1)
    kernel = smoother()
    pending = args.cache.with_suffix('.partial')
    if args.cache.exists():
        raise FileExistsError(args.cache)
    shape = (len(frame), 33, model.num_features)
    first = 0
    if pending.exists():
        cache = np.lib.format.open_memmap(pending, mode='r+')
        if cache.shape != shape or cache.dtype != np.float16:
            raise ValueError('Unexpected partial patch feature cache.')
        for lo in range(0, len(frame), 32):
            batch = cache[lo:lo + 32]
            if not np.isfinite(batch).all() or not np.any(batch != 0, axis=-1).all():
                break
            first = lo
    else:
        cache = np.lib.format.open_memmap(pending, mode='w+', dtype=np.float16, shape=shape)
    start = perf_counter()
    with torch.inference_mode():
        for lo in range(first, len(frame), 32):
            views = []
            for name in frame.file_name.iloc[lo:lo + 32]:
                with Image.open(source / name) as image:
                    views.append(np.asarray(image.convert('RGB').resize(SIZE[::-1], Image.Resampling.BICUBIC)))
            x = (denoise(to_device(np.stack(views)), kernel) - mean) / std
            with torch.autocast('mps', dtype=torch.bfloat16):
                tokens = model.forward_features(x)
                whole = model.forward_head(tokens, pre_logits=True).float()
            patches = tokens[:, model.num_prefix_tokens:].float().reshape(len(x), 4, 2, 8, 2, -1)
            tiles = patches.mean((2, 4)).flatten(1, 2)
            cache[lo:lo + len(x)] = torch.cat((whole[:, None], tiles), dim=1).cpu().numpy()
            if lo % 1920 == 0:
                cache.flush()
                print(f'{lo + len(x):,}/{len(frame):,} patches cached in {perf_counter() - start:.1f}s', flush=True)
    cache.flush()
    del cache
    pending.rename(args.cache)


def group_weights(cache, rows, labels):
    import torch
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    weights = np.ones(len(cache), dtype=np.float32)
    for cl in range(5):
        group = rows[labels[rows] == cl]
        features = normalize(np.asarray(cache[group, 0], dtype=np.float32))
        clusters = MiniBatchKMeans(n_clusters=16, batch_size=512, n_init=3, random_state=342)
        assignment = clusters.fit_predict(features)
        sizes = np.bincount(assignment, minlength=16)
        correction = np.clip(np.sqrt((len(group) / 16) / sizes[assignment]), 0.55, 2.5)
        weights[group] = correction / correction.mean()
        print(f'Game {cl} fitting group sizes: {sizes.min()} to {sizes.max()}', flush=True)
    return torch.from_numpy(weights)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('project', type=Path)
    ap.add_argument('--work', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--cache', required=True, type=Path)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--extract', action='store_true', required=True)
    options = ap.parse_args()
    options.backbone = 'dinov3'
    extract(options)

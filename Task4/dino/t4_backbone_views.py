"""Compare pretrained screenshot encoders and a canonical view using labeled images only."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
import timm
import torch

from t4_rotation_probe import padding_angle, rotate


SIZE = (128, 256)
BACKBONES = {
    'dinov3': ('vit_small_patch16_dinov3.lvd1689m', 'dinov3_small.safetensors'),
    'siglip2': ('vit_base_patch16_siglip_224.v2_webli', 'siglip2_base.safetensors'),
}


def canonical(image):
    upright = rotate(image, padding_angle(image))
    gray = upright.convert('L').filter(ImageFilter.GaussianBlur(radius=0.65))
    gray = ImageOps.autocontrast(gray, cutoff=2).convert('RGB')
    color = ImageOps.autocontrast(upright, cutoff=2)
    return Image.blend(gray, color, 0.30)


def images_for_batch(paths):
    image_batches = ([], [])
    for path in paths:
        with Image.open(path) as source:
            rgb = source.convert('RGB')
        image_batches[0].append(np.asarray(rgb.resize(SIZE[::-1], Image.Resampling.BICUBIC)))
        image_batches[1].append(np.asarray(canonical(rgb).resize(SIZE[::-1], Image.Resampling.BICUBIC)))
    return np.concatenate([np.stack(view) for view in image_batches])


def extract(args):
    if not torch.backends.mps.is_available():
        raise RuntimeError('An Apple GPU is required for this feature-extraction experiment.')
    name, checkpoint = BACKBONES[args.backbone]
    model = timm.create_model(
        name,
        pretrained=True,
        pretrained_cfg_overlay={'file': str((args.work / checkpoint).resolve())},
        img_size=SIZE,
        num_classes=0,
    ).to('mps').eval()
    mean = torch.tensor(model.pretrained_cfg['mean'], device='mps').view(1, 3, 1, 1)
    std = torch.tensor(model.pretrained_cfg['std'], device='mps').view(1, 3, 1, 1)
    frame = pd.read_csv(args.project / 'data_task4/train.csv')
    paths = [args.project / 'data_task4/train' / name for name in frame.file_name]
    output = args.work / f't4_views_{args.backbone}_train.npy'
    pending = output.with_suffix('.partial')
    if output.exists():
        raise FileExistsError(output)
    start = perf_counter()
    shape = (len(frame), 2, 5, model.num_features)
    first = 0
    if args.resume:
        features = np.lib.format.open_memmap(pending, mode='r+')
        if features.shape != shape or features.dtype != np.float16:
            raise ValueError(f'Unexpected partial feature cache schema: {features.shape}, {features.dtype}')
        for offset in range(0, len(frame), args.batch_size):
            block = features[offset:offset + args.batch_size]
            if not np.isfinite(block).all() or not np.any(block != 0, axis=-1).all():
                break
            first = offset
        print(f'{args.backbone}: resuming at row {first:,}, including a repeated batch.', flush=True)
    else:
        if pending.exists():
            raise FileExistsError(pending)
        features = np.lib.format.open_memmap(pending, mode='w+', dtype='float16', shape=shape)
    with torch.inference_mode():
        for offset in range(first, len(frame), args.batch_size):
            pixels = images_for_batch(paths[offset:offset + args.batch_size])
            rows = len(pixels) // 2
            x = torch.from_numpy(pixels).to('mps').permute(0, 3, 1, 2).float() / 255
            x = (x - mean) / std
            with torch.autocast('mps', dtype=torch.bfloat16):
                tokens = model.forward_features(x)
                native = model.forward_head(tokens, pre_logits=True).float()
            spatial = tokens[:, model.num_prefix_tokens:].float().reshape(len(x), 8, 16, -1)
            vectors = [native, spatial.mean(dim=(1, 2))]
            for first, last in [(0, 3), (2, 6), (5, 8)]:
                vectors.append(spatial[:, first:last].mean(dim=(1, 2)))
            batch = torch.stack(vectors, dim=1).cpu().numpy().reshape(2, rows, 5, -1)
            features[offset:offset + rows] = batch.transpose(1, 0, 2, 3)
            if offset % (args.batch_size * 60) == 0:
                features.flush()
                print(f'{args.backbone}: {offset + rows:,}/{len(frame):,} in {perf_counter() - start:.1f}s', flush=True)
    features.flush()
    del features
    pending.rename(output)
    print(f'Completed {output}, {perf_counter() - start:.1f}s', flush=True)


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
    parser.add_argument('--work', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--backbone', choices=BACKBONES)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--fold', type=int, default=3)
    parser.add_argument('--split', type=Path)
    options = parser.parse_args()
    if options.backbone:
        extract(options)
    else:
        benchmark(options)

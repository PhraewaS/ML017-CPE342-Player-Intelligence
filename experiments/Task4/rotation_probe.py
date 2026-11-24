"""Disposable Task 4 probe. Does undoing rotation improve frozen features on held-out scenes?"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

from Task4.dino.image_rotation import padding_angle, rotate


def score(name, probs, labels, hard):
    pred = probs.argmax(axis=1)
    all_score = f1_score(labels, pred, average='macro')
    hard_score = f1_score(labels[hard], pred[hard], average='macro', labels=np.arange(5))
    print(f'{name:24s} full={all_score:.5f}  small-margin={hard_score:.5f}', flush=True)


def run_probe(project, n_fit, n_val):
    import timm
    import torch
    from time import perf_counter

    torch.set_num_threads(4)
    data = project / 'data_task4'
    frame = pd.read_csv(data / 'train.csv')
    review = project / 'work/review'
    folds = sorted((review / 't4_runs').iterdir())[0]
    validation = np.load(folds / 'idx_0.npy', allow_pickle=False)
    training = np.setdiff1d(np.arange(len(frame)), validation)
    rng = np.random.default_rng(342)
    training = rng.choice(training, min(n_fit, len(training)), replace=False)
    validation = rng.choice(validation, min(n_val, len(validation)), replace=False)
    reference = np.load(review / 't4_oof_logprob_ens.npy', allow_pickle=False)[validation]
    sorted_ref = np.sort(reference, axis=1)
    margin = sorted_ref[:, -1] - sorted_ref[:, -2]
    hard = margin <= np.quantile(margin, 0.10)
    sample = frame.iloc[np.r_[training, validation]]
    labels = sample.label.to_numpy()
    model = timm.create_model('convnext_tiny.fb_in22k', pretrained=True, num_classes=0).eval()
    model.to(memory_format=torch.channels_last)
    mean = np.array(model.pretrained_cfg['mean'], dtype=np.float32)
    std = np.array(model.pretrained_cfg['std'], dtype=np.float32)
    all_raw, all_fixed, all_angles = [], [], []
    start = perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(sample), 48):
            images, corrected, changed = [], [], []
            filenames = sample.file_name.iloc[offset:offset + 48]
            for name in filenames:
                with Image.open(data / 'train' / name) as src:
                    image = src.convert('RGB')
                angle = padding_angle(image)
                all_angles.append(angle)
                images.append(np.asarray(image.resize((128, 64), Image.Resampling.BILINEAR)))
                if angle:
                    corrected.append(np.asarray(rotate(image, angle).resize((128, 64), Image.Resampling.BILINEAR)))
                    changed.append(len(images) - 1)
            pixels = np.stack(images + corrected).astype(np.float32) / 255
            x = torch.from_numpy((pixels - mean) / std).permute(0, 3, 1, 2)
            features = model(x.contiguous(memory_format=torch.channels_last)).numpy()
            raw = features[:len(images)]
            corrected = raw.copy()
            corrected[changed] = features[len(images):]
            all_raw.append(raw)
            all_fixed.append(corrected)
            if offset % 960 == 0:
                print(f'features {offset + len(images)}/{len(sample)} in {perf_counter() - start:.1f}s', flush=True)
    print(f'Detected rotation in {np.count_nonzero(all_angles):,}/{len(all_angles):,} images.')
    raw = normalize(np.concatenate(all_raw))
    fixed = normalize(np.concatenate(all_fixed))
    fit_size = len(training)
    y = labels[fit_size:]
    score('ConvNeXt reference', reference, y, hard)
    for name, x in [('Raw', raw), ('Straightened', fixed), ('Both views', np.c_[raw, fixed] / np.sqrt(2))]:
        model = LogisticRegression(C=10, max_iter=500)
        model.fit(x[:fit_size], labels[:fit_size])
        p = model.predict_log_proba(x[fit_size:])
        score(name, p, y, hard)
        score(f'{name} + reference', reference + p, y, hard)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project', type=Path)
    parser.add_argument('--train', type=int, default=3600)
    parser.add_argument('--validation', type=int, default=2000)
    args = parser.parse_args()
    run_probe(args.project, args.train, args.validation)

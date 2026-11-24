"""Train a DINOv2-Small Task 4 model locally and compare it on the original held-out split."""

import argparse
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
import timm
import torch
from torch.nn import functional as F


MODEL = 'vit_small_patch14_dinov2.lvd142m'
SIZE = (112, 224)
DEVICE = torch.device('mps')


def read_images(frame, folder, size=SIZE, denoise_input=False):
    if denoise_input:
        from scipy.ndimage import gaussian_filter

    images = np.empty((len(frame), size[0], size[1], 3), dtype=np.uint8)
    for index, name in enumerate(frame.file_name):
        with Image.open(folder / name) as source:
            image = source.convert('RGB')
        if denoise_input:
            pixels = np.asarray(image, dtype=np.float32) / 255
            pixels = gaussian_filter(pixels, (0.85, 0.85, 0))
            image = Image.fromarray(np.uint8(np.clip(pixels, 0, 1) * 255))
        images[index] = np.asarray(image.resize(size[::-1], Image.Resampling.BICUBIC))
    return images


def to_device(images):
    return torch.from_numpy(images).to(DEVICE).permute(0, 3, 1, 2).float() / 255


def augment(x):
    n = len(x)
    rng = torch.rand(10, n, device=DEVICE)
    turns = (rng[0] - 0.5) * (math.pi / 5) * (rng[1] < 0.65)
    scale = 0.93 + 0.18 * rng[2]
    flip = 1 - 2 * (rng[3] < 0.5).float()
    cosine = turns.cos() * scale
    sine = turns.sin() * scale
    dx, dy = (rng[4] - 0.5) * 0.10, (rng[5] - 0.5) * 0.10
    affine = torch.stack((cosine * flip, -0.5 * sine, dx, 2 * sine * flip, cosine, dy), dim=1)
    grid = F.affine_grid(affine.reshape(n, 2, 3), x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False)
    gray = x.mean(dim=1, keepdim=True)
    x = gray + (x - gray) * (0.45 + 1.1 * rng[6, :, None, None, None])
    x = (x - 0.5) * (0.75 + 0.5 * rng[7, :, None, None, None]) + 0.5
    x = x * (0.75 + 0.5 * rng[8, :, None, None, None])
    x = torch.where(rng[9, :, None, None, None] < 0.12, 1 - x, x)
    return x.clamp_(0, 1)


@torch.inference_mode()
def predict(model, inputs, batch_size, mean, std):
    model.eval()
    predictions = []
    for i in range(0, len(inputs), batch_size):
        x = (to_device(inputs[i:i + batch_size]) - mean) / std
        with torch.autocast('mps', dtype=torch.bfloat16):
            logits = model(x).float() + model(x.flip((3,))).float()
        predictions.append(F.log_softmax(logits / 2, dim=1).cpu().numpy())
    return np.concatenate(predictions)


def score(scores, labels):
    return f1_score(labels, scores.argmax(axis=1), average='macro', labels=np.arange(5))


def configure_model(checkpoint, warm_start):
    model = timm.create_model(
        MODEL,
        pretrained=warm_start is None,
        pretrained_cfg_overlay={'file': str(checkpoint.resolve())},
        num_classes=5,
        img_size=SIZE,
        drop_path_rate=0.12,
    )
    if warm_start:
        state = torch.load(warm_start, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    for parameter in model.patch_embed.parameters():
        parameter.requires_grad_(False)
    model.pos_embed.requires_grad_(False)
    model.cls_token.requires_grad_(False)
    for block in model.blocks[:4]:
        block.requires_grad_(False)
    return model


def train(args):
    if not torch.backends.mps.is_available():
        raise RuntimeError('This experiment requires the local Apple GPU.')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(3)
    data = args.project / 'data_task4'
    frame = pd.read_csv(data / 'train.csv')
    artifacts = args.project / 'work/review'
    indices = artifacts / 't4_runs/229f19b8de97' / f'idx_{args.fold}.npy'
    val = np.load(indices, allow_pickle=False)
    fit = np.setdiff1d(np.arange(len(frame)), val)
    if args.warm_start:
        previous_val = np.load(args.warm_start.with_name('val_indices.npy'), allow_pickle=False)
        if not np.array_equal(val, previous_val):
            raise ValueError('The checkpoint and requested validation fold differ.')
    if args.smoke:
        fit = rng.choice(fit, 384, replace=False)
        val = rng.choice(val, 200, replace=False)
    Xtrain = read_images(frame.iloc[fit], data / 'train', denoise_input=args.denoise_input)
    Xval = read_images(frame.iloc[val], data / 'train', denoise_input=args.denoise_input)
    y = frame.label.to_numpy(int)
    ytrain, yval = y[fit], y[val]
    ref = np.load(artifacts / 't4_oof_logprob_ens.npy', allow_pickle=False)[val]
    ranked = np.sort(ref, axis=1)
    low = (ranked[:, -1] - ranked[:, -2]) < 1.75
    print(f'Fold {args.fold}, n_train {len(fit):,}, n_val {len(val):,}, baseline F1 {score(ref, yval):.5f}', flush=True)

    model = configure_model(args.checkpoint, args.warm_start)
    mean = torch.tensor(model.pretrained_cfg['mean'], device=DEVICE).reshape(1, 3, 1, 1)
    std = torch.tensor(model.pretrained_cfg['std'], device=DEVICE).reshape(1, 3, 1, 1)
    logprior = torch.from_numpy(np.log(np.bincount(ytrain, minlength=5) / len(ytrain))).float().to(DEVICE)
    params = {'backbone': [], 'head': []}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params['head' if name.startswith('head.') else 'backbone'].append(param)
    lr_scale = 0.45 if args.warm_start else 1.0
    optimizer = torch.optim.AdamW([
        {'params': params['backbone'], 'lr': 0.00013 * lr_scale},
        {'params': params['head'], 'lr': 0.0006 * lr_scale},
    ], weight_decay=0.05)
    steps = math.ceil(len(fit) / args.batch_size)
    epochs = 1 if args.smoke else args.epochs
    warmup = max(1, steps // (10 if args.warm_start else 2))
    total = steps * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
        min(1, (step + 1) / warmup) * (0.10 + 0.90 * (1 + math.cos(math.pi * step / total)) / 2)
    ))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
        np.save(args.out / 'val_indices.npy', val)
    if args.warm_start:
        initial = predict(model, Xval, args.batch_size, mean, std)
        print(f'Starting checkpoint F1 {score(initial, yval):.5f}, blend {score(0.8 * ref + 0.2 * initial, yval):.5f}', flush=True)
        if args.out:
            np.save(args.out / 'val_initial.npy', initial)
    best = -1.0
    for epoch in range(epochs):
        start = perf_counter()
        model.train()
        losses = []
        order = rng.permutation(len(Xtrain))
        for step, positions in enumerate(np.array_split(order, steps), 1):
            x = (augment(to_device(Xtrain[positions])) - mean) / std
            target = torch.from_numpy(ytrain[positions]).to(DEVICE)
            operator = rng.random()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('mps', dtype=torch.bfloat16):
                if operator < 0.35:
                    mix = float(rng.beta(0.3, 0.3))
                    logits = model(mix * x + (1 - mix) * x.roll(1, 0)).float() + logprior
                    loss = mix * F.cross_entropy(logits, target, label_smoothing=0.05)
                    loss += (1 - mix) * F.cross_entropy(logits, target.roll(1, 0), label_smoothing=0.05)
                else:
                    logits = model(x).float() + logprior
                    loss = F.cross_entropy(logits, target, label_smoothing=0.05)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.detach())
            if step % 100 == 0:
                running = float(torch.stack(losses).mean().cpu())
                print(f'Epoch {epoch + 1} step {step}/{steps}, loss {running:.4f}, {perf_counter() - start:.0f}s', flush=True)
        aggregate = float(torch.stack(losses).mean().cpu())
        if not math.isfinite(aggregate):
            raise RuntimeError('The training loss is non-finite.')
        logits = predict(model, Xval, args.batch_size, mean, std)
        blend = 0.8 * ref + 0.2 * logits
        blend_score = score(blend, yval)
        hard_score = score(blend[low], yval[low]) if low.any() else float('nan')
        print(f'Epoch {epoch + 1}/{epochs}, loss {aggregate:.4f}, F1 {score(logits,yval):.5f}, blend {blend_score:.5f}, low-margin blend {hard_score:.5f}, {perf_counter()-start:.0f}s', flush=True)
        if args.out:
            np.save(args.out / f'val_epoch{epoch + 1}.npy', logits)
            if blend_score > best:
                best = blend_score
                tmp = args.out / 'weights.tmp'
                torch.save({key: value.cpu() for key, value in model.state_dict().items()}, tmp)
                tmp.replace(args.out / 'weights.pt')


def predict_test(args):
    if not torch.backends.mps.is_available():
        raise RuntimeError('This inference run requires the local Apple GPU.')
    data = args.project / 'data_task4'
    frame = pd.read_csv(data / 'test.csv')
    sample = pd.read_csv(args.project / 'cpe342-kaggle-main/sample_submission.csv')
    if len(frame) != 25889 or frame.id.tolist() != sample.id.tolist():
        raise ValueError('Task 4 test order differs from the submission template.')
    model = timm.create_model(MODEL, pretrained=False, num_classes=5, img_size=SIZE)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True), strict=True)
    model = model.to(DEVICE)
    mean = torch.tensor(model.pretrained_cfg['mean'], device=DEVICE).reshape(1, 3, 1, 1)
    std = torch.tensor(model.pretrained_cfg['std'], device=DEVICE).reshape(1, 3, 1, 1)
    images = read_images(frame, data / 'test', denoise_input=args.denoise_input)
    logits = predict(model, images, args.batch_size, mean, std)
    if logits.shape != (25889, 5) or not np.isfinite(logits).all():
        raise ValueError('Expected one finite five-class log probability vector per image.')
    with args.predictions.open('xb') as stream:
        np.save(stream, logits, allow_pickle=False)
    print(f'Wrote {len(logits):,} log-probability vectors to {args.predictions}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--fold', type=int, default=3)
    parser.add_argument('--seed', type=int, default=342)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--denoise-input', action='store_true')
    options = parser.parse_args()
    if bool(options.weights) != bool(options.predictions):
        parser.error('--weights and --predictions must be supplied together')
    if options.weights:
        predict_test(options)
    else:
        train(options)

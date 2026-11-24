"""Fine-tune a Task 4 encoder against a train-only scene holdout."""

import argparse
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import timm
import torch
from torch.nn import functional as F

from t4_backbone_views import BACKBONES, SIZE
from t4_local_finetune import DEVICE, read_images, score, to_device


SCENE_BACKBONES = {
    **BACKBONES,
    'dinov3_base': ('vit_base_patch16_dinov3.lvd1689m', 'dinov3_base.safetensors'),
    'convnext': ('convnext_small.fb_in22k', None),
    'convnext_dino': ('convnext_tiny.dinov3_lvd1689m', 'convnext_dinov3_tiny.safetensors'),
}


def smoother():
    x = torch.arange(-3, 4, device=DEVICE).float()
    weights = torch.exp(-x.square() / (2 * 0.92**2))
    kernel = torch.outer(weights, weights)
    return (kernel / kernel.sum()).expand(3, 1, 7, 7).contiguous()


def denoise(x, kernel):
    return F.conv2d(F.pad(x, (3, 3, 3, 3), mode='replicate'), kernel, groups=3)


def training_view(x, kernel):
    n = len(x)
    jitter = torch.rand(10, n, device=DEVICE, dtype=torch.float32)
    x = torch.where(jitter[0, :, None, None, None] < 0.60, denoise(x, kernel), x)
    angle = (jitter[1] - 0.5) * math.radians(18)
    zoom = 0.94 + 0.15 * jitter[2]
    sign = 1 - 2 * (jitter[3] < 0.5).float()
    c, s = zoom * angle.cos(), zoom * angle.sin()
    dx, dy = 0.075 * (jitter[4] - 0.5), 0.075 * (jitter[5] - 0.5)
    affine = torch.stack((c * sign, -0.5 * s, dx, 2 * s * sign, c, dy), dim=1).reshape(n, 2, 3)
    x = F.grid_sample(x, F.affine_grid(affine, x.size(), align_corners=False), align_corners=False)
    x = (x - 0.5) * (0.75 + 0.5 * jitter[6, :, None, None, None]) + 0.5
    x = x * (0.78 + 0.44 * jitter[7, :, None, None, None])
    gray = x.mean(1, keepdim=True)
    x = gray + (x - gray) * (0.52 + 0.92 * jitter[8, :, None, None, None])
    x = torch.where(jitter[9, :, None, None, None] < 0.06, 1 - x, x)
    return x.clamp_(0, 1)


def style_view(x):
    jitter = torch.rand(6, len(x), 1, 1, 1, device=x.device)
    spread, mean = torch.var_mean(x, dim=(2, 3), correction=0, keepdim=True)
    scale = (spread + 1e-5).sqrt()
    strength = 0.60 * jitter[0]
    target_scale = scale.lerp(scale.roll(1, 0), strength)
    target_mean = mean.lerp(mean.roll(1, 0), strength)
    x = ((x - mean) / scale * target_scale + target_mean).clamp(0, 1)
    gamma = torch.exp((2 * jitter[1] - 1) * 0.62)
    x = x.pow(gamma)
    gray = x.mean(1, keepdim=True)
    x = x.lerp(gray, 0.55 * jitter[2])
    x = (x - 0.5) * (0.72 + 0.65 * jitter[3]) + 0.5
    x = torch.where(jitter[4] < 0.15, 1 - x, x)
    return (x + (0.045 * jitter[5]) * torch.randn_like(x)).clamp(0, 1)


def paired_loss(primary, altered, targets):
    supervised = [F.cross_entropy(scores, targets, label_smoothing=0.035, reduction='none') for scores in (primary, altered)]
    a, b = F.log_softmax(primary, dim=1), F.log_softmax(altered, dim=1)
    mixture = torch.logaddexp(a, b) - math.log(2)
    disagreement = (a.exp() * (a - mixture) + b.exp() * (b - mixture)).sum(1)
    return (supervised[0] + supervised[1] + disagreement) / 2


def fitting_scene_groups(work, fit, labels, frame_size, seed):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    cached = np.load(work / 't4_views_dinov3_train.npy', mmap_mode='r', allow_pickle=False)
    if cached.ndim != 4 or cached.shape[0] != frame_size:
        raise ValueError('The scene feature cache differs from the labeled image data.')
    features = np.asarray(cached[fit, :, 0], dtype=np.float32)
    features = normalize(features.reshape(len(fit), -1))
    memberships = np.empty(len(fit), dtype=np.int64)
    for label in range(5):
        rows = labels == label
        groups = min(12, int(rows.sum()))
        if groups == 0:
            raise ValueError('Scene grouping requires all five classes in the fitting split.')
        km = MiniBatchKMeans(n_clusters=groups, batch_size=512, n_init=3, random_state=seed + label)
        memberships[rows] = 12 * label + km.fit_predict(features[rows])
    return memberships


def cross_scene_loss(embedding, targets, groups):
    unit = F.normalize(embedding.float(), dim=1)
    similarity = (unit @ unit.T).float() / 0.16
    same_class = targets[:, None] == targets[None, :]
    different_scene = groups[:, None] != groups[None, :]
    positive = same_class & different_scene
    candidates = ~same_class | different_scene
    count = positive.sum(dim=1)
    mean_positive = similarity.masked_fill(~positive, 0).sum(dim=1) / count.clamp_min(1)
    denominator = torch.logsumexp(similarity.masked_fill(~candidates, -1e4), dim=1)
    supported = count > 0
    return (denominator - mean_positive).masked_fill(~supported, 0).sum() / supported.sum().clamp_min(1)


def attach_mix_style(model, seed):
    rng = np.random.default_rng(seed)

    def mix_layer(layer, _inputs, features):
        if not layer.training or len(features) < 2 or rng.random() >= 0.5:
            return features
        x = features.float()
        spread, mean = torch.var_mean(x.detach(), dim=(2, 3), correction=0, keepdim=True)
        scale = (spread + 1e-6).sqrt()
        neighbors = torch.tensor(rng.permutation(len(x)), device=x.device)
        strength = torch.tensor(rng.beta(0.1, 0.1, size=len(x)), device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
        style_mean = strength * mean + (1 - strength) * mean[neighbors]
        style_scale = strength * scale + (1 - strength) * scale[neighbors]
        return ((x - mean) / scale * style_scale + style_mean).to(features.dtype)

    for stage in model.stages[:2]:
        stage.register_forward_hook(mix_layer)


def encoder(args, pretrained=True, warm_start=None):
    name, checkpoint = SCENE_BACKBONES[args.backbone]
    config = dict(num_classes=5, drop_path_rate=0.10)
    convnext = name.startswith('convnext_')
    if not convnext:
        config['img_size'] = SIZE
    if checkpoint is not None and pretrained and warm_start is None:
        config['pretrained_cfg_overlay'] = {'file': str((args.work / checkpoint).resolve())}
    model = timm.create_model(name, pretrained=pretrained and warm_start is None, **config)
    if not pretrained:
        return model
    if warm_start is not None:
        state = torch.load(warm_start, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    if convnext:
        model.stem.requires_grad_(False)
        model.stages[0].requires_grad_(False)
        return model
    model.patch_embed.requires_grad_(False)
    for field in ('pos_embed', 'cls_token', 'reg_token'):
        param = getattr(model, field, None)
        if isinstance(param, torch.nn.Parameter):
            param.requires_grad_(False)
    for block in model.blocks[:args.freeze]:
        block.requires_grad_(False)
    return model


@torch.no_grad()
def initialize_head(model, images, labels, weights, batch, kernel, mean, std, seed):
    from sklearn.linear_model import LogisticRegression

    classifier = model.get_classifier()
    if not isinstance(classifier, torch.nn.Linear) or classifier.bias is None:
        raise ValueError('Head initialization requires a linear classification layer with a bias.')
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == k) for k in range(5)]
    if any(len(group) == 0 for group in groups):
        raise ValueError('Head initialization requires fitting examples from all five classes.')
    sample = np.concatenate([rng.choice(group, min(len(group), 400), replace=False) for group in groups])
    model.eval()
    parts = []
    for start in range(0, len(sample), batch):
        x = denoise(to_device(images[sample[start:start + batch]]), kernel)
        with torch.autocast('mps', dtype=torch.bfloat16):
            representation = model.forward_head(model.forward_features((x - mean) / std), pre_logits=True)
        parts.append(representation.float().cpu().numpy())
    features = np.concatenate(parts)
    divisor = np.sqrt(features.shape[1])
    y = labels[sample]
    weight = np.ones(len(sample)) if weights is None else weights[sample].astype(np.float64)
    weight /= np.bincount(y, weights=weight)[y]
    weight /= weight.mean()
    fitted = LogisticRegression(C=6.0, max_iter=250).fit(features / divisor, y, sample_weight=weight)
    classifier.weight.copy_(torch.from_numpy(fitted.coef_ / divisor).to(classifier.weight))
    classifier.bias.copy_(torch.from_numpy(fitted.intercept_).to(classifier.bias))
    print(f'Initialized the classifier on {len(sample):,} denoised fitting images.', flush=True)


@torch.inference_mode()
def evaluate(model, X, batch, kernel, mean, std):
    model.eval()
    raw, clean = [], []
    for first in range(0, len(X), batch):
        x = to_device(X[first:first + batch])
        with torch.autocast('mps', dtype=torch.bfloat16):
            raw.append(model((x - mean) / std).float().cpu().numpy())
            clean.append(model((denoise(x, kernel) - mean) / std).float().cpu().numpy())
    return np.concatenate(raw), np.concatenate(clean)


def run(args):
    if not torch.backends.mps.is_available():
        raise RuntimeError('An Apple GPU is required for this experiment.')
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)
    data = args.project / 'data_task4'
    frame = pd.read_csv(data / 'train.csv')
    split = pd.read_csv(args.split).set_index('id')
    assert len(split) == len(frame) and split.index.is_unique and set(split.is_val) == {0, 1}
    holdout = split.loc[frame.id, 'is_val'].astype(bool).to_numpy()
    fit, val = np.flatnonzero(~holdout), np.flatnonzero(holdout)
    if args.warm_start:
        previous_indices = args.warm_start.parent / 'val_indices.npy'
        if not np.array_equal(np.load(previous_indices, allow_pickle=False), val):
            raise ValueError('The warm-start checkpoint uses a different held-out split.')
    if args.smoke:
        fit = rng.choice(fit, 256, replace=False)
        val = rng.choice(val, 120, replace=False)
    y = frame.label.to_numpy(int)
    fit_weights = None
    if args.sample_weights:
        source_weights = np.load(args.sample_weights, allow_pickle=False)
        weights = source_weights['weights']
        if not np.array_equal(source_weights['ids'], frame.id.to_numpy()) or not np.array_equal(source_weights['holdout'], holdout):
            raise ValueError('Sample weights do not match the labeled rows and split.')
        if weights.shape != (len(frame),) or not np.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError('Expected one finite positive weight per labeled row.')
        fit_weights = weights[fit]
    X, Z = [read_images(frame.iloc[ind], data / 'train', SIZE) for ind in (fit, val)]
    y_fit, y_val = y[fit], y[val]
    scene_groups = fitting_scene_groups(args.work, fit, y_fit, len(frame), args.seed) if args.contrastive_scenes else None
    model = encoder(args, warm_start=args.warm_start)
    if args.mix_style:
        attach_mix_style(model, args.seed + 487)
    mean = torch.tensor(model.pretrained_cfg['mean'], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(model.pretrained_cfg['std'], device=DEVICE).view(1, 3, 1, 1)
    prior = torch.tensor(np.log(np.bincount(y_fit, minlength=5) / len(y_fit)), device=DEVICE, dtype=torch.float32)
    kernel = smoother()
    if args.initialize_head:
        initialize_head(model, X, y_fit, fit_weights, args.batch_size, kernel, mean, std, args.seed)
    body, head = [], []
    for name, tensor in model.named_parameters():
        if tensor.requires_grad:
            (head if name.startswith('head.') else body).append(tensor)
    rate = args.learning_rate_scale
    opt = torch.optim.AdamW([{'params': body, 'lr': rate * 8e-5}, {'params': head, 'lr': rate * 5e-4}], weight_decay=0.04)
    batches = math.ceil(len(fit) / args.batch_size)
    epochs = 1 if args.smoke else args.epochs
    total = epochs * batches
    warm = max(1, batches // (12 if args.warm_start else 2))
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1, (step + 1) / warm) * (0.10 + 0.90 * (1 + math.cos(math.pi * step / total)) / 2))
    if args.output:
        args.output.mkdir(exist_ok=False, parents=True)
        np.save(args.output / 'val_indices.npy', val)
    best = 0
    if args.warm_start:
        print(f'Warm start from {args.warm_start}, fresh optimizer, learning-rate scale {rate:g}.', flush=True)
    if args.mix_style:
        print('MixStyle augmentation in the first two ConvNeXt stages.', flush=True)
    if args.style_consistency:
        print('Training with a second photometric view and prediction agreement in alternate batches.', flush=True)
    if scene_groups is not None:
        print(f'Using contrastive pairs from {len(np.unique(scene_groups))} fitting scene groups.', flush=True)
    print(f'{args.backbone}: {len(fit):,} training rows, {len(val):,} isolated validation rows', flush=True)
    for epoch in range(epochs):
        start = perf_counter()
        model.train()
        running = []
        for step, ids in enumerate(np.array_split(rng.permutation(len(fit)), batches), 1):
            view = training_view(to_device(X[ids]), kernel)
            x = (view - mean) / std
            yb = torch.tensor(y_fit[ids], device=DEVICE)
            weight = torch.tensor(fit_weights[ids], device=DEVICE) if fit_weights is not None else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast('mps', dtype=torch.bfloat16):
                if args.style_consistency and step % 2 == 1:
                    altered = (style_view(view) - mean) / std
                    logits = model(torch.cat((x, altered))).float() + prior
                    primary, alternate = logits.chunk(2)
                    losses = paired_loss(primary, alternate, yb)
                    loss = (losses * weight).mean() if weight is not None else losses.mean()
                elif rng.random() < 0.16:
                    mix = float(rng.beta(0.4, 0.4))
                    logits = model(mix * x + (1 - mix) * x.roll(1, 0)).float() + prior
                    first = F.cross_entropy(logits, yb, label_smoothing=0.035, reduction='none')
                    second = F.cross_entropy(logits, yb.roll(1, 0), label_smoothing=0.035, reduction='none')
                    if weight is not None:
                        first = first * weight
                        second = second * weight.roll(1, 0)
                    loss = (mix * first + (1 - mix) * second).mean()
                else:
                    if scene_groups is None:
                        logits = model(x).float() + prior
                    else:
                        embedding = model.forward_head(model.forward_features(x), pre_logits=True)
                        logits = model.get_classifier()(embedding).float() + prior
                    losses = F.cross_entropy(logits, yb, label_smoothing=0.035, reduction='none')
                    loss = (losses * weight).mean() if weight is not None else losses.mean()
                    if scene_groups is not None:
                        groups = torch.as_tensor(scene_groups[ids], device=DEVICE)
                        loss = loss + 0.12 * cross_scene_loss(embedding, yb, groups)
            loss.backward()
            opt.step()
            schedule.step()
            running.append(loss.detach())
            if step % 125 == 0:
                print(f'epoch {epoch + 1}, batch {step}/{batches}, loss {float(torch.stack(running).mean().cpu()):.4f}, {perf_counter() - start:.0f}s', flush=True)
        raw, clean = evaluate(model, Z, args.batch_size, kernel, mean, std)
        if not np.isfinite(raw).all() or not np.isfinite(clean).all():
            raise RuntimeError('Non-finite validation outputs.')
        blend = (raw + clean) / 2
        metric = score(blend, y_val)
        print(f'epoch {epoch + 1}/{epochs}, raw macro F1={score(raw, y_val):.5f}, denoised={score(clean, y_val):.5f}, mean={metric:.5f}, {perf_counter() - start:.0f}s', flush=True)
        if args.output:
            np.savez(args.output / f'epoch_{epoch + 1}.npz', raw=raw, clean=clean)
            if metric > best:
                path = args.output / 'best.tmp'
                torch.save({key: value.cpu() for key, value in model.state_dict().items()}, path)
                path.replace(args.output / 'best.pt')
                best = metric


def infer(args):
    if not torch.backends.mps.is_available():
        raise RuntimeError('An Apple GPU is required for this experiment.')
    data = args.project / 'data_task4'
    frame = pd.read_csv(data / 'test.csv')
    sample = pd.read_csv(args.project / 'cpe342-kaggle-main/sample_submission.csv', usecols=['id'])
    if len(frame) != 25889 or not frame.id.is_unique or not frame.id.equals(sample.id):
        raise ValueError('Task 4 test rows do not match the submission order.')
    model = encoder(args, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True), strict=True)
    model.to(DEVICE)
    mean = torch.tensor(model.pretrained_cfg['mean'], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(model.pretrained_cfg['std'], device=DEVICE).view(1, 3, 1, 1)
    start = perf_counter()
    raw, clean = evaluate(model, read_images(frame, data / 'test', SIZE), args.batch_size, smoother(), mean, std)
    if any(z.shape != (len(frame), 5) or not np.isfinite(z).all() for z in (raw, clean)):
        raise RuntimeError('Invalid Task 4 test output.')
    with args.test_output.open('xb') as stream:
        np.savez(stream, raw=raw, clean=clean)
    print(f'Wrote {len(raw):,} raw and denoised predictions in {perf_counter() - start:.0f}s.', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('project', type=Path)
    ap.add_argument('--split', type=Path)
    ap.add_argument('--work', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--backbone', choices=SCENE_BACKBONES, default='dinov3')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=48)
    ap.add_argument('--freeze', type=int, default=3)
    ap.add_argument('--seed', type=int, default=342)
    ap.add_argument('--warm-start', type=Path)
    ap.add_argument('--sample-weights', type=Path)
    ap.add_argument('--mix-style', action='store_true')
    ap.add_argument('--style-consistency', action='store_true')
    ap.add_argument('--contrastive-scenes', action='store_true')
    ap.add_argument('--initialize-head', action='store_true')
    ap.add_argument('--learning-rate-scale', type=float, default=1.0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--weights', type=Path)
    ap.add_argument('--test-output', type=Path)
    args = ap.parse_args()
    if args.learning_rate_scale <= 0:
        ap.error('--learning-rate-scale must be positive')
    if args.mix_style and not SCENE_BACKBONES[args.backbone][0].startswith('convnext_'):
        ap.error('--mix-style currently supports ConvNeXt')
    if args.initialize_head and (args.warm_start or args.weights):
        ap.error('--initialize-head is only for training from a pretrained encoder')
    if bool(args.weights) != bool(args.test_output):
        ap.error('--weights and --test-output must be supplied together')
    if args.weights:
        infer(args)
    else:
        if args.split is None:
            ap.error('--split is required for training')
        run(args)

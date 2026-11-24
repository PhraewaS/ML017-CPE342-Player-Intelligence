"""Task 4 training core, shared by the Modal app (and runnable locally for smoke tests).

Everything takes an explicit `cfg` dict and data root, so a remote worker only needs:
    run_screen(cfg, data_root, groups_csv)          -> scene-held-out score on fold 0
    run_fold(cfg, fold, data_root, groups_csv)      -> OOF + test logits for one fold
The test images are only ever predicted, never trained or tuned on (competition rule).
"""
import math, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F, timm
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BASE = dict(holdout_csv='', model='convnext_tiny.fb_in22k', h=160, w=288, epochs=10, bs=64, lr=2e-4, wd=0.05,
            randaug=False, cutmix=0.0, mixup=0.0, sampler='random', logit_adjust=True,
            label_smoothing=0.1, drop_path=0.1, seed=0, smoke=0)
N_FOLDS = 5
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def device():
    return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


def find_root(start):
    for f in Path(start).rglob('train.csv'):
        if (f.parent / 'train').is_dir() and (f.parent / 'test').is_dir():
            return f.parent
    raise FileNotFoundError(f'task 4 data not found under {start}')


def load_data(data_root, smoke=0):
    root = find_root(data_root)
    tr = pd.read_csv(root / 'train.csv'); te = pd.read_csv(root / 'test.csv')
    tr['path'] = [str(root / 'train' / f) for f in tr.file_name]
    te['path'] = [str(root / 'test' / f) for f in te.file_name]
    if smoke:
        tr = tr.sample(1500, random_state=0).reset_index(drop=True); te = te.iloc[:200].reset_index(drop=True)
    return tr, te


def folds(tr, groups_csv):
    g = pd.read_csv(groups_csv).set_index('id').loc[tr.id, 'g'].values
    return list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=0).split(tr, tr.label, g))


class RandInvert(torch.nn.Module):
    def forward(self, x):
        return 1.0 - x if torch.rand(1) < 0.2 else x


class RandNoise(torch.nn.Module):
    def forward(self, x):
        return (x + torch.randn_like(x) * torch.rand(1) * 0.12).clamp(0, 1) if torch.rand(1) < 0.5 else x


def make_tf(cfg, train):
    t = [T.ToImage()]
    if train and cfg['randaug']:
        t += [T.RandAugment(num_ops=2, magnitude=9)]
    t += [T.ToDtype(torch.float32, scale=True), T.Resize((cfg['h'], cfg['w']), antialias=True)]
    if train:  # corruption-style augmentation matching the dataset (inversion, hue, noise, rotation/shear)
        t += [T.RandomHorizontalFlip(),
              T.RandomApply([T.RandomAffine(degrees=25, translate=(0.06, 0.06), scale=(0.85, 1.15), shear=15)], p=0.6),
              T.RandomApply([T.ColorJitter(0.4, 0.4, 0.6, 0.5)], p=0.8),
              T.RandomGrayscale(0.1), RandInvert(),
              T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2), RandNoise()]
    t += [T.Normalize(MEAN, STD)]
    if train:
        t += [T.RandomErasing(p=0.25, scale=(0.02, 0.15))]
    return T.Compose(t)


class ImgDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.p, self.y, self.tf = list(paths), labels, tf
    def __len__(self):
        return len(self.p)
    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert('RGB')), (int(self.y[i]) if self.y is not None else -1)


def make_loader(df, cfg, train, labels=None, workers=None):
    dev = device()
    workers = (min(15, __import__('os').cpu_count() or 2) if dev == 'cuda' else 0) if workers is None else workers
    ds = ImgDS(df.path.values, labels, make_tf(cfg, train))
    kw = dict(num_workers=workers, pin_memory=dev == 'cuda', persistent_workers=workers > 0)
    if not train:
        return DataLoader(ds, batch_size=cfg['bs'] * 2, shuffle=False, **kw)
    if cfg['sampler'] == 'balanced':
        w = 1.0 / np.bincount(labels, minlength=5)[labels]
        return DataLoader(ds, batch_size=cfg['bs'], sampler=WeightedRandomSampler(w, len(w), replacement=True), drop_last=True, **kw)
    return DataLoader(ds, batch_size=cfg['bs'], shuffle=True, drop_last=True, **kw)


@torch.no_grad()
def predict(model, df, cfg):
    dev = device(); model.eval(); out = []
    for x, _ in make_loader(df, cfg, False):
        x = x.to(dev, non_blocking=True)
        if dev == 'cuda':
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev, dtype=AMP_DTYPE, enabled=dev == 'cuda'):
            lo = model(x).float() + model(torch.flip(x, dims=[3])).float()
        out.append((lo / 2).cpu())
    return torch.cat(out).numpy()


def train(cfg, trn, val=None, tag='', log=print):
    dev = device()
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    y = trn.label.values
    prior = np.bincount(y, minlength=5) / len(y)
    lp = torch.tensor(np.log(prior), dtype=torch.float32, device=dev) if cfg['logit_adjust'] else torch.zeros(5, device=dev)
    model = timm.create_model(cfg['model'], pretrained=True, num_classes=5, drop_path_rate=cfg['drop_path']).to(dev)
    if dev == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        log(f'{tag} gpu={torch.cuda.get_device_name(0)} cpus={__import__("os").cpu_count()} amp={AMP_DTYPE}')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    dl = make_loader(trn, cfg, True, y)
    spe = min(len(dl), cfg['smoke']) if cfg['smoke'] else len(dl)
    total, warm = cfg['epochs'] * spe, max(1, spe // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = torch.amp.GradScaler(enabled=dev == 'cuda' and AMP_DTYPE == torch.float16)
    mixers = ([(cfg['cutmix'], T.CutMix(num_classes=5))] if cfg['cutmix'] > 0 else []) + \
             ([(cfg['mixup'], T.MixUp(num_classes=5))] if cfg['mixup'] > 0 else [])
    hist = []
    for ep in range(cfg['epochs']):
        model.train(); t0 = time.time(); run = 0
        for i, (x, yb) in enumerate(dl):
            if cfg['smoke'] and i >= cfg['smoke']:
                break
            target = F.one_hot(yb, 5).float()
            for p, mixer in mixers:
                if random.random() < p:
                    x, target = mixer(x, yb)
                    break
            x, target = x.to(dev, non_blocking=True), target.to(dev, non_blocking=True)
            if dev == 'cuda':
                x = x.contiguous(memory_format=torch.channels_last)
            target = target * (1 - cfg['label_smoothing']) + cfg['label_smoothing'] / 5
            with torch.autocast(dev, dtype=AMP_DTYPE, enabled=dev == 'cuda'):
                logits = model(x).float() + lp
            loss = torch.sum(-target * F.log_softmax(logits, 1), 1).mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        dt = time.time() - t0
        msg = f'{tag} ep {ep + 1}/{cfg["epochs"]} loss {run / spe:.4f} ({dt:.0f}s, {spe * cfg["bs"] / dt:.0f} img/s)'
        if val is not None and (ep >= cfg['epochs'] - 2 or cfg['smoke']):
            f = f1_score(val.label, predict(model, val, cfg).argmax(1), average='macro'); hist.append(round(f, 5))
            msg += f' | scene-held-out macro F1 {f:.4f}'
        log(msg)
    return model, hist


def full_cfg(overrides):
    cfg = dict(BASE, **overrides)
    if cfg['smoke']:
        cfg.update(epochs=1, bs=8)
    return cfg


def run_screen(overrides, data_root, groups_csv, fold=0):
    """Train on 4 folds, score on the held-out-scene fold. Returns a result dict."""
    cfg = full_cfg(overrides); t0 = time.time()
    tr, _ = load_data(data_root, cfg['smoke'])
    if cfg.get('holdout_csv'):  # custom split, e.g. the most test-like scenes
        is_val = pd.read_csv(cfg['holdout_csv']).set_index('id').loc[tr.id, 'is_val'].values.astype(bool)
        a, b = np.where(~is_val)[0], np.where(is_val)[0]
    else:
        a, b = folds(tr, groups_csv)[fold]
    val = tr.iloc[b].reset_index(drop=True)
    model, hist = train(cfg, tr.iloc[a].reset_index(drop=True), val, tag=str(overrides))
    pred = predict(model, val, cfg).argmax(1)
    return dict(macro_f1=float(f1_score(val.label, pred, average='macro')),
                per_class=np.round(f1_score(val.label, pred, average=None), 4).tolist(),
                last_epochs=hist, minutes=round((time.time() - t0) / 60, 1), cfg=overrides)


def run_fold(overrides, fold, data_root, groups_csv):
    """Train on the other 4 folds; return (valid indices, OOF logits, test logits)."""
    cfg = full_cfg(overrides)
    tr, te = load_data(data_root, cfg['smoke'])
    a, b = folds(tr, groups_csv)[fold]
    model, _ = train(cfg, tr.iloc[a].reset_index(drop=True), tag=f'fold {fold} {overrides}')
    return b, predict(model, tr.iloc[b].reset_index(drop=True), cfg), predict(model, te, cfg)


def make_groups(data_root, out_csv, k=200, smoke=0):
    """Scene groups: KMeans over frozen ImageNet embeddings of the train images."""
    from sklearn.cluster import KMeans
    tr, _ = load_data(data_root, smoke)
    cfg = dict(BASE, bs=128); dev = device()
    m = timm.create_model(BASE['model'], pretrained=True, num_classes=0).to(dev).eval()
    with torch.no_grad():
        E = np.concatenate([m(x.to(dev)).float().cpu().numpy() for x, _ in make_loader(tr, cfg, False)])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    g = KMeans(min(k, len(tr) // 5), n_init=2, random_state=0).fit_predict(E)
    pd.DataFrame({'id': tr.id, 'g': g}).to_csv(out_csv, index=False)
    return int(g.max() + 1)

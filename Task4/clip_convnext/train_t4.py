"""CPE342 Task 4 (game title from screenshots): local GPU trainer. Works on Windows, WSL or Linux.

Same recipe as our Modal runs (pretrained timm backbone + corruption augmentations + hflip TTA). The output that
matters is the TEST LOG-PROBABILITIES. We average them with our existing ensemble on our side.

  python train_t4.py --data DATA_DIR --smoke                        # 2-minute pipeline check
  python train_t4.py --data DATA_DIR --holdout                      # score on the most test-like train scenes
  python train_t4.py --data DATA_DIR --full                         # train on ALL train images -> test predictions
  python train_t4.py --data DATA_DIR --cv                           # 5 scene-grouped folds (slow, optional)

DATA_DIR must contain train.csv, test.csv, train/ and test/ (the contents of task4_images.zip).
Rules: test images are only predicted, never trained or tuned on. Pretrained weights are allowed for Task 4 only.
"""
import argparse, json, math, os, platform, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F, timm
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
PRESETS = {
    # CLIP-pretrained ConvNeXt-Base: expected to be more robust to the unseen test scenes. Recommended run.
    'clip': dict(model='convnext_base.clip_laion2b_augreg_ft_in12k', h=160, w=288, bs=32, lr=8e-5, epochs=12,
                 cutmix=0.6, mixup=0.0, sampler='random', logit_adjust=True, label_smoothing=0.1, drop_path=0.2),
    # Our best Modal config (ConvNeXt-Small, ImageNet-22k). Holdout macro F1 was about 0.990 there.
    'convnext_small': dict(model='convnext_small.fb_in22k', h=160, w=288, bs=48, lr=9.84e-5, epochs=12, cutmix=0.6,
                           mixup=0.0, sampler='random', logit_adjust=True, label_smoothing=0.0, drop_path=0.2),
}
BASE = dict(wd=0.05, randaug=False, seed=0, smoke=0)


def device():
    return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


def amp_dtype():
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def find_root(start):
    start = Path(start)
    for f in [start / 'train.csv'] + list(start.rglob('train.csv')):
        if f.exists() and (f.parent / 'train').is_dir() and (f.parent / 'test').is_dir():
            return f.parent
    raise FileNotFoundError(f'No train.csv with train/ and test/ folders under {start}')


def load_data(data, smoke=0):
    root = find_root(data)
    tr = pd.read_csv(root / 'train.csv'); te = pd.read_csv(root / 'test.csv')
    tr['path'] = [str(root / 'train' / f) for f in tr.file_name]
    te['path'] = [str(root / 'test' / f) for f in te.file_name]
    assert len(te) == 25889, f'expected 25889 test rows, got {len(te)}'
    if smoke:
        tr = tr.sample(1200, random_state=0).reset_index(drop=True); te = te.iloc[:200].reset_index(drop=True)
    return tr, te


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
    t += [T.Normalize(cfg['mean'], cfg['std'])]
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


def make_loader(df, cfg, train, labels=None):
    w = cfg['workers']; dev = device()
    ds = ImgDS(df.path.values, labels, make_tf(cfg, train))
    kw = dict(num_workers=w, pin_memory=dev == 'cuda', persistent_workers=w > 0)
    if not train:
        return DataLoader(ds, batch_size=cfg['bs'] * 2, shuffle=False, **kw)
    if cfg['sampler'] == 'balanced':
        wt = 1.0 / np.bincount(labels, minlength=5)[labels]
        return DataLoader(ds, batch_size=cfg['bs'], sampler=WeightedRandomSampler(wt, len(wt), replacement=True), drop_last=True, **kw)
    return DataLoader(ds, batch_size=cfg['bs'], shuffle=True, drop_last=True, **kw)


@torch.no_grad()
def predict(model, df, cfg):
    """Log-probabilities with horizontal-flip TTA."""
    dev = device(); model.eval(); out = []
    for x, _ in make_loader(df, cfg, False):
        x = x.to(dev, non_blocking=True)
        if dev == 'cuda':
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev, dtype=amp_dtype(), enabled=dev == 'cuda'):
            lo = model(x).float() + model(torch.flip(x, dims=[3])).float()
        out.append(F.log_softmax(lo / 2, 1).cpu())
    return torch.cat(out).numpy()


def train(cfg, trn, val=None, log=print):
    dev = device()
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    y = trn.label.values
    prior = np.bincount(y, minlength=5) / len(y)
    lp = torch.tensor(np.log(prior), dtype=torch.float32, device=dev) if cfg['logit_adjust'] else torch.zeros(5, device=dev)
    model = timm.create_model(cfg['model'], pretrained=True, num_classes=5, drop_path_rate=cfg['drop_path']).to(dev)
    if dev == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        log(f'gpu={torch.cuda.get_device_name(0)} amp={amp_dtype()} workers={cfg["workers"]}')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    dl = make_loader(trn, cfg, True, y)
    spe = min(len(dl), cfg['smoke']) if cfg['smoke'] else len(dl)
    total, warm = cfg['epochs'] * spe, max(1, spe // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = torch.amp.GradScaler(enabled=dev == 'cuda' and amp_dtype() == torch.float16)
    mixers = ([(cfg['cutmix'], T.CutMix(num_classes=5))] if cfg['cutmix'] > 0 else []) + \
             ([(cfg['mixup'], T.MixUp(num_classes=5))] if cfg['mixup'] > 0 else [])
    for ep in range(cfg['epochs']):
        model.train(); t0 = time.time(); run = 0.0
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
            with torch.autocast(dev, dtype=amp_dtype(), enabled=dev == 'cuda'):
                logits = model(x).float() + lp
            loss = torch.sum(-target * F.log_softmax(logits, 1), 1).mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        dt = time.time() - t0
        msg = f'ep {ep + 1}/{cfg["epochs"]} loss {run / spe:.4f} ({dt:.0f}s, {spe * cfg["bs"] / dt:.0f} img/s)'
        if val is not None and (ep >= cfg['epochs'] - 2 or cfg['smoke']):
            msg += f' | held-out macro F1 {f1_score(val.label, predict(model, val, cfg).argmax(1), average="macro"):.4f}'
        log(msg)
    return model


def write_csv(te, logp, path):
    pd.DataFrame({'id': te.id, 'task4': logp.argmax(1).astype(int)}).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='folder with train.csv, test.csv, train/, test/')
    ap.add_argument('--preset', default='clip', choices=list(PRESETS))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true'); mode.add_argument('--holdout', action='store_true')
    mode.add_argument('--full', action='store_true'); mode.add_argument('--cv', action='store_true')
    ap.add_argument('--bs', type=int, help='batch size (lower it if you run out of GPU memory; lr is scaled to match)')
    ap.add_argument('--epochs', type=int); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4 if platform.system() == 'Windows' else 8)
    ap.add_argument('--out', default=str(HERE / 'out'))
    a = ap.parse_args()

    cfg = dict(BASE, **PRESETS[a.preset], seed=a.seed, workers=a.workers)
    if a.bs:
        cfg['lr'] *= a.bs / cfg['bs']; cfg['bs'] = a.bs
    if a.epochs:
        cfg['epochs'] = a.epochs
    if a.smoke:
        cfg.update(smoke=3, epochs=1, bs=8)
    pc = timm.models.get_pretrained_cfg(cfg['model'])
    cfg['mean'], cfg['std'] = tuple(pc.mean), tuple(pc.std)

    mode_name = 'smoke' if a.smoke else 'holdout' if a.holdout else 'full' if a.full else 'cv'
    run = Path(a.out) / f'{a.preset}_{mode_name}_s{a.seed}'
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / 'log.txt', 'a', encoding='utf-8')

    def log(m):
        print(m, flush=True); logf.write(m + '\n'); logf.flush()

    log(f'== {time.ctime()} {mode_name} device={device()} torch={torch.__version__} timm={timm.__version__}\n{json.dumps(cfg)}')
    tr, te = load_data(a.data, cfg['smoke'])
    t0 = time.time(); res = dict(cfg=cfg, mode=mode_name)

    if a.smoke or a.holdout:
        if a.holdout:   # most test-like train scenes, picked by a train-vs-test classifier on image embeddings
            is_val = pd.read_csv(HERE / 'testlike_holdout.csv').set_index('id').loc[tr.id, 'is_val'].values.astype(bool)
        else:
            is_val = np.zeros(len(tr), bool); is_val[:300] = True
        trn, val = tr[~is_val].reset_index(drop=True), tr[is_val].reset_index(drop=True)
        model = train(cfg, trn, val, log)
        p = predict(model, val, cfg).argmax(1)
        res.update(macro_f1=float(f1_score(val.label, p, average='macro')),
                   per_class=np.round(f1_score(val.label, p, average=None, labels=range(5)), 4).tolist())
        if a.smoke:   # also exercise the test path
            write_csv(te, predict(model, te, cfg), run / 'task4_smoke.csv')
    elif a.full:
        model = train(cfg, tr, None, log)
        lp = predict(model, te, cfg)
        np.save(run / 'test_logprob.npy', lp.astype(np.float32)); write_csv(te, lp, run / 'task4.csv')
        res.update(test_mix=np.round(np.bincount(lp.argmax(1), minlength=5) / len(lp), 3).tolist())
    else:
        g = pd.read_csv(HERE / 'scene_groups.csv').set_index('id').loc[tr.id, 'g'].values
        oof = np.zeros((len(tr), 5), np.float32); test = np.zeros((len(te), 5), np.float32)
        for k, (i_tr, i_va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, tr.label, g)):
            f_done = run / f'test_fold{k}.npy'
            if f_done.exists():     # resumable: finished folds are skipped
                oof[i_va] = np.load(run / f'oof_fold{k}.npy'); test += np.load(f_done) / 5; continue
            log(f'-- fold {k}')
            model = train(cfg, tr.iloc[i_tr].reset_index(drop=True), None, log)
            o = predict(model, tr.iloc[i_va].reset_index(drop=True), cfg); t = predict(model, te, cfg)
            np.save(run / f'oof_fold{k}.npy', o); np.save(f_done, t); oof[i_va] = o; test += t / 5
            log(f'fold {k} macro F1 {f1_score(tr.label.values[i_va], o.argmax(1), average="macro"):.4f}')
        np.save(run / 'test_logprob.npy', test); np.save(run / 'oof_logprob.npy', oof); write_csv(te, test, run / 'task4.csv')
        res.update(oof_macro_f1=float(f1_score(tr.label, oof.argmax(1), average='macro')),
                   test_mix=np.round(np.bincount(test.argmax(1), minlength=5) / len(test), 3).tolist())
    res['minutes'] = round((time.time() - t0) / 60, 1)
    (run / 'result.json').write_text(json.dumps(res, indent=1))
    log(json.dumps({k: v for k, v in res.items() if k != 'cfg'}))
    log(f'outputs in {run}')


if __name__ == '__main__':      # required on Windows (DataLoader workers re-import this file)
    torch.multiprocessing.freeze_support()
    main()

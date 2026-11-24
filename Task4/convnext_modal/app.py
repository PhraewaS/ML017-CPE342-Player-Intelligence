"""Task 4 on Modal: automated screening (Optuna x parallel GPUs) + 5-fold final ensemble.

Usage (from the repo root):
    modal run Task4/convnext_modal/app.py --mode screen --trials 16 --parallel 4
    modal run Task4/convnext_modal/app.py --mode leaderboard
    modal run Task4/convnext_modal/app.py --mode final --top 2
    modal run Task4/convnext_modal/app.py --mode final --trials-to-use 3,7
    add --smoke to any command for a tiny, cheap end-to-end check.

Screening is resumable: the Optuna study lives in Task4/convnext_modal/t4_optuna.db, and every finished trial/fold is
cached on the `cpe342-task4-runs` Volume, so re-running never repeats finished work.
Validation always holds out whole scenes (the test set has scenes not seen in train). Test images are only
predicted, never trained or tuned on.
"""
import hashlib, json, time
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ZIP_LOCAL = REPO / 'data_task4' / 'task4_images.zip'
GPU = ['H100', 'A100-80GB', 'L40S']   # tried in order
CPU, MEM = 16.0, 32768   # JPEG decode + augmentation run on CPU; too few cores starve the GPU

app = modal.App('cpe342-task4')
data_vol = modal.Volume.from_name('cpe342-task4-data', create_if_missing=True, version=2)
runs_vol = modal.Volume.from_name('cpe342-task4-runs', create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install('torch==2.8.0', 'torchvision==0.23.0', 'timm==1.0.19', 'numpy==2.2.6',
                    'pandas==2.2.3', 'scikit-learn==1.6.1', 'pillow==11.3.0',
                    'huggingface_hub==0.34.4', 'safetensors==0.6.2')
    .add_local_file(HERE / 't4core.py', '/root/t4core.py')
)
VOLS = {'/data': data_vol, '/runs': runs_vol}


def _key(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def _data():
    """Unzip the image archive from the Volume to fast local disk (once per container)."""
    import zipfile
    dst = Path('/tmp/task4')
    if not (dst / '.done').exists():
        t = time.time()
        with zipfile.ZipFile('/data/task4_images.zip') as z:
            z.extractall(dst)
        (dst / '.done').touch()
        print(f'extracted images in {time.time() - t:.0f}s', flush=True)
    return dst


def _groups_csv(smoke):
    return f'/runs/scene_groups{"_smoke" if smoke else ""}.csv'


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEM, volumes=VOLS, timeout=3600)
def make_groups(smoke: bool = False) -> str:
    import t4core
    p = _groups_csv(smoke)
    runs_vol.reload()
    if not Path(p).exists():
        n = t4core.make_groups(_data(), p, smoke=int(smoke) * 3)
        runs_vol.commit()
        return f'created {p} with {n} scene groups'
    return f'{p} already exists'


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEM, volumes=VOLS, timeout=4 * 3600, retries=1)
def screen_trial(overrides: dict) -> dict:
    import t4core
    out = Path(f'/runs/screen/{_key(overrides)}.json')
    runs_vol.reload()
    if out.exists():
        return json.loads(out.read_text()) | {'cached': True}
    res = t4core.run_screen(overrides, _data(), _groups_csv(overrides.get('smoke', 0)))
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res)); runs_vol.commit()
    return res


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEM, volumes=VOLS, timeout=6 * 3600, retries=1)
def fold_job(overrides: dict, fold: int) -> str:
    import numpy as np, t4core
    d = Path(f'/runs/final/{_key(overrides)}'); f_test = d / f'test_{fold}.npy'
    runs_vol.reload()
    if f_test.exists():
        return f'{d.name} fold {fold}: cached'
    idx, oof, test = t4core.run_fold(overrides, fold, _data(), _groups_csv(overrides.get('smoke', 0)))
    d.mkdir(parents=True, exist_ok=True)
    (d / 'cfg.json').write_text(json.dumps(overrides))
    np.save(d / f'idx_{fold}.npy', idx); np.save(d / f'oof_{fold}.npy', oof); np.save(f_test, test)
    runs_vol.commit()
    return f'{d.name} fold {fold}: done'


@app.function(image=image, volumes=VOLS, timeout=1800)
def ensemble(configs: list) -> dict:
    """Average log-probs over folds and configs; report scene-held-out OOF macro F1."""
    import numpy as np, t4core
    from sklearn.metrics import f1_score
    runs_vol.reload()
    smoke = configs[0].get('smoke', 0)
    tr, te = t4core.load_data(_data(), smoke)
    lsm = lambda z: z - z.max(1, keepdims=True) - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))
    oofs, tests, scores = [], [], {}
    for cfg in configs:
        d = Path(f'/runs/final/{_key(cfg)}')
        oof = np.zeros((len(tr), 5)); test = np.zeros((len(te), 5))
        for k in range(t4core.N_FOLDS):
            oof[np.load(d / f'idx_{k}.npy')] = np.load(d / f'oof_{k}.npy')
            test += np.load(d / f'test_{k}.npy') / t4core.N_FOLDS
        scores[json.dumps(cfg, sort_keys=True)] = float(f1_score(tr.label, oof.argmax(1), average='macro'))
        oofs.append(lsm(oof)); tests.append(lsm(test))
    oof_e, test_e = np.mean(oofs, 0), np.mean(tests, 0)
    pred = test_e.argmax(1)
    return dict(per_config=scores,
                ensemble_oof_macro_f1=float(f1_score(tr.label, oof_e.argmax(1), average='macro')),
                per_class=np.round(f1_score(tr.label, oof_e.argmax(1), average=None), 4).tolist(),
                test_mix=np.round(np.bincount(pred, minlength=5) / len(pred), 3).tolist(),
                csv='id,task4\n' + '\n'.join(f'{i},{p}' for i, p in zip(te.id, pred)) + '\n')


# ---------------------------------------------------------------- local side (runs on your machine)
MODELS = {  # name -> (batch size, base lr)
    'convnext_tiny.fb_in22k': (64, 2e-4),
    'convnext_small.fb_in22k': (48, 1.5e-4),
    'tf_efficientnetv2_s.in21k_ft_in1k': (48, 3e-4),
}
SIZES = {128: 240, 160: 288, 192: 352}   # height -> width (images are 230x120, ~1.9:1)


def suggest(trial):
    """Search space. Edit freely; Optuna (TPE) learns which choices help."""
    model = trial.suggest_categorical('model', list(MODELS))
    h = trial.suggest_categorical('h', list(SIZES))
    bs, lr0 = MODELS[model]
    balance = trial.suggest_categorical('balance', ['logit_adjust', 'balanced_sampler', 'none'])
    return dict(
        model=model, h=h, w=SIZES[h], bs=bs if h < 192 else max(32, bs - 16),
        lr=lr0 * trial.suggest_float('lr_mult', 0.4, 2.5, log=True),
        epochs=trial.suggest_categorical('epochs', [8, 12]),
        cutmix=trial.suggest_categorical('cutmix', [0.0, 0.3, 0.6]),
        mixup=trial.suggest_categorical('mixup', [0.0, 0.2]),
        randaug=trial.suggest_categorical('randaug', [False, True]),
        logit_adjust=balance == 'logit_adjust',
        sampler='balanced' if balance == 'balanced_sampler' else 'random',
        label_smoothing=trial.suggest_categorical('label_smoothing', [0.0, 0.1]),
        drop_path=trial.suggest_categorical('drop_path', [0.1, 0.2]),
    )


BASELINE = dict(model='convnext_tiny.fb_in22k', h=160, balance='logit_adjust', lr_mult=1.0, epochs=12,
                cutmix=0.0, mixup=0.0, randaug=False, label_smoothing=0.1, drop_path=0.1)


def _study(smoke):
    import optuna
    st = optuna.create_study(study_name='t4' + ('_smoke' if smoke else ''), direction='maximize',
                             storage=f'sqlite:///{HERE / "t4_optuna.db"}', load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, constant_liar=True))
    if not st.trials:
        st.enqueue_trial(BASELINE)
    return st


def _leaderboard(st, n=15):
    import optuna
    rows = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    rows.sort(key=lambda t: -t.value)
    print(f'\n{"trial":>5} {"macroF1":>8}  params')
    for t in rows[:n]:
        print(f'{t.number:>5} {t.value:8.4f}  {t.params}  per-class={t.user_attrs.get("per_class")}')
    return rows


def _ensure_uploaded():
    names = {e.path.strip('/') for e in data_vol.listdir('/')}
    if 'task4_images.zip' not in names:
        print(f'uploading {ZIP_LOCAL} ({ZIP_LOCAL.stat().st_size / 1e6:.0f} MB) to Volume cpe342-task4-data ...')
        with data_vol.batch_upload() as b:
            b.put_file(str(ZIP_LOCAL), '/task4_images.zip')


@app.local_entrypoint()
def main(mode: str = 'screen', trials: int = 8, parallel: int = 4, top: int = 2,
         trials_to_use: str = '', smoke: bool = False):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = _study(smoke)
    if mode == 'leaderboard':
        _leaderboard(st); return

    _ensure_uploaded()
    print(make_groups.remote(smoke))
    extra = dict(smoke=3) if smoke else {}

    if mode == 'screen':
        done = 0
        while done < trials:
            batch = [st.ask() for _ in range(min(parallel, trials - done))]
            cfgs = [dict(suggest(t), **extra) for t in batch]
            print(f'\n--- launching trials {[t.number for t in batch]} on {len(batch)} GPUs', flush=True)
            for t, cfg, res in zip(batch, cfgs, screen_trial.map(cfgs, return_exceptions=True)):
                if isinstance(res, Exception):
                    print(f'trial {t.number} FAILED: {res!r}')
                    st.tell(t, state=optuna.trial.TrialState.FAIL)
                    continue
                t.set_user_attr('per_class', res['per_class']); t.set_user_attr('cfg', cfg)
                t.set_user_attr('minutes', res['minutes'])
                st.tell(t, res['macro_f1'])
                print(f'trial {t.number}: macro F1 {res["macro_f1"]:.4f} ({res["minutes"]} min) {t.params}', flush=True)
            done += len(batch)
        _leaderboard(st)

    elif mode == 'final':
        rows = _leaderboard(st)
        chosen = ([st.trials[int(i)] for i in trials_to_use.split(',')] if trials_to_use else rows[:top])
        cfgs = [dict(t.user_attrs['cfg'], **extra) for t in chosen]
        print(f'\nfinal: trials {[t.number for t in chosen]} x 5 folds = {len(cfgs) * 5} GPU jobs', flush=True)
        jobs = [(c, k) for c in cfgs for k in range(5)]
        for r in fold_job.starmap(jobs, return_exceptions=True):
            print(r if not isinstance(r, Exception) else f'FAILED: {r!r}', flush=True)
        res = ensemble.remote(cfgs)
        out = REPO / 'work' / 'out'; out.mkdir(parents=True, exist_ok=True)
        name = 'task4_smoke.csv' if smoke else 'task4.csv'
        (out / name).write_text(res.pop('csv'))
        (out / name.replace('.csv', '_metrics.json')).write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1)); print('wrote', out / name)
    elif mode == 'check':
        # Re-score chosen trials on the most test-like train scenes (picked by a train-vs-test classifier on
        # image embeddings; uses no test labels and never trains on test images).
        local = REPO / 'work' / 'testlike_holdout.csv'
        with runs_vol.batch_upload(force=True) as b:
            b.put_file(str(local), '/testlike_holdout.csv')
        rows = _leaderboard(st, n=0)
        chosen = ([st.trials[int(i)] for i in trials_to_use.split(',')] if trials_to_use else rows[:top])
        cfgs = [dict(t.user_attrs['cfg'], holdout_csv='/runs/testlike_holdout.csv', **extra) for t in chosen]
        print(f'checking trials {[t.number for t in chosen]} on the test-like holdout', flush=True)
        for t, res in zip(chosen, screen_trial.map(cfgs, return_exceptions=True)):
            if isinstance(res, Exception):
                print(f'trial {t.number} FAILED: {res!r}'); continue
            print(f'trial {t.number}: screen {t.value:.4f} -> test-like holdout {res["macro_f1"]:.4f} per-class {res["per_class"]}', flush=True)
            log = HERE / 'testlike_check.csv'
            new = not log.exists()
            with open(log, 'a') as f:
                f.write(('trial,screen_f1,testlike_f1,per_class\n' if new else '') +
                        f'{t.number},{t.value:.5f},{res["macro_f1"]:.5f},"{res["per_class"]}"\n')
    else:
        raise SystemExit(f'unknown mode {mode}')

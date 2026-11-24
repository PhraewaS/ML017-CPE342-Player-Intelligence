"""Train and score the experimental patch classifier on held-out training images."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from Task4.dino.t4_patch_head import group_weights


def score(args):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class PatchPool(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.patch_scores = nn.Sequential(nn.LayerNorm(width), nn.Dropout(0.25), nn.Linear(width, 5))

        def forward(self, tokens):
            scores = self.patch_scores(tokens)
            if self.training:
                keep = torch.rand_like(scores[:, :, :1]) >= 0.2
                keep[:, 0] = True
                n = keep.sum(1)
                scores = scores.masked_fill(~keep, -torch.inf)
            else:
                n = tokens.shape[1]
            return 0.7 * (torch.logsumexp(scores / 0.7, dim=1) - torch.as_tensor(n).log())

    torch.manual_seed(342)
    torch.set_num_threads(2)
    rng = np.random.default_rng(342)
    frame = pd.read_csv(args.project / 'data_task4/train.csv')
    y = torch.tensor(frame.label.to_numpy(), dtype=torch.long)
    holdout = np.load(args.checkpoint.parent / 'val_indices.npy', allow_pickle=False)
    fit = np.setdiff1d(np.arange(len(frame)), holdout)
    cache = np.load(args.cache, mmap_mode='r', allow_pickle=False)
    assert cache.shape == (len(y), 33, 384)
    state = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    native = F.linear(torch.from_numpy(cache[:, 0].astype(np.float32)), state['head.weight'], state['head.bias'])
    truth = y[holdout].numpy()
    print('DINOv3 native holdout macro F1:', f1_score(truth, native[holdout].argmax(1), average='macro'))
    prior = (torch.bincount(y[fit], minlength=5) / len(fit)).log()
    weights = group_weights(cache, fit, y.numpy()) if args.group_balance else None
    for residual in (False, True):
        head = PatchPool(cache.shape[-1])
        opt = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.04)
        start = perf_counter()
        best_f1 = -1.0
        best_state = None
        best_logits = None
        for epoch in range(1, args.epochs + 1):
            head.train()
            for ids in np.array_split(rng.permutation(fit), int(np.ceil(len(fit) / 256))):
                features = torch.from_numpy(cache[ids, 1:].astype(np.float32))
                logits = head(features)
                if residual:
                    logits = native[ids] + 0.6 * logits
                losses = F.cross_entropy(logits + prior, y[ids], label_smoothing=0.04, reduction='none')
                loss = (losses * weights[ids]).mean() if weights is not None else losses.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            head.eval()
            batches = []
            with torch.inference_mode():
                for ids in np.array_split(holdout, int(np.ceil(len(holdout) / 256))):
                    batches.append(head(torch.from_numpy(cache[ids, 1:].astype(np.float32))))
            result = torch.cat(batches)
            combine = native[holdout] + 0.6 * result if residual else 0.7 * native[holdout] + 0.3 * result
            f1 = f1_score(truth, result.argmax(1), average='macro')
            joint = f1_score(truth, combine.argmax(1), average='macro')
            print(f'epoch {epoch} residual={residual}: patch {f1:.5f}, joint {joint:.5f}, {perf_counter() - start:.1f}s', flush=True)
            if joint > best_f1:
                best_f1 = joint
                best_logits = combine.numpy().copy()
                best_state = {name: tensor.detach().clone() for name, tensor in head.state_dict().items()}
        suffix = 'residual' if residual else 'pool'
        if args.group_balance:
            suffix = 'group_balanced_' + suffix
        result_path = args.cache.with_name(f'{args.cache.stem}_{suffix}_val.npz')
        with result_path.open('xb') as output:
            np.savez(output, indices=holdout, logits=best_logits, native=native[holdout].numpy())
        with result_path.with_suffix('.pt').open('xb') as output:
            torch.save(best_state, output)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('project', type=Path)
    ap.add_argument('--cache', required=True, type=Path)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--group-balance', action='store_true')
    score(ap.parse_args())

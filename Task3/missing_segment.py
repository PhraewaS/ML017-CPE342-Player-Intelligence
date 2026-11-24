"""Integrate uncertain player segments when forecasting monthly spend."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from . import spend_model as spending

ROOT = Path(__file__).resolve().parent
v7, v9 = spending.v7, spending.v9


def spending_median(fit, target, query):
    known = fit.dropna(subset=["segment"])
    columns = ["event_participation_rate", "scale", "total_transactions", "vip_status",
               "avg_transaction_value", "friend_count", "achievement_count", "competitive_rank",
               "tournament_participation", "guild_membership", "daily_login_streak", "social_interactions"]
    estimator = HistGradientBoostingClassifier(max_iter=105, max_leaf_nodes=12, random_state=481)
    estimator.fit(known[columns], known.segment)
    chances = estimator.predict_proba(query[columns])
    fit, query = [v7.activity(f) for f in v7.v6.recover(fit, query)]
    model = v9.fit_distribution(fit, target)
    draws = 32
    levels = (np.arange(draws) + .5) / draws
    segments = (levels[None, :, None] > chances.cumsum(axis=1)[:, None, :-1]).sum(axis=2).ravel()
    parts = []
    for seg in range(4):
        expanded = v9.complete(fit, query.assign(segment=seg), draws)
        parts.append(expanded.loc[segments == seg])
    expanded = pd.concat(parts).sort_index()
    return v9.mixture_median(expanded, model, draws)


def run(source, out, base_amount, audit_out=None):
    raw = v7.v6.base.read_aligned(source, "03_regression", "train.csv")
    y = raw.pop("spending_30d").to_numpy()
    frame = v7.v6.base.inputs(raw)
    with np.load(ROOT / "cache/spend_baselines.npz") as cache:
        v7_pred, folds = cache["v7"], cache["fold"]
    old, new = v7_pred.copy(), v7_pred.copy()
    missing = frame.segment.isna().to_numpy() & (frame.scale.fillna(0).to_numpy() > 0)
    for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=42).split(frame)):
        rows = b[missing[b]]
        q = frame.iloc[rows]
        # On this group v9 exactly retained v7. Reconstruct the first v10
        # model only for rows where it replaced part of the v9 prediction.
        current = spending.predict(frame.iloc[a], y[a], q)
        use = ~v9.is_incomplete(q).to_numpy() & np.isfinite(current)
        old[rows] += .5 * np.where(use, current - old[rows], 0)
        new[rows] = spending_median(frame.iloc[a], y[a], q)
        print("Missing-segment fold", k + 1, flush=True)
    dev = (folds < 2) & missing
    weight = min([0, .25, .5, .75, 1], key=lambda w: np.abs(y[dev]-old[dev]-w*(new[dev]-old[dev])).mean())
    candidate = old + weight * (new-old)
    audit = (folds >= 2) & missing
    for k in range(2, 5):
        rows = (folds == k) & missing
        print("Missing-segment fold MAE", k+1, *(np.abs(y[rows]-p[rows]).mean() for p in (old, candidate)))
    gain = np.abs(y[audit]-old[audit]) - np.abs(y[audit]-candidate[audit])
    if audit_out is not None:
        audit_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(audit_out, rows=np.flatnonzero(audit), gain=gain)
    print("Weight, audit players, total absolute-error reduction", weight, sum(audit), gain.sum(), flush=True)
    if gain.sum() <= 0:
        raise SystemExit("No missing-segment improvement.")
    test = v7.v6.base.inputs(v7.v6.base.read_aligned(source, "03_regression", "test.csv"))
    updated = np.load(base_amount, allow_pickle=False).copy()
    rows = test.segment.isna().to_numpy() & (test.scale.fillna(0).to_numpy() > 0)
    updated[rows] += weight * (spending_median(frame, y, test.loc[rows]) - updated[rows])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, updated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=ROOT / "cache/amount.npy")
    parser.add_argument("--out", type=Path, default=ROOT / "trained/segment_amount.npy")
    parser.add_argument("--audit-out",type=Path,help="Save paired changes on the missing-segment subset")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data, args.out, args.base, args.audit_out)

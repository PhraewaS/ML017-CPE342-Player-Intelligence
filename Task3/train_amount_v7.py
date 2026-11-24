"""Estimate conditional spending quantiles with recovered activity bands and censored residuals."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from . import train_amount as v6
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


def activity(frame):
    frame = frame.copy()
    freq, days = frame.purchase_frequency, frame.days_since_last_purchase
    frame["recent"] = frame.recent.mask(days.isna() & (freq >= 15), 0)
    frame["recent"] = frame.recent.mask(days.isna() & freq.between(8, 10, inclusive="left"), 1)
    frame["recent"] = frame.recent.mask(days.isna() & (freq < 5), 2)
    # A bounded log-scale adjustment combines the two measurements of past spend.
    frame["prior_scale"] = frame.scale * np.exp(.35 * frame.previous_relative.fillna(0))
    return frame


def design(frame):
    cat = frame[["segment", "premium", "vip_status", "recent", "freq", "season", "games_played"]].copy()
    cat = cat.fillna(-1).astype(int).astype(str)
    for col in ["vip_status", "premium", "recent", "season", "freq", "games_played"]:
        cat["segment_" + col] = cat.segment + "/" + cat[col]
    cat["vip_premium"] = cat.vip_status + "/" + cat.premium
    cat["frequency_recency"] = cat.freq + "/" + cat.recent
    num = pd.DataFrame(index=frame.index)
    for name, center, unit in [("count", 55, 40), ("event_participation_rate", 50, 30),
                               ("cross_game_activity", 12, 16), ("previous_relative", 0, .12),
                               ("purchase_frequency", 10, 10)]:
        value = ((frame[name] - center) / unit).clip(-3, 3).fillna(0)
        num[name] = value
        for seg in range(4):
            num[f"{name}_{seg}"] = value * (frame.segment == seg)
        for knot in [-.6,.3,1.2]:
            num[f"{name}_bend_{knot}"] = (value-knot).clip(lower=0)
    return cat, num


def residual_quantile(errors, censored, quantiles):
    # Product-limit estimator retains capped observations up to their known
    # lower bounds instead of dropping the highest spenders.
    order = np.argsort(errors)
    z, observed = errors[order], ~censored[order]
    risk = np.arange(len(z), 0, -1)
    cdf = 1 - np.cumprod(1 - observed / risk)
    return np.interp(quantiles, cdf[observed], z[observed])


def estimate(fit, target, query):
    fit, query = (activity(f) for f in v6.recover(fit, query))
    ac, an = design(fit)
    bc, bn = design(query)
    enc = OneHotEncoder(handle_unknown="ignore")
    x = hstack([enc.fit_transform(ac), csr_matrix(an)], format="csr")
    t = hstack([enc.transform(bc), csr_matrix(bn)], format="csr")
    pos = (target > 0) & (fit.prior_scale.to_numpy() > 0)
    z = np.log(target[pos] / fit.prior_scale.to_numpy()[pos])
    cap = target[pos] >= 300000
    model = Ridge(alpha=7, solver="lsqr", tol=1e-6)
    weights = np.ones(len(z))
    for _ in range(9):
        model.fit(x[pos], z, sample_weight=weights)
        errors = z - model.predict(x[pos])
        weights = 1 / (1 + (errors / .29) ** 4)
        weights[cap & (errors < 0)] = .0001
    states, qstates = v6.base.states(fit), v6.base.states(query)
    keys = list(states.columns)
    states["paid"] = target > 0
    full = states.groupby(keys).paid.agg(["count","sum"])
    parent = states.groupby(keys[:-1]).paid.mean()
    rows = full.reindex(pd.MultiIndex.from_frame(qstates[keys])).fillna(0)
    prior = parent.reindex(pd.MultiIndex.from_frame(qstates[keys[:-1]])).fillna(.5)
    rate = (rows["sum"].to_numpy() + 24*prior.to_numpy()) / (rows["count"].to_numpy()+24)
    tau = np.clip(1-.5/np.maximum(rate,.001), .002, .5)
    loc = model.predict(t)
    for (seg,premium,recent),idx in qstates.groupby(["segment","premium","recent"]).indices.items():
        same = fit.segment.to_numpy()[pos] == seg
        local = same & (states.premium.to_numpy()[pos]==premium) & (states.recent.to_numpy()[pos]==recent)
        shift = residual_quantile(errors[same], cap[same], tau[idx])
        if (local & ~cap).sum() > 25:
            w = local.sum() / (local.sum()+220)
            shift += w*(residual_quantile(errors[local], cap[local], tau[idx])-shift)
        loc[idx] += shift
    return np.where(rate>.5, np.exp(loc)*query.prior_scale, 0.).clip(0,300000)


def run(source, out):
    task = "03_regression"
    raw = v6.base.read_aligned(source, task, "train.csv")
    y = raw.pop("spending_30d").to_numpy()
    frame = v6.base.inputs(raw)
    original = np.load(ROOT/"cache/v3-spend-base-oof.npy") @ np.array([.325,.325,.35])
    v3 = v6.base.apply_cutoff(original.clip(0,300000),frame.segment,frame.scale,.06)
    group = frame.segment.fillna(4).to_numpy(int)
    released_weights = [0,1,1,.5,.5]
    older, base, candidate = [np.empty(len(y)) for _ in range(3)]
    folds = np.empty(len(y),int)
    for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=42).split(frame)):
        f,q = frame.iloc[a],frame.iloc[b]
        older[b] = v6.base.blend(v3[b],v6.base.conditional_amount(f,y[a],q),q,[1,1,.25,0])
        raw6 = v6.estimate(f,y[a],q)
        p6 = np.where(q.scale.notna() & np.isfinite(raw6),raw6,older[b])
        base[b] = older[b] + np.take(released_weights,group[b])*(p6-older[b])
        val = estimate(f,y[a],q)
        candidate[b] = np.where(q.scale.notna() & np.isfinite(val),val,base[b])
        folds[b] = k
        print("Completed fold",k+1,flush=True)
    weights=[]
    for k in range(5):
        m = (group==k)&(folds<2)
        def mae(w):
            return np.abs(y[m]-base[m]-w*(candidate[m]-base[m])).mean()
        weights.append(min([0,.25,.5,.75,1],key=mae))
    result=base+np.take(weights,group)*(candidate-base)
    metric=v6.base.previous.metric
    for k in range(5):
        m=folds==k
        print("Fold",k+1,*[metric(y[m],p[m]) for p in (older,base,result)])
    m=folds>=2
    print("Weights",weights,"audit",*[metric(y[m],p[m]) for p in (older,base,result)],flush=True)
    print("MAE",*[np.abs(y[m]-p[m]).mean() for p in (older,base,result)],flush=True)
    actual = y[m]
    err_before = np.abs(actual - base[m])
    err_after = np.abs(actual - result[m])
    rng = np.random.default_rng(712)
    gains = []
    for _ in range(1000):
        sample = rng.integers(len(actual),size=len(actual))
        total = actual[sample].sum()
        gains.append(total/(total+err_after[sample].sum())-total/(total+err_before[sample].sum()))
    print("Paired player bootstrap 95% interval for Task 3 gain",np.quantile(gains,[.025,.975]),flush=True)
    if metric(y[m],result[m]) <= metric(y[m],base[m]):
        raise SystemExit("No validation gain over v6.")
    query=v6.base.inputs(v6.base.read_aligned(source,task,"test.csv"))
    prior=np.load(ROOT / "cache/baseline_v6.npy", allow_pickle=False)
    new=estimate(frame,y,query)
    release=prior+np.take(weights,query.segment.fillna(4).to_numpy(int))*(np.where(query.scale.notna()&np.isfinite(new),new,prior)-prior)
    out.parent.mkdir(exist_ok=True,parents=True)
    np.save(out,release)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=ROOT/"trained/amount.npy")
    args=parser.parse_args()
    run(args.data,args.out)

"""Estimate per-segment probabilities using folded or truncated Gaussian measurements."""
import argparse
import hashlib
import os
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr, softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
from . import train_segments as old

# Lower endpoints of nonnegative measurements. The other continuous variables
# can have boundary atoms fitted via the normal distribution's tail probability.
LOWER = {"play_frequency":1, "avg_session_duration":20, "total_playtime_hours":100,
         "login_streak":0, "total_spending_thb":500, "avg_monthly_spending":100,
         "spending_frequency":1, "friend_count":0, "chat_activity_score":0,
         "friend_invites_sent":0, "gifts_sent_received":0, "tournament_entries":0,
         "rare_items_count":0, "speed_of_progression":0, "peak_concurrent_hours":0,
         "days_since_last_login":0}


def measurements(fit, query):
    cols = list(fit.columns[1:25]) + ["vip_tier","peak_concurrent_hours","loyalty_tier","engagement_level"]
    result = []
    for col in cols:
        s = fit[col]
        if s.dtype.kind not in "fi" or s.nunique() <= 20:
            levels = sorted(s.dropna().unique())
            a,b = [pd.Categorical(frame[col],categories=levels).codes.astype(int)+1 for frame in (fit,query)]
            result.append((a,b,len(levels),None))
            continue
        values = s.dropna().to_numpy()
        unique = np.unique(values)
        if np.all(values == np.floor(values)):
            # Integer counts are rounded down in the continuous model.
            edges = unique[1:]
        else:
            edges = np.unique(np.quantile(values,np.linspace(0,1,201)))[1:-1]
            if np.mean(values == unique[0]) > .002:
                edges = np.r_[np.nextafter(unique[0], np.inf), edges]
            if np.mean(values == unique[-1]) > .002:
                edges = np.r_[edges, unique[-1]]
            edges = np.unique(edges)
        origin = LOWER.get(col, np.median(values))
        unit = np.std(values)
        def codes(v):
            return np.where(v.notna(), 1 + np.searchsorted(edges,v,side="right"), 0)
        result.append((codes(s),codes(query[col]),len(edges)+1,((edges-origin)/unit,col in LOWER)))
    return result


def normal_cdf(edges, mean, log_sd):
    sd = np.exp(log_sd)
    z = (edges-mean)/sd
    density = np.exp(-.5*z*z) / np.sqrt(2*np.pi)
    return ndtr(z), np.stack([-density/sd,-density*z],axis=1)


def interval_probs(edges, parameters, family):
    mean, log_sd = parameters
    cdf, derivative = normal_cdf(edges, mean, log_sd)
    if family == "folded":
        negative, dneg = normal_cdf(-edges, mean, log_sd)
        cdf = cdf - negative
        derivative -= dneg
    if family == "truncated":
        p0, d0 = normal_cdf(np.array([0.]), mean, log_sd)
        den = np.maximum(1-p0,1e-15)
        cdf = (cdf-p0)/den
        derivative = (derivative + (cdf[:,None]-1)*d0)/den
    if family != "normal":
        derivative[edges<=0] = 0
        cdf[edges<=0] = 0
    p = np.diff(np.r_[0.,cdf,1.]).clip(1e-15)
    grad = np.diff(np.vstack([np.zeros(2),derivative,np.zeros(2)]),axis=0)
    return p, grad


def fit_family(edges, count, family, start):
    fraction = count / count.sum()
    def objective(theta):
        p, j = interval_probs(edges, theta, family)
        return -fraction @ np.log(p), -(fraction / p) @ j
    fit = minimize(objective, start, jac=True, method="L-BFGS-B",
                   bounds=[(.015 if family=="folded" else -4,9),(-2.5,2)],
                   options={"ftol":1e-10,"maxiter":60})
    return fit.fun, fit.x


def predict(fit,y,query):
    features = measurements(fit,query)
    targets = np.eye(4)[y]
    membership = targets.copy()
    noise = .976*np.eye(4)+.006
    models = {}
    for step in range(12):
        prior = np.log(membership.mean(0))
        train = np.tile(prior,(len(y),1))
        if step==11:
            out = np.tile(prior,(len(query),1))
        for j,(a,b,n,spec) in enumerate(features):
            counts = np.stack([np.bincount(a,weights=w,minlength=n+1)[1:] for w in membership.T],axis=1)
            dist = (counts+.35)/(counts.sum(0)+.35*n)
            if spec is not None:
                edges, bounded = spec
                for k in range(4):
                    key = (j,k)
                    if key not in models:
                        fits = []
                        for family in (("folded","truncated") if bounded else ("normal",)):
                            loss, theta = fit_family(edges,counts[:,k],family,[1.2 if bounded else 0.,0.])
                            fits.append((loss,family,theta))
                        _,family,theta = min(fits,key=lambda v:v[0])
                    else:
                        family,theta = models[key]
                        _,theta = fit_family(edges,counts[:,k],family,theta)
                    models[key] = family,theta
                    density,_ = interval_probs(edges,theta,family)
                    # Mild shrinkage towards observed frequencies guards
                    # against non-Gaussian tails and numeric boundaries.
                    dist[:,k] = .95*density + .05*dist[:,k]
            logs = np.vstack([np.zeros(4),np.log(dist)])
            train += logs[a]
            if step==11:
                out += logs[b]
        membership = softmax(train+np.log(noise[:,y].T),axis=1)
        confusion = membership.T @ targets+.5
        noise = confusion/confusion.sum(axis=1,keepdims=True)
    return softmax(out,axis=1)@noise


def official_rows(source, name, expected_hash):
    frame = old.read_task(source,"02_multiclass_classification",name)
    digest = hashlib.sha256("\n".join(frame.player_id).encode()).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"Unexpected player order in Task 2 {name}.")
    return frame


def run(source, output, validation_seed=None):
    raw = official_rows(source,"train.csv","74aa154a99834d0b1d0bf9a4671952eeadffc1c2bdffafe337481c048068b7ed")
    labels = raw.pop("segment").to_numpy()
    check_only = validation_seed is not None
    test = raw.iloc[:0] if check_only else official_rows(
        source,"test.csv","1b89cac49cd5c1e5be49e1729472a2eaec9976940492d3fc18c542103b9118da")
    base = np.empty((len(raw),4))
    candidate = np.empty_like(base)
    test_base, test_candidate = [], []
    folds = np.empty(len(raw),int)
    seed = validation_seed if check_only else 42
    for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(raw,labels)):
        fit = raw.iloc[a]
        query = pd.concat([raw.iloc[b],test],ignore_index=True)
        original,_ = old.fit_model(fit.drop(columns="player_id"),labels[a])
        early = .15*original.predict_proba(query.drop(columns="player_id")) + .85*old.new_model(fit,labels[a],query)
        combined = .35*early + .65*old.predict(fit,labels[a],query)
        new = predict(fit,labels[a],query)
        base[b],candidate[b] = combined[:len(b)],new[:len(b)]
        test_base.append(combined[len(b):])
        test_candidate.append(new[len(b):])
        folds[b] = k
        print("Completed segment fold",k+1,flush=True)

    def classifications(w):
        p = (1-w)*base+w*candidate
        result = np.empty(len(raw),int)
        for k in range(5):
            mask = folds == k
            result[mask] = old.label(p[mask],p[~mask])
        return result
    def score(pred,mask):
        return f1_score(labels[mask],pred[mask],average="macro")
    if check_only:
        # Repeat validation with the released weight fixed and new fold assignments.
        original, updated = classifications(0), classifications(1)
        for k in range(5):
            print("Repeat fold",k+1,score(original,folds==k),score(updated,folds==k))
        print("Repeated whole sample",f1_score(labels,original,average="macro"),
              f1_score(labels,updated,average="macro"),flush=True)
        return
    decisions = {w:classifications(w) for w in [0,.25,.5,.75,1.]}
    w = max(decisions,key=lambda w:score(decisions[w],folds<2))
    for k in range(5):
        print(k+1,score(decisions[0],folds==k),score(decisions[w],folds==k))
    audit = folds >= 2
    print("Weight, audit macro F1 before and after",w,score(decisions[0],audit),score(decisions[w],audit),flush=True)
    if score(decisions[w],audit) <= score(decisions[0],audit):
        raise SystemExit("No improvement over v4. No predictions written.")
    prior = np.mean(test_base,axis=0)
    expected = np.load(ROOT / "cache/baseline_v4.npy", allow_pickle=False)
    if not np.array_equal(old.label(prior,base),expected):
        raise ValueError("Reconstructed v4 Task 2 differs from reference.")
    result = old.label((1-w)*prior+w*np.mean(test_candidate,axis=0),(1-w)*base+w*candidate)
    output.parent.mkdir(exist_ok=True,parents=True)
    np.save(output,result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=ROOT/"trained/segments.npy")
    parser.add_argument("--validation-seed",type=int,help="Repeat validation at fixed release weight, without writing predictions")
    args=parser.parse_args()
    run(args.data,args.out,args.validation_seed)

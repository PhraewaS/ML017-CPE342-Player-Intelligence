"""Estimate smooth feature likelihoods and a small train-label confusion matrix."""
import argparse
from pathlib import Path
import os
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
from .parametric_nb import fit_model, macro_f1_rule
from shared.competition_data import read_task


def encode(fit, query):
    features = []
    cols = list(fit.columns[1:25]) + ["vip_tier", "peak_concurrent_hours", "loyalty_tier", "engagement_level"]
    for col in cols:
        v, q = fit[col], query[col]
        if v.nunique() <= 20 or not pd.api.types.is_numeric_dtype(v):
            levels = sorted(v.dropna().unique())
            a, b = (pd.Categorical(s, categories=levels).codes.astype(int) + 1 for s in (v,q))
            features.append((a,b,len(levels)+1,None))
            continue
        lo, hi = v.min(), v.max()
        atoms = [x for x in (lo,hi) if (v==x).mean()>.002]
        core = v[~v.isin(atoms)].dropna()
        edges = np.unique(core.quantile(np.linspace(0,1,66)).values)[1:-1]
        def codes(s):
            code = 1+np.searchsorted(edges,s)
            for index, atom in enumerate(atoms):
                code[s==atom] = len(edges)+2+index
            return np.where(s.notna(),code,0)
        features.append((codes(v),codes(q),len(edges)+2+len(atoms),len(edges)+1))
    return features


def estimate(features, weights):
    prior = weights.mean(axis=0)
    train = np.tile(np.log(prior),(len(weights),1))
    query = np.tile(np.log(prior),(len(features[0][1]),1))
    for a,b,n,continuous in features:
        counts = np.stack([np.bincount(a,weights=w,minlength=n) for w in weights.T],axis=1)
        total = counts[1:].sum(axis=0)
        ll = np.log((counts+.4)/(total+.4*(n-1)))
        if continuous:
            # Smooth log density ratios across neighboring quantile bins; the
            # marginal quantile bin mass cancels in Bayes' rule.
            sl = slice(1,1+continuous)
            pooled = counts[sl].sum(axis=1)
            diff = np.diff(np.eye(continuous),2,axis=0)
            for k in range(4):
                weight = counts[sl,k]+.4
                ratios = ll[sl,k]-np.log(pooled)
                fitted = np.linalg.solve(np.diag(weight)+2800*(diff.T@diff),weight*ratios)
                fitted += np.log(pooled)
                ll[sl,k] = fitted + np.log(np.exp(ll[sl,k]).sum()/np.exp(fitted).sum())
        ll[0] = 0
        train += ll[a]
        query += ll[b]
    return train,query


def new_model(fit,y,query):
    features = encode(fit,query)
    resp = np.eye(4)[y]
    noise = np.full((4,4),.006)
    np.fill_diagonal(noise,.982)
    for _ in range(12):
        ltrain,lquery = estimate(features,resp)
        resp = softmax(ltrain+np.log(noise[:,y].T),axis=1)
        confusion = resp.T@np.eye(4)[y] + .5
        noise = confusion/confusion.sum(axis=1,keepdims=True)
    return softmax(lquery,axis=1)@noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Competition ZIP or extracted data directory")
    parser.add_argument("--out", type=Path, default=ROOT/"trained/refined-segments.npy")
    args = parser.parse_args()
    folder = "02_multiclass_classification"
    df = read_task(args.data, folder, "train.csv")
    y=df.pop("segment").to_numpy()
    test=read_task(args.data, folder, "test.csv")
    result=[]
    test_probs=[]
    for k,(fit,val) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(df,y)):
        q=pd.concat([df.iloc[val],test],ignore_index=True)
        model,_=fit_model(df.iloc[fit].drop(columns="player_id"),y[fit])
        old=model.predict_proba(q.drop(columns="player_id"))
        new=new_model(df.iloc[fit],y[fit],q)
        result.append((val,old[:len(val)],new[:len(val)]))
        test_probs.append((old[len(val):],new[len(val):]))
        print("fold",k,"log loss",log_loss(y[val],old[:len(val)]),log_loss(y[val],new[:len(val)]),flush=True)
    old,new=(np.empty((len(y),4)) for _ in range(2))
    groups=np.zeros(len(y),int)
    for k,(idx,a,b) in enumerate(result):
        groups[idx]=k
        old[idx],new[idx]=a,b
    def decision(p,cal):
        offset,scale=macro_f1_rule(cal)
        return ((p-offset)/scale).argmax(1)
    def scores(weight):
        p=(1-weight)*old+weight*new
        preds=np.zeros(len(y),int)
        for fold in range(5):
            m=groups==fold
            preds[m]=decision(p[m],p[~m])
        return [f1_score(y[m],preds[m],average="macro") for m in [groups<2,groups>=2]]
    weights=[0,.5,.85,1.]
    table=np.array([scores(w) for w in weights])
    best=np.argmax(table[:,0])
    print("selection/audit",list(zip(weights,table.tolist())),flush=True)
    a,b=np.mean(test_probs,axis=0)
    base=decision(a,old)
    print("baseline label mismatches", (base!=np.load(TEAM.parent/"artifacts/task2.npy")).sum())
    if table[best,1]>table[0,1] and best:
        w=weights[best]
        predictions = decision((1-w)*a+w*b,(1-w)*old+w*new)
    else:
        predictions = np.load(TEAM.parent/"artifacts/task2.npy")
        print("Using the team's prior Task 2 predictions after validation.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, predictions)
    print("Saved", args.out)


if __name__=="__main__":
    main()

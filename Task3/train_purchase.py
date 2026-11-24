"""Average spending distributions over plausible values of unobserved purchase attributes."""
import argparse
from pathlib import Path

from .rebuild_baselines import ROOT, v7
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def is_incomplete(f):
    return f[["vip_status","days_since_last_purchase","purchase_frequency","games_played","seasonal_spending_pattern"]].isna().any(axis=1)


def fill_from_donors(fit,expanded,cols,keys,quantile):
    # An observed companion (e.g. frequency band) constrains the donor pool.
    # Sort donors by their values so stratified draws cover the available range.
    missing=expanded[cols].isna().any(axis=1).to_numpy(copy=True)
    if not missing.any():
        return
    donors=fit.dropna(subset=cols).sort_values(cols)
    for width in range(len(keys),0,-1):
        chosen=keys[:width]
        groups=donors.groupby(chosen,dropna=False).indices
        for key,idx in expanded.groupby(chosen,dropna=False).indices.items():
            idx=idx[missing[idx]]
            pool=groups.get(key)
            if not len(idx) or pool is None or len(pool)<8:
                continue
            use=pool[np.minimum((quantile[idx]*len(pool)).astype(int),len(pool)-1)]
            for col in cols:
                m=expanded[col].to_numpy()[idx]
                expanded.iloc[idx,expanded.columns.get_loc(col)]=np.where(np.isfinite(m),m,donors[col].to_numpy()[use])
            missing[idx]=False


def complete(fit,query,draws):
    expanded=query.iloc[np.repeat(np.arange(len(query)),draws)].reset_index(drop=True)
    for f in (fit,expanded):
        f["prem_group"]=f.premium.fillna(-1).astype(int)
        f["freq_mark"]=np.digitize(f.purchase_frequency.fillna(-1),[0,5,8,10,15])
        f["days_mark"]=np.digitize(f.days_since_last_purchase.fillna(-1),[0,15,20,45])
        f["activity_mark"]=np.digitize(f.cross_game_activity.fillna(-1),[0,5,10,15,22,30,43])
    for j,(cols,keys) in enumerate([
        (["vip_status"],["segment","prem_group"]),
        (["days_since_last_purchase"],["segment","freq_mark"]),
        (["purchase_frequency"],["segment","days_mark"]),
        (["games_played"],["segment","activity_mark"]),
        (["seasonal_spending_pattern"],["segment"])]):
        # Independent, reproducible permutations for different missing attributes.
        rng=np.random.default_rng(170+j)
        permutations=np.argsort(rng.random((len(query),draws)),axis=1).ravel()
        u=(permutations+.5)/draws
        fill_from_donors(fit,expanded,cols,keys,u)
        if cols==["days_since_last_purchase"]:
            expanded["days_mark"]=np.digitize(expanded.days_since_last_purchase.fillna(-1),[0,15,20,45])
    expanded["premium"]=expanded.premium.fillna((expanded.vip_status>=3).astype(float))
    expanded["recent"]=np.digitize(expanded.days_since_last_purchase,[15,45])
    expanded["freq"]=np.digitize(expanded.purchase_frequency,[5,8,10,15])
    expanded["season"]=np.select([expanded.seasonal_spending_pattern.isin([0,11]),
                                  expanded.seasonal_spending_pattern.isin([5,6])],[2,1],default=0)
    return expanded


def fit_distribution(fit,y):
    cats,nums=v7.design(fit)
    enc=v7.OneHotEncoder(handle_unknown="ignore")
    x=hstack([enc.fit_transform(cats),csr_matrix(nums)],format="csr")
    pos=(y>0)&(fit.prior_scale.to_numpy()>0)
    z=np.log(y[pos]/fit.prior_scale.to_numpy()[pos])
    cap=y[pos]>=300000
    model=Ridge(alpha=7,solver="lsqr",tol=1e-6)
    weights=np.ones(len(z))
    for _ in range(9):
        model.fit(x[pos],z,sample_weight=weights)
        errors=z-model.predict(x[pos])
        weights=1/(1+(errors/.29)**4)
        weights[cap&(errors<0)]=.0001
    state=v7.v6.base.states(fit)
    keys=list(state.columns)
    state["paid"]=y>0
    full=state.groupby(keys).paid.agg(["count","sum"])
    parent=state.groupby(keys[:-1]).paid.mean()
    grid=np.linspace(.001,.999,251)
    residual={}
    complete_fit=~is_incomplete(fit).to_numpy()[pos]
    for seg in range(4):
        same=state.segment.to_numpy()[pos]==seg
        common=v7.residual_quantile(errors[same],cap[same],grid)
        for prem in (0,1):
            local=same&complete_fit&(state.premium.to_numpy()[pos]==prem)
            ref=common.copy()
            if (local&~cap).sum()>30:
                weight=local.sum()/(local.sum()+150)
                ref+=weight*(v7.residual_quantile(errors[local],cap[local],grid)-common)
            residual[seg,prem]=ref
    return enc,model,full,parent,grid,residual


def mixture_median(expanded,model_info,draws):
    enc,model,full,parent,grid,residual=model_info
    c,d=v7.design(expanded)
    t=hstack([enc.transform(c),csr_matrix(d)],format="csr")
    loc=model.predict(t).reshape(-1,draws)
    state=v7.v6.base.states(expanded)
    rows=full.reindex(pd.MultiIndex.from_frame(state)).fillna(0)
    prior=parent.reindex(pd.MultiIndex.from_frame(state.iloc[:,:-1])).fillna(.5)
    rate=((rows["sum"].to_numpy()+24*prior.to_numpy())/(rows["count"].to_numpy()+24)).reshape(loc.shape)
    groups=state.groupby(["segment","premium"]).indices
    low=loc.min(1)-5
    high=loc.max(1)+5
    for _ in range(19):
        mid=(low+high)/2
        distance=(mid[:,None]-loc).ravel()
        cdf=np.zeros(len(distance))
        for key,idx in groups.items():
            cdf[idx]=np.interp(distance[idx],residual[key],grid,left=0,right=1)
        below=(1-rate+rate*cdf.reshape(loc.shape)).mean(1)<.5
        low=np.where(below,mid,low)
        high=np.where(below,high,mid)
    scale=expanded.prior_scale.to_numpy()[::draws]
    return np.where(rate.mean(1)>.5,np.exp((low+high)/2)*scale,0).clip(0,300000)


def propose(fit,y,query,prior):
    fit,query=[v7.activity(f) for f in v7.v6.recover(fit,query)]
    use=is_incomplete(query).to_numpy() & (query.prior_scale.to_numpy()>0)
    result=prior.copy()
    if use.any():
        model=fit_distribution(fit,y)
        expanded=complete(fit,query.iloc[np.flatnonzero(use)],24)
        result[use]=mixture_median(expanded,model,24)
    return result


def run(data,out):
    raw=v7.v6.base.read_aligned(data,"03_regression","train.csv")
    y=raw.pop("spending_30d").to_numpy()
    frame=v7.v6.base.inputs(raw)
    with np.load(ROOT/"cache/spend_baselines.npz",allow_pickle=False) as f:
        base,fold=f["v7"],f["fold"]
    new=base.copy()
    for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=42).split(frame)):
        if not np.all(fold[b]==k):
            raise ValueError("Unexpected fold assignment.")
        new[b]=propose(frame.iloc[a],y[a],frame.iloc[b],base[b])
        print("Purchase fold",k+1,flush=True)
    dev=fold<2
    group=frame.segment.fillna(4).to_numpy(int)
    weights=[]
    for j in range(5):
        m=dev&(group==j)
        weights.append(min([0,.25,.5,.75,1],key=lambda w:np.abs(y[m]-base[m]-w*(new[m]-base[m])).mean()))
    trial=base+np.take(weights,group)*(new-base)
    metric=v7.v6.base.previous.metric
    for k in range(5):
        m=fold==k
        print(k+1,metric(y[m],base[m]),metric(y[m],trial[m]))
    audit=~dev
    print("Weights, audit scores",weights,metric(y[audit],base[audit]),metric(y[audit],trial[audit]),flush=True)
    print("Audit MAEs",np.abs(y[audit]-base[audit]).mean(),np.abs(y[audit]-trial[audit]).mean())
    truth=y[audit]
    old_error=np.abs(truth-base[audit])
    new_error=np.abs(truth-trial[audit])
    rng=np.random.default_rng(931)
    paired=[]
    for _ in range(1000):
        sample=rng.integers(len(truth),size=len(truth))
        total=truth[sample].sum()
        paired.append(total/(total+new_error[sample].sum())-total/(total+old_error[sample].sum()))
    print("Paired player bootstrap interval for score change",np.quantile(paired,[.025,.975]),flush=True)
    if metric(y[audit],trial[audit])<=metric(y[audit],base[audit]):
        raise SystemExit("No improvement over v8.")
    test=v7.v6.base.inputs(v7.v6.base.read_aligned(data,"03_regression","test.csv"))
    prior=np.load(ROOT / "cache/baseline_v8.npy", allow_pickle=False)
    val=propose(frame,y,test,prior)
    result=prior+np.take(weights,test.segment.fillna(4).to_numpy(int))*(val-prior)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.save(out,result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--out",type=Path,default=ROOT/"trained/amount.npy")
    args=parser.parse_args()
    run(args.data,args.out)

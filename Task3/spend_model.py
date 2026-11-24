"""Estimate spending using shared effects across multiple conditional quantiles."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from . import train_purchase as v9
from .rebuild_baselines import v7
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def design(frame):
    cat,num=v7.design(frame)
    # Model differences in the effect of past spending and transaction counts
    # across VIP tiers. Allow recency and seasonal effects to vary with premium status.
    cat["premium_recency"]=cat.premium+"/"+cat.recent
    cat["premium_season"]=cat.premium+"/"+cat.season
    cat["games_season"]=cat.games_played+"/"+cat.season
    for col in ("previous_relative","count","cross_game_activity"):
        for premium in (0,1):
            num[f"{col}_premium_{premium}"]=num[col]*(frame.premium==premium)
    return cat,num


def common_slopes(x,z,capped,group):
    # Composite quantile IRLS. Five conditional quantiles share feature slopes,
    # with separate intercepts by segment. This uses information from both
    # sides of the spending distribution while limiting the effect of spikes.
    quantiles=np.array([.10,.24,.42,.63,.80])
    model=Ridge(alpha=14,solver="lsqr",tol=1e-6)
    weights=np.ones(len(z))
    for _ in range(6):
        model.fit(x,z,sample_weight=weights)
        residual=z-model.predict(x)
        weights=1/(1+(residual/.29)**4)
        weights[capped&(residual<0)]=.0001
    for _ in range(13):
        residual=z-model.predict(x)
        offsets=np.zeros((len(z),len(quantiles)))
        for seg in np.unique(group):
            rows=group==seg
            offsets[rows]=v7.residual_quantile(residual[rows],capped[rows],quantiles)
        err=residual[:,None]-offsets
        w=np.where(err>=0,quantiles,1-quantiles)/np.hypot(.022,err)
        w[capped[:,None]&(err<0)]=.00001
        total=w.sum(1)
        target=z-(w*offsets).sum(1)/total
        model.fit(x,target,sample_weight=total/len(quantiles))
    return model,z-model.predict(x)


def predict(fit,y,query):
    fit,query=[v7.activity(f) for f in v7.v6.recover(fit,query)]
    a,b=design(fit)
    c,d=design(query)
    enc=v7.OneHotEncoder(handle_unknown="ignore")
    x=hstack([enc.fit_transform(a),csr_matrix(b)],format="csr")
    t=hstack([enc.transform(c),csr_matrix(d)],format="csr")
    pos=(y>0)&(fit.prior_scale.to_numpy()>0)
    z=np.log(y[pos]/fit.prior_scale.to_numpy()[pos])
    cap=y[pos]>=300000
    seg=fit.segment.to_numpy()[pos]
    model,residual=common_slopes(x[pos],z,cap,seg)
    loc=model.predict(t)
    states,qstates=v7.v6.base.states(fit),v7.v6.base.states(query)
    keys=list(states.columns)
    states["paid"]=y>0
    table=states.groupby(keys).paid.agg(["count","sum"])
    parent=states.groupby(keys[:-1]).paid.mean()
    rows=table.reindex(pd.MultiIndex.from_frame(qstates)).fillna(0)
    prior=parent.reindex(pd.MultiIndex.from_frame(qstates[keys[:-1]])).fillna(.5).to_numpy()
    rate=(rows["sum"].to_numpy()+24*prior)/(rows["count"].to_numpy()+24)
    tau=np.clip(1-.5/np.maximum(rate,.001),.002,.50)
    # A large unexpected purchase has a different relative effect on a player
    # with little spending history; estimate a partially pooled residual curve.
    size=fit.prior_scale.to_numpy()[pos]
    qsize=query.prior_scale.to_numpy()
    for g in np.unique(seg):
        same=seg==g
        edges=np.unique(np.quantile(size[same],[.15,.45,.78]))
        fit_size=np.searchsorted(edges,size)
        qbins=np.searchsorted(edges,qsize)
        for k in range(len(edges)+1):
            for prem in [-1,0,1]:
                active=(query.segment.to_numpy()==g)&(qbins==k)&(query.premium.fillna(-1).to_numpy()==prem)
                if not active.any():
                    continue
                local=same&(fit_size==k)&(fit.premium.fillna(-1).to_numpy()[pos]==prem)
                shift=v7.residual_quantile(residual[same],cap[same],tau[active])
                if (local&~cap).sum()>40:
                    frac=local.sum()/(local.sum()+360)
                    shift+=frac*(v7.residual_quantile(residual[local],cap[local],tau[active])-shift)
                loc[active]+=shift
    return np.where(rate>.5,np.exp(loc)*qsize,0.).clip(0,300000)


def run(source,out,audit_out=None):
    raw=v7.v6.base.read_aligned(source,"03_regression","train.csv")
    y=raw.pop("spending_30d").to_numpy()
    frame=v7.v6.base.inputs(raw)
    group=frame.segment.fillna(4).to_numpy(int)
    with np.load(ROOT/"cache/spend_baselines.npz",allow_pickle=False) as cached:
        prior,fold=cached["v7"],cached["fold"]
    baseline,new=np.empty(len(y)),np.empty(len(y))
    for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=42).split(frame)):
        if not np.all(fold[b]==k):
            raise ValueError("Unexpected fold assignment.")
        fit,q=frame.iloc[a],frame.iloc[b]
        v9pred=v9.propose(fit,y[a],q,prior[b])
        baseline[b]=prior[b]+np.take([1,.5,1,.5,0],group[b])*(v9pred-prior[b])
        proposed=predict(fit,y[a],q)
        use=(~v9.is_incomplete(q)).to_numpy() & q.scale.notna().to_numpy() & np.isfinite(proposed)
        new[b]=np.where(use,proposed,baseline[b])
        print("Completed spending fold",k+1,flush=True)
    weights=[]
    for j in range(5):
        m=(fold<2)&(group==j)
        weights.append(min([0,.25,.5,.75,1],key=lambda w:np.abs(y[m]-baseline[m]-w*(new[m]-baseline[m])).mean()))
    prediction=baseline+np.take(weights,group)*(new-baseline)
    metric=v7.v6.base.previous.metric
    for k in range(5):
        m=fold==k
        print(k+1,metric(y[m],baseline[m]),metric(y[m],prediction[m]))
    audit=fold>=2
    print("Weights and audit",weights,metric(y[audit],baseline[audit]),metric(y[audit],prediction[audit]),flush=True)
    print("Audit MAE",np.abs(y[audit]-baseline[audit]).mean(),np.abs(y[audit]-prediction[audit]).mean())
    truth=y[audit]
    before=np.abs(truth-baseline[audit])
    after=np.abs(truth-prediction[audit])
    if audit_out is not None:
        audit_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(audit_out,rows=np.flatnonzero(audit),truth=truth,before=before,after=after)
    rng=np.random.default_rng(3802)
    gains=[]
    for _ in range(1000):
        rows=rng.integers(len(truth),size=len(truth))
        total=truth[rows].sum()
        gains.append(total/(total+after[rows].sum())-total/(total+before[rows].sum()))
    print("Players and paired 95% bootstrap gain interval",len(truth),np.quantile(gains,[.025,.975]),flush=True)
    if metric(y[audit],prediction[audit])<=metric(y[audit],baseline[audit]):
        raise SystemExit("No improvement over v9.")
    query=v7.v6.base.inputs(v7.v6.base.read_aligned(source,"03_regression","test.csv"))
    ref=np.load(ROOT / "cache/baseline_v9.npy", allow_pickle=False)
    candidate=predict(frame,y,query)
    use=(~v9.is_incomplete(query)).to_numpy()&query.scale.notna().to_numpy()&np.isfinite(candidate)
    updated=ref+np.take(weights,query.segment.fillna(4).to_numpy(int))*(np.where(use,candidate,ref)-ref)
    out.parent.mkdir(exist_ok=True,parents=True)
    np.save(out,updated)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=ROOT/"trained/amount.npy")
    parser.add_argument("--audit-out",type=Path,help="Save paired validation errors for further analysis")
    args=parser.parse_args()
    run(args.data,args.out,args.audit_out)

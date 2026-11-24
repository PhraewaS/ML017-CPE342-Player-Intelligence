"""Estimate spending with shared effects and conditional residual distributions."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from . import train_spending as base
import numpy as np
import pandas as pd
from scipy.sparse import hstack,csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


def recover(fit,query):
    fit,query=fit.copy(),query.copy()
    known=fit.segment.notna()
    cols=["event_participation_rate","scale","total_transactions","vip_status",
          "avg_transaction_value","friend_count","achievement_count","competitive_rank",
          "tournament_participation","guild_membership","daily_login_streak","social_interactions"]
    model=HistGradientBoostingClassifier(max_iter=105,max_leaf_nodes=12,random_state=481)
    model.fit(fit.loc[known,cols],fit.loc[known,"segment"])
    for f in (fit,query):
        missing=f.segment.isna()
        if missing.any():
            f.loc[missing,"segment"]=model.predict(f.loc[missing,cols])
        count=f.total_transactions.fillna((f.historical_spending/f.avg_transaction_value.replace(0,np.nan)).round())
        f["count"]=count.fillna((f.purchases_on_discount/f.discount_rate_used.replace(0,np.nan)).round())
    return fit,query


def design(f):
    cat=f[["segment","premium","vip_status","season","recent","freq","games_played"]].fillna(-1).astype(int).astype(str)
    for c in ("vip_status","recent","freq","season","games_played"):
        cat["seg_"+c]=cat.segment+"/"+cat[c]
    cat["vip_premium"]=cat.vip_status+"/"+cat.premium
    cat["freq_recent"]=cat.freq+"/"+cat.recent
    num=pd.DataFrame(index=f.index)
    for c,center,s in [("count",50,40),("event_participation_rate",50,30),
                       ("previous_relative",0,.12),("cross_game_activity",15,20)]:
        x=((f[c]-center)/s).fillna(0).clip(-3,3)
        num[c]=x
        for k in range(4):
            num[f"{c}_{k}"]=x*(f.segment==k)
    return cat,num


def estimate(fit,y,query):
    fit,query=recover(fit,query)
    a,b=design(fit)
    c,d=design(query)
    encoder=OneHotEncoder(handle_unknown="ignore")
    x=hstack([encoder.fit_transform(a),csr_matrix(b)],format="csr")
    t=hstack([encoder.transform(c),csr_matrix(d)],format="csr")
    positive=(y>0)&(fit.scale.to_numpy()>0)
    z=np.log(y[positive]/fit.scale.to_numpy()[positive])
    capped=y[positive]>=300000
    model=Ridge(alpha=10,solver="lsqr",tol=1e-6)
    weights=np.ones(len(z))
    for _ in range(9):
        model.fit(x[positive],z,sample_weight=weights)
        residual=z-model.predict(x[positive])
        weights=np.minimum(1,.18/np.maximum(np.abs(residual),1e-6))
        weights[capped&(residual<0)]=.0001
    loc=model.predict(t)
    train_states=base.states(fit)
    test_states=base.states(query)
    keys=list(train_states.columns)
    train_states["paid"]=y>0
    table=train_states.groupby(keys).paid.agg(["count","sum"])
    parent=train_states.groupby(keys[:-1]).paid.mean()
    pcount=table.reindex(pd.MultiIndex.from_frame(test_states[keys])).fillna(0)
    prior=parent.reindex(pd.MultiIndex.from_frame(test_states[keys[:-1]])).fillna(.5).to_numpy()
    rate=(pcount["sum"].to_numpy()+20*prior)/(pcount["count"].to_numpy()+20)
    # Shrink cohort residual quantiles toward their segment distribution. This
    # accounts for the probability of spending zero when minimizing MAE.
    for key,idx in test_states.groupby(["segment","premium","recent"]).indices.items():
        seg,prem,rec=key
        same=fit.segment.to_numpy()[positive]==seg
        local=same&(train_states.premium.to_numpy()[positive]==prem)&(train_states.recent.to_numpy()[positive]==rec)&~capped
        tau=np.clip(1-.5/np.maximum(rate[idx],1e-10),.002,.5)
        offset=np.quantile(residual[same&~capped],tau)
        if local.sum()>20:
            shrink=local.sum()/(local.sum()+220)
            offset+=shrink*(np.quantile(residual[local],tau)-offset)
        loc[idx]+=offset
    return np.where(rate>.5,np.exp(loc)*query.scale,0.).clip(0,300000)


def run(source,out):
    task="03_regression"
    raw=base.read_aligned(source,task,"train.csv")
    y=raw.pop("spending_30d").to_numpy()
    frame=base.inputs(raw)
    v3=np.load(ROOT/"cache/v3-spend-base-oof.npy")@np.array([.325,.325,.35])
    v3=base.apply_cutoff(v3.clip(0,300000),frame.segment,frame.scale,.06)
    ref,new=np.empty(len(y)),np.empty(len(y))
    folds=np.empty(len(y),int)
    for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=42).split(frame)):
        f,q=frame.iloc[a],frame.iloc[b]
        ref[b]=base.blend(v3[b],base.conditional_amount(f,y[a],q),q,[1,1,.25,0])
        pred=estimate(f,y[a],q)
        new[b]=np.where(q.scale.notna()&np.isfinite(pred),pred,ref[b])
        folds[b]=k
        print("Spending fold",k+1,flush=True)
    def strata(f):
        return f.segment.fillna(4).astype(int).to_numpy()
    group=strata(frame)
    weights=[]
    for k in range(5):
        mask=(group==k)&(folds<2)
        weights.append(min([0,.25,.5,.75,1],key=lambda w:np.abs(y[mask]-ref[mask]-w*(new[mask]-ref[mask])).mean()))
    blended=ref+np.take(weights,group)*(new-ref)
    for k in range(5):
        m=folds==k
        print(k+1,base.previous.metric(y[m],ref[m]),base.previous.metric(y[m],blended[m]))
    m=folds>=2
    before,after=[base.previous.metric(y[m],p[m]) for p in (ref,blended)]
    print("Weights and audit",weights,before,after,flush=True)
    print("Audit size, MAE before and after",m.sum(),
          np.abs(y[m]-ref[m]).mean(),np.abs(y[m]-blended[m]).mean(),flush=True)
    if after<=before:
        raise SystemExit("No improvement.")
    test=base.inputs(base.read_aligned(source,task,"test.csv"))
    old=np.load(ROOT / "cache/baseline_v5.npy", allow_pickle=False)
    predicted=estimate(frame,y,test)
    value=old+np.take(weights,strata(test))*(np.where(test.scale.notna()&np.isfinite(predicted),predicted,old)-old)
    out.parent.mkdir(exist_ok=True,parents=True)
    np.save(out,value)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--out",type=Path,default=ROOT/"trained/amount.npy")
    args=parser.parse_args()
    run(args.data,args.out)

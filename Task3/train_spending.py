"""Learn a spending gate from recent purchases, premium status and past monthly spend."""
import argparse
import os
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
from .spending import apply_cutoff, inputs, previous
from shared.official_data import read_aligned


def states(frame):
    days = frame.days_since_last_purchase
    freq = frame.purchase_frequency
    recency = np.where(days.notna(), (days < 20).astype(int),
                       np.select([freq >= 15, freq < 5], [1, 0], default=2))
    past = frame.prev_month_spending
    # Missing previous spend can still be put into a band if the historical
    # amount is well away from the cutoff.
    above = np.where(past.notna(), (past > 1000).astype(int),
                     np.select([frame.scale > 3400, frame.scale < 2450], [1, 0], default=2))
    return pd.DataFrame({"segment":frame.segment.fillna(-1).astype(int),
                         "premium":frame.premium.fillna(-1).astype(int),
                         "recent":recency, "previous_band":above},index=frame.index)


def amount_inputs(frame):
    group = pd.DataFrame(index=frame.index)
    for key in ["segment","vip_status","premium","season","games_played","recent","freq"]:
        group[key] = frame[key].fillna(-1).astype(int).astype(str)
    numeric = pd.DataFrame(index=frame.index)
    for key,center,unit in [("event_participation_rate",45,30),("purchase_frequency",10,10),
                            ("previous_relative",0,.1),("cross_game_activity",15,20)]:
        value = ((frame[key]-center)/unit).clip(-3,3).fillna(0)
        numeric[key] = value
    return group,numeric


def conditional_amount(fit,target,query):
    fit_cat,fit_num=amount_inputs(fit)
    query_cat,query_num=amount_inputs(query)
    enc=OneHotEncoder(handle_unknown="ignore")
    x=hstack([enc.fit_transform(fit_cat),csr_matrix(fit_num)],format="csr")
    t=hstack([enc.transform(query_cat),csr_matrix(query_num)],format="csr")
    pos=(target>0)&(fit.scale.to_numpy()>0)
    response=np.log(target[pos]/fit.scale.to_numpy()[pos])
    capped=target[pos]>=300000
    model=Ridge(alpha=4,solver="lsqr",tol=1e-6)
    weights=np.ones(len(response))
    for _ in range(7):
        model.fit(x[pos],response,sample_weight=weights)
        error=response-model.predict(x[pos])
        weights=np.minimum(1,.16/np.maximum(np.abs(error),.0001))
        weights[capped&(error<0)]=.0001
    gate_fit=states(fit)
    gate_query=states(query)
    gate_fit["paid"]=target>0
    keys=["segment","premium","recent","previous_band"]
    full=gate_fit.groupby(keys).paid.agg(["mean","count"])
    reduced=gate_fit.groupby(keys[:3]).paid.mean()
    generic=gate_fit.groupby("segment").paid.mean()
    lookup=pd.MultiIndex.from_frame(gate_query[keys])
    lookup_partial=pd.MultiIndex.from_frame(gate_query[keys[:3]])
    prior=reduced.reindex(lookup_partial).to_numpy()
    prior=np.where(np.isfinite(prior),prior,generic.reindex(gate_query.segment).to_numpy())
    counts=full["count"].reindex(lookup).fillna(0).to_numpy()
    rates=full["mean"].reindex(lookup).fillna(0).to_numpy()
    chance=(counts*rates+24*prior)/(counts+24)
    log_ratio=model.predict(t)
    for group in np.unique(gate_query.segment):
        mask=gate_query.segment.to_numpy()==group
        reference=error[(gate_fit.segment.to_numpy()[pos]==group)&~capped]
        if not len(reference):continue
        quantile=np.clip(1-.5/np.maximum(chance[mask],.001),.002,.5)
        log_ratio[mask]+=np.quantile(reference,quantile)
    return np.where(chance>.5,np.exp(log_ratio)*query.scale,0.).clip(0,300000)


def blend(base,candidate,frame,weights):
    # Rows missing the segment or both sources of spending history keep their base estimate.
    out=base.copy()
    seg=frame.segment.to_numpy()
    for group,weight in enumerate(weights):
        valid=(seg==group)&np.isfinite(candidate)&np.isfinite(frame.scale)
        out[valid]=(1-weight)*base[valid]+weight*candidate[valid]
    return out


def run(source,out):
    folder="03_regression"
    raw=read_aligned(source,folder,"train.csv")
    target=raw.pop("spending_30d").to_numpy()
    frame=inputs(raw)
    base=np.load(ROOT/"cache/v3-spend-base-oof.npy",allow_pickle=False)@np.array([.325,.325,.35])
    base=apply_cutoff(np.clip(base,0,300000),frame.segment,frame.scale,.06)
    folds=np.empty(len(base),int)
    forecasts=np.empty(len(base))
    for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=42).split(frame)):
        forecasts[b]=conditional_amount(frame.iloc[a],target[a],frame.iloc[b])
        folds[b]=k
    weights=[]
    for group in range(4):
        sel=(folds<2)&(frame.segment==group)&np.isfinite(forecasts)&frame.scale.notna()
        mae=lambda w:np.abs(target[sel]-((1-w)*base[sel]+w*forecasts[sel])).mean()
        weights.append(min([0,.25,.5,.75,1],key=mae))
    updated=blend(base,forecasts,frame,weights)
    for k in range(5):
        mask=folds==k
        print("fold",k,previous.metric(target[mask],base[mask]),previous.metric(target[mask],updated[mask]))
    audit=folds>=2
    old_score=previous.metric(target[audit],base[audit])
    score=previous.metric(target[audit],updated[audit])
    print("weights",weights,"audit",old_score,score,"average absolute error",
          np.abs(target[audit]-base[audit]).mean(),np.abs(target[audit]-updated[audit]).mean(),flush=True)
    test=inputs(read_aligned(source,folder,"test.csv"))
    old=np.load(ROOT / "cache/baseline_v3.npy", allow_pickle=False)
    if score>old_score:
        prediction=conditional_amount(frame,target,test)
        old=blend(old,prediction,test,weights)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.save(out,old)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--out",type=Path,default=ROOT/"trained/spending.npy")
    args=parser.parse_args()
    run(args.data,args.out)

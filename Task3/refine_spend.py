"""Censor-aware robust spending regression with a separate probability of zero."""
import argparse
import os
os.environ.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack,csr_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parent
from . import train_task3 as previous
from shared.competition_data import read_task


def inputs(raw):
    f=previous.prepare(raw)
    count=f.total_transactions.fillna((f.purchases_on_discount/f.discount_rate_used.replace(0,np.nan)).round())
    h=f.historical_spending.fillna(count.clip(lower=1)*f.avg_transaction_value)
    prev=f.prev_month_spending/.35
    f["scale"]=h.fillna(prev)
    f["previous_relative"]=np.log(prev/h).clip(-.4,.4)
    f["missing_history"]=h.isna().astype(int)
    f["spend_bin"]=np.floor(np.log1p(f.scale)*2)
    return f


def make_design(f):
    cat=pd.DataFrame(index=f.index)
    for key in ["segment","premium","vip_status","freq","season","recent","games_played","spend_bin","missing_history"]:
        cat[key]=f[key].fillna(-1).astype(int).astype(str)
    cat["rec_freq"]=cat.recent+"/"+cat.freq
    cat["seg_vip"]=cat.segment+"/"+cat.vip_status
    cat["seg_recent"]=cat.segment+"/"+cat.recent
    cat["seg_freq"]=cat.segment+"/"+cat.freq
    cat["seg_prem"]=cat.segment+"/"+cat.premium
    for col in ["event_participation_rate","cross_game_activity"]:
        cat[col]=np.floor(f[col].fillna(-10)/10).astype(int).astype(str)
    cont=pd.DataFrame(index=f.index)
    for col,center,scale in [("event_participation_rate",45,25),("cross_game_activity",15,15),
                              ("previous_relative",0,.1),("purchase_frequency",10,8),
                              ("days_since_last_purchase",30,30)]:
        s=((f[col]-center)/scale).fillna(0).clip(-3,3)
        cont[col]=s
        for segment in range(4):
            cont[col+str(segment)]=s*(f.segment==segment)
    return cat,cont


def predict(a,y,q):
    a,q=a.copy(),q.copy()
    cols=a.columns.difference(["segment"])
    known=a.segment.notna()
    imputer=HistGradientBoostingClassifier(max_iter=85,max_leaf_nodes=15,random_state=14)
    imputer.fit(a.loc[known,cols],a.loc[known,"segment"])
    for df in [a,q]:
        m=df.segment.isna()
        if m.any(): df.loc[m,"segment"]=imputer.predict(df.loc[m,cols])
    ca,na=make_design(a)
    cq,nq=make_design(q)
    enc=OneHotEncoder(handle_unknown="ignore")
    x=hstack([enc.fit_transform(ca),csr_matrix(na)],format="csr")
    t=hstack([enc.transform(cq),csr_matrix(nq)],format="csr")
    gate=LogisticRegression(C=.07,max_iter=220)
    gate.fit(x,y>0)
    p=gate.predict_proba(t)[:,1]
    mask=(y>0)&(a.scale.fillna(0).to_numpy()>0)
    xp=x[mask]
    response=np.log(y[mask]/a.scale[mask].to_numpy().clip(.01))
    capped=y[mask]>=299999.99
    weights=np.ones(len(response))
    reg=Ridge(alpha=4,solver="lsqr",tol=1e-6)
    for _ in range(8):
        reg.fit(xp,response,sample_weight=weights)
        res=response-reg.predict(xp)
        weights=np.minimum(1,.12/np.maximum(np.abs(res),.001))
        weights[capped&(res<0)]=.001
    loc=reg.predict(t)
    for k in range(4):
        m=q.segment.to_numpy()==k
        ref=(a.segment.to_numpy()[mask]==k)&~capped
        if ref.any() and m.any():
            loc[m]+=np.quantile(res[ref],np.clip(1-.5/np.maximum(p[m],.001),.005,.5))
    return np.where(p>.5,np.exp(loc)*q.scale,0).clip(0,300000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Competition ZIP or extracted data directory")
    parser.add_argument("--out", type=Path, default=ROOT/"trained/refined-spend.npy")
    args = parser.parse_args()
    folder = "03_regression"
    raw=read_task(args.data, folder, "train.csv")
    y=raw.pop("spending_30d").to_numpy()
    df=inputs(raw)
    first=previous.prepare(raw)
    old=previous.original_estimates("_oof")
    ref=np.empty(len(y))
    prop=np.empty(len(y))
    folds=np.empty(len(y),int)
    for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=42).split(df)):
        ov=previous.fit_predict(first.iloc[a],y[a],first.iloc[b])
        ref[b]=.5*old[b]+.5*np.where(first.iloc[b].base.isna(),old[b],ov)
        est=predict(df.iloc[a],y[a],df.iloc[b])
        prop[b]=np.where(df.iloc[b].scale.isna(),ref[b],est)
        folds[b]=k
        print("spend",k,previous.metric(y[b],ref[b]),previous.metric(y[b],prop[b]),flush=True)
    weights=[0,.35,.65,.85,1.]
    table=[[previous.metric(y[ix],((1-w)*ref+w*prop)[ix]) for ix in [folds<2,folds>=2]] for w in weights]
    print("spend selection/audit",table,flush=True)
    best=np.argmax(np.array(table)[:,0])
    oldtest=np.load(ROOT/"cache/task3-verified.npy")
    predictions = oldtest
    if best and table[best][1]>table[0][1]:
        w=weights[best]
        test=inputs(read_task(args.data, folder, "test.csv"))
        val=predict(df,y,test)
        predictions = (1-w)*oldtest+w*np.where(test.scale.isna(),oldtest,val)
    else:
        print("Using the team's prior Task 3 predictions after validation.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, predictions)
    print("Saved", args.out)


if __name__=="__main__":
    main()

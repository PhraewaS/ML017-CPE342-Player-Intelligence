"""Combine a combat-only class model with the likelihood of observed account attributes."""
import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parent
from . import train_combat as v4
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold


def estimate(fit,y,query):
    cols=v4.gb.COMBAT+["input_consistency_score","movement_pattern_score","buy_decision_score",
                       "weapon_switch_speed","first_blood_rate","damage_per_round"]
    x,t=fit[cols],query[cols]
    settings=dict(max_iter=130,max_leaf_nodes=9,learning_rate=.065,min_samples_leaf=130,
                  l2_regularization=12,early_stopping=False,random_state=821)
    classifier=HistGradientBoostingClassifier(**settings).fit(x,y)
    chance=classifier.predict_proba(t)[:,1]
    pop=v4.gb.population_features(fit)
    pop_test=v4.gb.population_features(query)
    legit=y==0
    prior_probs=pop[legit,:4]
    valid=prior_probs.sum(1)>0
    prior_probs=prior_probs[valid]
    subset=x.iloc[np.flatnonzero(legit)[valid]]
    # Account features narrow membership to known components for training;
    # combat features alone predict membership before observing those features.
    responsibility=prior_probs/prior_probs.sum(1,keepdims=True)
    for step in range(3):
        train_probs=[]
        test_probs=[]
        for weight in responsibility.T:
            model=HistGradientBoostingRegressor(**settings).fit(subset,weight)
            train_probs.append(model.predict(subset).clip(.001,1))
            if step==2:
                test_probs.append(model.predict(t).clip(.001,1))
        joint=prior_probs*np.column_stack(train_probs)
        responsibility=joint/joint.sum(1,keepdims=True)
    legit_prior=np.column_stack(test_probs)
    legit_prior/=legit_prior.sum(1,keepdims=True)
    observed_age=fit.loc[(y==1)&fit[v4.gb.AGE].notna(),v4.gb.AGE]
    young=(observed_age<190).mean()
    cheat_density=(young*pop_test[:,4]+(1-young)*pop_test[:,5])*v4.report_ratio(fit,y,query)
    legit_density=(legit_prior*pop_test[:,:4]).sum(1)
    num=chance*cheat_density
    den=num+(1-chance)*legit_density
    return np.divide(num,den,out=np.full(len(t),np.nan),where=den>0)


def run(data,out):
    folder="01_binary_classification"
    raw=v4.read_aligned(data,folder,"train.csv").dropna(subset=["is_cheater"])
    y=raw.pop("is_cheater").to_numpy(int)
    frame=raw.drop(columns="player_id")
    boost=np.load(ROOT/"cache/task1_oof_prob.npy")
    old,new=np.empty(len(y)),np.empty(len(y))
    folds=np.empty(len(y),int)
    for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(frame,y)):
        fit,query=frame.iloc[a],frame.iloc[b]
        _,old[b]=v4.estimate(fit,y[a],query,boost[b])
        p=estimate(fit,y[a],query)
        new[b]=np.where(np.isfinite(p),p,old[b])
        folds[b]=k
        print("Completed combat fold",k+1,flush=True)
    def metric(w,m):
        return fbeta_score(y[m],old[m]+w*(new[m]-old[m])>v4.gb.THRESHOLD,beta=2)
    w=max([0,.25,.5,.75,1],key=lambda w:metric(w,folds<2))
    for k in range(5):
        print(k+1,metric(0,folds==k),metric(w,folds==k))
    print("Weight and audit",w,metric(0,folds>=2),metric(w,folds>=2),flush=True)
    audit=folds>=2
    previous=old[audit]>v4.gb.THRESHOLD
    updated=old[audit]+w*(new[audit]-old[audit])>v4.gb.THRESHOLD
    # Paired row bootstrap, grouped by the eight possible triples
    # (true class, baseline prediction, revised prediction).
    counts=np.bincount(4*y[audit]+2*previous+updated,minlength=8)
    draws=np.random.default_rng(817).multinomial(counts.sum(),counts/counts.sum(),size=5000)
    truth,prior_flag,new_flag=np.indices((2,2,2)).reshape(3,8).astype(bool)
    def bootstrap_f2(pred):
        return 5*draws[:,truth&pred].sum(axis=1)/(4*draws[:,truth].sum(axis=1)+draws[:,pred].sum(axis=1))
    interval=np.quantile(bootstrap_f2(new_flag)-bootstrap_f2(prior_flag),[.025,.975])
    print("Audit players and paired 95% F2 gain interval",counts.sum(),interval,flush=True)
    if metric(w,folds>=2)<=metric(0,folds>=2):
        raise SystemExit("No improvement over v7.")
    q=v4.read_aligned(data,folder,"test.csv").drop(columns="player_id")
    _,prior=v4.estimate(frame,y,q,np.load(ROOT/"cache/task1_test_prob.npy"))
    if not np.array_equal(prior>v4.gb.THRESHOLD,np.load(ROOT / "cache/baseline_v7.npy", allow_pickle=False)):
        raise ValueError("Baseline mismatch.")
    val=estimate(frame,y,q)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.save(out,(prior+w*(np.where(np.isfinite(val),val,prior)-prior)>v4.gb.THRESHOLD).astype(int))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--out",type=Path,default=ROOT/"trained/accounts.npy")
    args=parser.parse_args()
    run(args.data,args.out)

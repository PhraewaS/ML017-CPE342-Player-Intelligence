"""Estimate evidence for low stability from the four observed login and purchase periods."""
import numpy as np
from sklearn.metrics import fbeta_score


def purchase_likelihood(x):
    noise=np.array([.50,.40,.30,.20])
    seen=np.isfinite(x)
    n=seen.sum(1)
    inside=(~seen|((x>=3)&(x<=6))).all(1)
    low=np.maximum(np.where(seen,x/(1+noise),0).max(1),1.e-12)
    high=np.minimum(np.where(seen,x/(1-noise),np.inf).min(1),5)
    den=np.prod(np.where(seen,2*noise,1),axis=1)*5.
    nominal=np.zeros(len(x))
    nominal[n==0]=1.
    for count in range(1,5):
        ix=(n==count)&(high>low)
        if count==1:
            integral=np.log(high[ix]/low[ix])
        else:
            integral=(low[ix]**(1-count)-high[ix]**(1-count))/(count-1)
        nominal[ix]=integral/den[ix]
    alternative=np.where(inside,3.**(-n),0.)
    return nominal,alternative


def login_likelihood(x,reduced):
    noise=np.array([.2,.3,.25,.15])
    totals=np.zeros(len(x))
    for base in range(20,100):
        lo=base*(1-noise)
        hi=base*(1+noise)
        if reduced:
            lower=np.maximum(.4,x/hi)
            upper=np.minimum(.72,x/lo)
            ratio=np.maximum(upper/np.maximum(lower,1e-20),1)
            density=np.log(ratio)/(.32*(hi-lo))
        else:
            density=((x>=lo)&(x<=hi))/(hi-lo)
        totals+=np.prod(np.where(np.isnan(x),1,density),axis=1)
    return totals/80


def refine(data,baseline,odds_cutoff=None):
    purchases=data[[f"purchase_frequency_{i}" for i in range(1,5)]].to_numpy(float)
    logins=data[[f"login_count_{i}" for i in range(1,5)]].to_numpy(float)
    p0,p1=purchase_likelihood(purchases)
    l0,l1=login_likelihood(logins,False),login_likelihood(logins,True)
    odds=(.01*p1*l1)/(.99*p0*l0+1e-99)
    measured=data.behavior_stability
    unknown=measured.isna().to_numpy()
    if odds_cutoff is None:
        # Evaluate recovery of an observed feature by hiding it. This is a
        # surrogate endpoint; ground-truth anomaly labels are unavailable.
        low=measured.lt(.5).to_numpy()
        eligible=(~baseline|low)&~unknown
        prior=((p0==0)|(l0==0))&(p1>0)
        dev=np.random.default_rng(372).uniform(size=len(data))<.45
        select=dev&eligible
        audit=~dev&eligible
        def score(t,mask):
            return fbeta_score(low[mask],(prior|(odds>t))[mask],beta=3)
        odds_cutoff=max([np.inf,4,2,1,.5,.25],key=lambda t:score(t,select))
        print("Masked-stability F3, original and candidate",score(np.inf,audit),score(odds_cutoff,audit))
        if score(odds_cutoff,audit)<=score(np.inf,audit):
            odds_cutoff=np.inf
    return (baseline|(unknown&(odds>odds_cutoff))).astype(int)

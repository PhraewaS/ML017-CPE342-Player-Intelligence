"""Reconstruct v6 and v7 spending forecasts for the original five training folds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from . import train_amount_v7 as v7
import numpy as np
from sklearn.model_selection import KFold


def out_of_fold(frame,y):
    v6 = v7.v6
    raw = np.load(ROOT/"cache/v3-spend-base-oof.npy") @ np.array([.325,.325,.35])
    prior = v6.base.apply_cutoff(raw.clip(0,300000),frame.segment,frame.scale,.06)
    group = frame.segment.fillna(4).to_numpy(int)
    six, seven = np.empty(len(y)), np.empty(len(y))
    folds = np.empty(len(y),int)
    for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=42).split(frame)):
        fit,query=frame.iloc[a],frame.iloc[b]
        five=v6.base.blend(prior[b],v6.base.conditional_amount(fit,y[a],query),query,[1,1,.25,0])
        subtotal=five
        for estimator,weights,store in [(v6.estimate,[0,1,1,.5,.5],six),(v7.estimate,[0,.5,1,.75,.5],seven)]:
            candidate=estimator(fit,y[a],query)
            delta=np.where(query.scale.notna()&np.isfinite(candidate),candidate,subtotal)-subtotal
            store[b]=subtotal+np.take(weights,group[b])*delta
            subtotal=store[b]
        folds[b]=k
        print("Reconstructed baseline fold",k+1,flush=True)
    return six,seven,folds


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    raw=v7.v6.base.read_aligned(args.data,"03_regression","train.csv")
    y=raw.pop("spending_30d").to_numpy()
    six,seven,folds=out_of_fold(v7.v6.base.inputs(raw),y)
    args.out.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(args.out,v6=six,v7=seven,fold=folds)

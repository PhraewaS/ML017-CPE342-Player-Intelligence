"""Calibrate a zero-spend cutoff on out-of-fold forecasts for the original training rows."""
import argparse
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
from shared.competition_data import read_task
from .refine_spend import inputs,previous,predict,KFold


def apply_cutoff(prices,segment,scale,cutoff):
    eligible=np.isin(segment,[0,1]) & np.isfinite(scale) & (scale>0)
    return np.where(eligible & (prices<cutoff*scale),0.,prices)


def reconstruct(raw,prepared,y,splits):
    first=previous.prepare(raw)
    early=previous.original_estimates("_oof")
    predictions=np.empty((len(y),3))
    for i,(a,b) in enumerate(splits):
        middle=previous.fit_predict(first.iloc[a],y[a],first.iloc[b])
        middle=np.where(first.iloc[b].base.isna(),early[b],middle)
        later=predict(prepared.iloc[a],y[a],prepared.iloc[b])
        later=np.where(prepared.iloc[b].scale.isna(),(early[b]+middle)/2,later)
        predictions[b]=np.column_stack([early[b],middle,later])
        print("Reconstructed spending fold",i,flush=True)
    return predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--retrain-base",action="store_true",help="Recompute the three previous out-of-fold forecasts")
    parser.add_argument("--out",type=Path,default=ROOT/"trained/spending_inputs.npz")
    args=parser.parse_args()
    folder="03_regression"
    raw=read_task(args.data,folder,"train.csv")
    y=raw.pop("spending_30d").to_numpy()
    prepared=inputs(raw)
    splits=list(KFold(5,shuffle=True,random_state=42).split(prepared))
    components=(reconstruct(raw,prepared,y,splits) if args.retrain_base else
                np.load(ROOT/"cache/v3-spend-base-oof.npy",allow_pickle=False))
    base=(components@np.array([.325,.325,.35])).clip(0,300000)
    fold=np.empty(len(y),int)
    for i,(_,b) in enumerate(splits):
        fold[b]=i
    def score(cutoff,mask):
        pred=apply_cutoff(base,prepared.segment,prepared.scale,cutoff)
        return previous.metric(y[mask],pred[mask])
    sel=fold<2
    cutoff=max([0,.01,.03,.06,.10,.14],key=lambda c:score(c,sel))
    print("Cutoff, audit baseline and candidate",cutoff,score(0,~sel),score(cutoff,~sel))
    if score(cutoff,~sel)<=score(0,~sel):
        cutoff=0
    test=inputs(read_task(args.data,folder,"test.csv"))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,segment=test.segment.to_numpy(),scale=test.scale.to_numpy(),cutoff=cutoff)
    print("Wrote",args.out)


if __name__=="__main__":
    main()

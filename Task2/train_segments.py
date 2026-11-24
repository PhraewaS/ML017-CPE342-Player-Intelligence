"""Fit smooth density ratios on the original measurement scale using latent clean classes."""
import os
os.environ.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from .refine_segments import np,pd,softmax,fit_model,new_model,macro_f1_rule,StratifiedKFold,f1_score
from shared.competition_data import read_task
from scipy.interpolate import BSpline

def enc(fit,query):
    cols=list(fit.columns[1:25])+["vip_tier","peak_concurrent_hours","loyalty_tier","engagement_level"]
    features=[]
    for c in cols:
        s=fit[c]
        if s.dtype.kind not in "if" or s.nunique()<=20:
            levels=sorted(s.dropna().unique())
            a,b=(pd.Categorical(f[c],categories=levels).codes.astype(int)+1 for f in (fit,query))
            features.append((a,b,len(levels)+1,None))
            continue
        atoms=[x for x in (s.min(),s.max()) if (s==x).mean()>.002]
        core=s[~s.isin(atoms)].dropna()
        values,counts=np.unique(core,return_counts=True)
        if len(values)<280:
            edges=(values[1:]+values[:-1])/2
        else:
            edges=np.unique(core.quantile(np.linspace(0,1,241)).values)[1:-1]
        def encode(x):
            out=1+np.searchsorted(edges,x)
            for j,v in enumerate(atoms): out[x==v]=len(edges)+2+j
            return np.where(x.notna(),out,0)
        a,b=encode(s),encode(query[c])
        cells=len(edges)+1
        code=encode(core)-1
        n=np.bincount(code,minlength=cells)
        mid=np.bincount(code,weights=core,minlength=cells)/np.maximum(n,1)
        good=n>0
        mid=np.interp(np.arange(cells),np.flatnonzero(good),mid[good])
        knots=np.unique(np.quantile(core,[0,.10,.32,.58,.85,.98,1]))
        knots=np.r_[[knots[0]]*3,knots,[knots[-1]]*3]
        basis=BSpline.design_matrix(mid.clip(knots[0],knots[-1]),knots,3).toarray()
        features.append((a,b,cells+1+len(atoms),basis))
    return features

def predict(fit,y,query):
    feats=enc(fit,query)
    resp=np.eye(4)[y]
    noise=np.full((4,4),.006)
    np.fill_diagonal(noise,.982)
    true=np.eye(4)[y]
    for iteration in range(18):
        train=np.tile(np.log(resp.mean(0)),(len(y),1))
        if iteration==17: result=np.tile(np.log(resp.mean(0)),(len(query),1))
        for a,b,n,B in feats:
            cnt=np.stack([np.bincount(a,weights=v,minlength=n) for v in resp.T],axis=1)
            ll=np.log((cnt+.22)/(cnt[1:].sum(0)+.22*(n-1)))
            if B is not None:
                sl=slice(1,1+len(B))
                pooled=cnt[sl].sum(1)+.88
                for k in range(4):
                    w=cnt[sl,k]+.22
                    # Fit a cubic curve to the log density ratio, then preserve
                    # the probability assigned to the continuous part.
                    v=ll[sl,k]-np.log(pooled)
                    H=B.T@(w[:,None]*B)
                    D=np.diff(np.eye(B.shape[1]),2,axis=0)
                    coef=np.linalg.solve(H+1.5*D.T@D+.0001*np.eye(len(H)),B.T@(w*v))
                    modeled=B@coef+np.log(pooled)
                    ll[sl,k]=modeled+np.log(np.exp(ll[sl,k]).sum()/np.exp(modeled).sum())
            ll[0]=0
            train+=ll[a]
            if iteration==17: result+=ll[b]
        resp=softmax(train+np.log(noise[:,y].T),axis=1)
        confusion=resp.T@true+.5
        noise=confusion/confusion.sum(1,keepdims=True)
    return softmax(result,axis=1)@noise

def label(p,calibration):
    off,scale=macro_f1_rule(calibration)
    return ((p-off)/scale).argmax(1)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path,help="Official competition ZIP or extracted folder")
    parser.add_argument("--out",default=ROOT/"trained/segments.npy",type=Path)
    args=parser.parse_args()
    folder="02_multiclass_classification"
    df=read_task(args.data,folder,"train.csv")
    y=df.pop("segment").to_numpy()
    test=read_task(args.data,folder,"test.csv")
    base,new=np.empty((len(y),4)),np.empty((len(y),4))
    test_base=[];test_new=[];groups=np.empty(len(y),int)
    for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(df,y)):
        fit=df.iloc[a]; q=pd.concat([df.iloc[b],test],ignore_index=True)
        initial,_=fit_model(fit.drop(columns="player_id"),y[a])
        prob=.15*initial.predict_proba(q.drop(columns="player_id"))+.85*new_model(fit,y[a],q)
        pred=predict(fit,y[a],q)
        base[b],new[b]=prob[:len(b)],pred[:len(b)]
        test_base.append(prob[len(b):]);test_new.append(pred[len(b):])
        groups[b]=k
        print("segment fold",k,flush=True)
    def scores(w):
        p=(1-w)*base+w*new
        out=np.empty(len(p),int)
        for k in range(5):
            m=groups==k
            out[m]=label(p[m],p[~m])
        return tuple(f1_score(y[m],out[m],average="macro") for m in (groups<2,groups>=2))
    scores_by_weight={w:scores(w) for w in [0,.35,.65,.85,1.]}
    w=max(scores_by_weight,key=lambda v:scores_by_weight[v][0])
    print("segments",scores_by_weight,"selected",w,flush=True)
    a,b=np.mean(test_base,axis=0),np.mean(test_new,axis=0)
    reference=np.load(ROOT/"cache/refined-segments.npy",allow_pickle=False)
    print("Reconstructed base segment differences",np.sum(label(a,base)!=reference))
    predictions=reference
    if w and scores_by_weight[w][1]>scores_by_weight[0][1]:
        predictions=label((1-w)*a+w*b,(1-w)*base+w*new)
    args.out.parent.mkdir(exist_ok=True,parents=True)
    np.save(args.out,predictions)
    print("Wrote",args.out)

if __name__=="__main__": main()

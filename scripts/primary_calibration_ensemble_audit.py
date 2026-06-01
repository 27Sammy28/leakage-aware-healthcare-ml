#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SEED=42
FEATURES=['age_years','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
TARGET='cardio'

class Prep:
    def fit(self, df):
        x=df[FEATURES].to_numpy(float)
        self.mean=x.mean(axis=0); self.std=x.std(axis=0); self.std[self.std==0]=1
        return self
    def transform(self, df):
        return (df[FEATURES].to_numpy(float)-self.mean)/self.std

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-40,40)))
class LR:
    def __init__(self,c=10): self.c=c
    def fit(self,x,y):
        xb=np.c_[np.ones(len(x)),x]; lam=1/self.c
        def fg(w):
            z=xb@w; p=sigmoid(z)
            f=np.mean(np.logaddexp(0,z)-y*z)+0.5*lam*np.sum(w[1:]**2)/len(y)
            g=xb.T@(p-y)/len(y); g[1:]+=lam*w[1:]/len(y)
            return f,g
        res=minimize(lambda w:fg(w)[0],np.zeros(xb.shape[1]),jac=lambda w:fg(w)[1],method='L-BFGS-B',options={'maxiter':500})
        self.w=res.x; return self
    def predict_proba(self,x): return sigmoid(np.c_[np.ones(len(x)),x]@self.w)
class Tree:
    def __init__(self,max_depth=4,min_leaf=300,feature_subsample=None,seed=0):
        self.max_depth=max_depth; self.min_leaf=min_leaf; self.feature_subsample=feature_subsample; self.rng=np.random.default_rng(seed)
    def fit(self,x,y): self.n_features=x.shape[1]; self.tree=self.build(x,y,0); return self
    def gini(self,y):
        if len(y)==0: return 0
        p=y.mean(); return 2*p*(1-p)
    def build(self,x,y,d):
        if d>=self.max_depth or len(y)<2*self.min_leaf or y.mean() in (0,1): return ('leaf',float(y.mean()))
        cols=np.arange(x.shape[1])
        if self.feature_subsample:
            k=max(1,int(np.ceil(self.feature_subsample*x.shape[1]))); cols=self.rng.choice(cols,k,replace=False)
        parent=self.gini(y); best=None
        for j in cols:
            qs=np.unique(np.quantile(x[:,j],[.1,.2,.3,.4,.5,.6,.7,.8,.9]))
            for t in qs:
                left=x[:,j]<=t; nl=left.sum(); nr=len(y)-nl
                if nl<self.min_leaf or nr<self.min_leaf: continue
                gain=parent-(nl*self.gini(y[left])+nr*self.gini(y[~left]))/len(y)
                if best is None or gain>best[0]: best=(gain,j,float(t),left)
        if best is None or best[0]<=1e-12: return ('leaf',float(y.mean()))
        _,j,t,left=best
        return ('node',j,t,self.build(x[left],y[left],d+1),self.build(x[~left],y[~left],d+1))
    def one(self,row,node):
        if node[0]=='leaf': return node[1]
        _,j,t,l,r=node; return self.one(row,l if row[j]<=t else r)
    def predict_proba(self,x): return np.array([self.one(row,self.tree) for row in x])
class BaggedTrees:
    def __init__(self,n=25,max_depth=4,min_leaf=300,seed=42): self.n=n; self.max_depth=max_depth; self.min_leaf=min_leaf; self.seed=seed
    def fit(self,x,y):
        rng=np.random.default_rng(self.seed); self.trees=[]
        n=len(y)
        for i in range(self.n):
            idx=rng.integers(0,n,n)
            t=Tree(self.max_depth,self.min_leaf,feature_subsample=0.7,seed=self.seed+i).fit(x[idx],y[idx])
            self.trees.append(t)
        return self
    def predict_proba(self,x): return np.mean([t.predict_proba(x) for t in self.trees],axis=0)
class NB:
    def fit(self,x,y):
        self.pr=np.array([(y==0).mean(),(y==1).mean()]); self.mu=np.vstack([x[y==0].mean(0),x[y==1].mean(0)]); self.va=np.vstack([x[y==0].var(0)+1e-6,x[y==1].var(0)+1e-6]); return self
    def predict_proba(self,x):
        logs=[]
        for i in [0,1]: logs.append(np.log(self.pr[i]+1e-12)-.5*np.sum(np.log(2*np.pi*self.va[i])+((x-self.mu[i])**2)/self.va[i],1))
        logs=np.vstack(logs).T; logs-=logs.max(1,keepdims=True); p=np.exp(logs); p/=p.sum(1,keepdims=True); return p[:,1]

def auc(y,s):
    order=np.argsort(s); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(s)+1); pos=y==1; npos=pos.sum(); nneg=len(y)-npos
    return float((ranks[pos].sum()-npos*(npos+1)/2)/(npos*nneg))
def ap(y,s):
    o=np.argsort(-s); yy=y[o]; tp=np.cumsum(yy==1); prec=tp/(np.arange(len(y))+1); return float((prec*(yy==1)).sum()/max(1,(y==1).sum()))
def cal_slope(y,p):
    eps=1e-6; lp=np.log(np.clip(p,eps,1-eps)/(1-np.clip(p,eps,1-eps)))
    X=np.c_[np.ones(len(y)),lp]
    def fg(b):
        z=X@b; pr=sigmoid(z); f=np.mean(np.logaddexp(0,z)-y*z); g=X.T@(pr-y)/len(y); return f,g
    res=minimize(lambda b:fg(b)[0],np.array([0.,1.]),jac=lambda b:fg(b)[1],method='BFGS')
    return float(res.x[0]), float(res.x[1])
def metrics(y,p):
    pred=(p>=.5).astype(int); tn=int(((y==0)&(pred==0)).sum()); fp=int(((y==0)&(pred==1)).sum()); fn=int(((y==1)&(pred==0)).sum()); tp=int(((y==1)&(pred==1)).sum())
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; spec=tn/(tn+fp) if tn+fp else 0; f1=2*prec*rec/(prec+rec) if prec+rec else 0
    intercept,slope=cal_slope(y,p)
    return dict(accuracy=float((pred==y).mean()),precision=prec,recall=rec,f1=f1,balanced_accuracy=(rec+spec)/2,roc_auc=auc(y,p),pr_auc=ap(y,p),brier=float(np.mean((p-y)**2)),calibration_intercept=intercept,calibration_slope=slope,tn=tn,fp=fp,fn=fn,tp=tp)
def split(y):
    rng=np.random.default_rng(SEED); train=[]; test=[]
    for c in [0,1]:
        idx=np.where(y==c)[0]; rng.shuffle(idx); ntest=int(round(.2*len(idx))); test.extend(idx[:ntest]); train.extend(idx[ntest:])
    return np.array(sorted(train)),np.array(sorted(test))
def folds(y,k=5):
    rng=np.random.default_rng(SEED); fs=[[] for _ in range(k)]
    for c in [0,1]:
        idx=np.where(y==c)[0]; rng.shuffle(idx)
        for i,v in enumerate(idx): fs[i%k].append(v)
    return [np.array(sorted(f)) for f in fs]

def main():
    df=pd.read_csv('prism-uploads/cardio_train.csv',sep=';') if Path('prism-uploads/cardio_train.csv').read_text().splitlines()[0].count(';') else pd.read_csv('prism-uploads/cardio_train.csv')
    df['age_years']=df['age']/365.25
    y=df[TARGET].to_numpy(int); tr,te=split(y)
    base_model_factories = {
        'LR': lambda: LR(10),
        'Decision Tree': lambda: Tree(4,300),
        'Bagged Trees': lambda: BaggedTrees(25,4,300),
        'Gaussian NB': lambda: NB(),
    }
    lacve_base = ['LR', 'Decision Tree', 'Bagged Trees', 'Gaussian NB']
    rows=[]; weight_rows=[]
    prep=Prep().fit(df.iloc[tr]); xtr=prep.transform(df.iloc[tr]); xte=prep.transform(df.iloc[te]); ytr=y[tr]; yte=y[te]
    fitted={}; probs={}; metric_cache={}
    for name, factory in base_model_factories.items():
        m=factory().fit(xtr,ytr); fitted[name]=m; p=m.predict_proba(xte); probs[name]=p
        row=metrics(yte,p); metric_cache[name]=row; row.update(model=name,evaluation='held-out',fold=''); rows.append(row)
    raw_weights={}
    for name in lacve_base:
        mt=metric_cache[name]
        raw_weights[name]=max(mt['roc_auc'],1e-6)/(mt['brier']+abs(mt['calibration_slope']-1.0)+1e-6)
    total=sum(raw_weights.values()); weights={k:v/total for k,v in raw_weights.items()}
    p_lacve=sum(weights[name]*probs[name] for name in lacve_base)
    row=metrics(yte,p_lacve); row.update(model='LACVE',evaluation='held-out',fold=''); rows.append(row)
    for name in lacve_base:
        weight_rows.append({'evaluation':'held-out','fold':'','model':name,'raw_weight':raw_weights[name],'normalized_weight':weights[name]})
    for fid,test in enumerate(folds(y,5),1):
        train=np.setdiff1d(np.arange(len(y)),test); prep=Prep().fit(df.iloc[train]); xtr=prep.transform(df.iloc[train]); xte=prep.transform(df.iloc[test]); ytr=y[train]; yte=y[test]
        fold_probs={}; fold_metrics={}
        for name in ['LR','Decision Tree','Bagged Trees','Gaussian NB']:
            m=base_model_factories[name]().fit(xtr,ytr); p=m.predict_proba(xte); fold_probs[name]=p
            row=metrics(yte,p); fold_metrics[name]=row; row.update(model=name,evaluation='5-fold CV',fold=fid); rows.append(row)
        raw_weights={}
        for name in lacve_base:
            mt=fold_metrics[name]
            raw_weights[name]=max(mt['roc_auc'],1e-6)/(mt['brier']+abs(mt['calibration_slope']-1.0)+1e-6)
        total=sum(raw_weights.values()); weights={k:v/total for k,v in raw_weights.items()}
        p_lacve=sum(weights[name]*fold_probs[name] for name in lacve_base)
        row=metrics(yte,p_lacve); row.update(model='LACVE',evaluation='5-fold CV',fold=fid); rows.append(row)
        for name in lacve_base:
            weight_rows.append({'evaluation':'5-fold CV','fold':fid,'model':name,'raw_weight':raw_weights[name],'normalized_weight':weights[name]})
    out=Path('primary_audit_artifacts'); out.mkdir(exist_ok=True)
    res=pd.DataFrame(rows); res.to_csv(out/'calibration_and_ensemble_results.csv',index=False)
    pd.DataFrame(weight_rows).to_csv(out/'lacve_weights.csv',index=False)
    summary=res.groupby(['evaluation','model'])[['accuracy','f1','balanced_accuracy','roc_auc','pr_auc','brier','calibration_intercept','calibration_slope']].agg(['mean','std']).reset_index()
    summary.columns=['_'.join([str(x) for x in c if x]) for c in summary.columns]
    summary.to_csv(out/'calibration_and_ensemble_summary.csv',index=False)
    (out/'calibration_and_ensemble_metadata.json').write_text(json.dumps({'seed':SEED,'bagged_trees':'25 bootstrap shallow trees, max_depth=4, min_leaf=300, 70% feature subsampling','lacve':'Leakage-Aware Calibrated Voting Ensemble; normalized weight = AUC / (Brier + abs(calibration_slope - 1) + epsilon)','calibration_slope':'logistic recalibration model y ~ intercept + slope*logit(p)'},indent=2))
    print(summary.to_string(index=False))
if __name__=='__main__': main()

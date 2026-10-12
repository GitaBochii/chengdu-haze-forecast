"""Five training-quantile knots with monotone interval-distance LAD fitting."""

import numpy as np
from scipy.optimize import linprog


def design(x, knots):
    basis=np.eye(len(knots))
    return np.column_stack([np.interp(x,knots,basis[:,j]) for j in range(len(knots))])


def predict_map(model, values):
    x=np.asarray(values,dtype=float)
    k=np.asarray(model['knots'],dtype=float)
    y=np.asarray(model['values'],dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all() or (x<=0).any():
        raise ValueError('Expected positive finite predictions')
    if k.ndim!=1 or not len(k) or y.shape!=k.shape or not np.isfinite(k).all() or not np.isfinite(y).all():
        raise ValueError('Invalid mapping')
    if (np.diff(k)<=0).any() or (np.diff(y)<-1e-8).any() or (y<1-1e-8).any():
        raise ValueError('Mapping must be ordered and positive')
    offset=x-np.clip(x,k[0],k[-1])
    return np.maximum(1.,design(x,k)@y+offset)


def fit_map(values, lower, upper, weights):
    x,lo,hi,w=[np.asarray(v,dtype=float) for v in (values,lower,upper,weights)]
    if x.ndim!=1 or not len(x) or any(v.shape!=x.shape for v in (lo,hi,w)):
        raise ValueError('Require aligned nonempty vectors')
    if not all(np.isfinite(v).all() for v in (x,lo,hi,w)) or (x<=0).any() or (lo<=0).any() or (hi<lo).any() or (w<0).any() or w.sum()<=0:
        raise ValueError('Require finite positive predictions, valid intervals and weights')
    knots=np.unique(np.quantile(x,[0.,.25,.5,.75,1.]))
    a=design(x,knots); n=len(x); m=len(knots)
    # One nonnegative slack per interval: max(lower-prediction, prediction-upper, 0).
    monotone=np.zeros((max(0,m-1),m+n))
    for j in range(m-1):
        monotone[j,j]=1.; monotone[j,j+1]=-1.
    constraints=np.vstack([np.column_stack([a,-np.eye(n)]),
                           np.column_stack([-a,-np.eye(n)]),monotone])
    result=linprog(np.r_[np.zeros(m),w/w.sum()],A_ub=constraints,
                   b_ub=np.r_[hi,-lo,np.zeros(m-1)],bounds=[(1,None)]*m+[(0,None)]*n,method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    model={'knots':knots.tolist(),'values':result.x[:m].tolist()}
    p=predict_map(model,x)
    distance=np.maximum.reduce([lo-p,p-hi,np.zeros(n)])
    loss=float(np.average(distance,weights=w))
    np.testing.assert_allclose(loss,result.fun,atol=1e-7)
    model['training_distance']=loss
    return model

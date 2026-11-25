import sys, os; import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','src','mopabe'))
from mop import *
for k in (5,7):
    P=choose_params(k); sc=MOP(P,n_attr=3,seed=21+k); sc.setup(); z,zn,w=sc.user_keygen()
    X=[];R2=[]
    for _ in range(20):
        tk=sc.tk_gen([0,1,2],w); X.append(tk['x']); R2.append(tk['r2'])
    X=np.array(X,dtype=float);R2=np.array(R2,dtype=float)
    print(k,'s',round(P.s),'std r1',round(X.std()),'ratio',round(X.std()/P.s,3),'std r2',round(R2.std()),'ratio',round(R2.std()/P.s,3),'mean r1/s',round(X.mean()/P.s,4),'mean r2/s',round(R2.mean()/P.s,4),flush=True)

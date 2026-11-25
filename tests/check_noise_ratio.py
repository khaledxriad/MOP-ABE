import sys, os; import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','src','mopabe'))
from mop import *
P=choose_params(7); sc=MOP(P,n_attr=4,seed=12); sc.setup()
z,zn,w=sc.user_keygen(); tk=sc.tk_gen([0,1,2,3],w)
rng=np.random.default_rng(6); errs=[]; bad=0
for t in range(12):
    mu=rng.integers(0,2,size=P.d); ct=sc.encrypt([0,1,2,3],mu); o=sc.transform(ct,tk); g,ph=sc.user_decrypt(o,zn)
    q=P.q; errs.append(((ph-mu*(q//2)+q//2)%q-q//2)/q); bad+=int((g!=mu).sum())
e=np.concatenate(errs); print('std',e.std(),'pred',P.sigma_total/P.q,'ratio',e.std()/(P.sigma_total/P.q),'bad',bad,'of',12*P.d)

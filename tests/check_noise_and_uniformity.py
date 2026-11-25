import sys, os; import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','src','mopabe'))
from mop import *
import time
for k in (7,):
    P=choose_params(k); sc=MOP(P,n_attr=4,seed=11)
    t=time.perf_counter(); sc.setup(); print('k',k,'setup',time.perf_counter()-t,flush=True)
    z,zn,w=sc.user_keygen(); tk=sc.tk_gen([0,1,2,3],w)
    rng=np.random.default_rng(5); mu=rng.integers(0,2,size=P.d)
    ct=sc.encrypt([0,1,2,3],mu); o=sc.transform(ct,tk); g,ph=sc.user_decrypt(o,zn)
    q=P.q; err=((ph-mu*(q//2)+q//2)%q-q//2)/q
    print('ok',(g==mu).all(),'std/q',err.std(),'pred',P.sigma_total/q)
    # check B_i = A R - G structure is uniform-looking: mean/var of coefficients
    b=sc.ring.intt(sc.B_n[0]); print('B mean/q',b.mean()/q,'std/q',b.std()/q,'(uniform 0.5, 0.2887)')

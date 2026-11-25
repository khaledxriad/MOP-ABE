import sys, os; import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','src','mopabe'))
from mop import *
import time
k=int(sys.argv[1]); ell=int(sys.argv[2])
P = choose_params(k)
print(P.describe())
sc = MOP(P, n_attr=ell, seed=3)
t=time.time(); sc.setup(); print('setup', time.time()-t)
z, zn, w_n = sc.user_keygen()
t=time.time(); tk = sc.tk_gen([0,1,2], w_n); print('tkgen', time.time()-t)
rng=np.random.default_rng(5); mu = rng.integers(0,2,size=256)
t=time.time(); ct = sc.encrypt(list(range(ell)), mu); print('enc', time.time()-t)
t=time.time(); out = sc.transform(ct, tk); print('transform', time.time()-t)
t=time.time(); got, ph = sc.user_decrypt(out, zn); print('user', time.time()-t)
print('ok', (got==mu).all())
q=P.q; ideal=mu*(q//2); err=(ph-ideal+q//2)%q-q//2
print('noise std (packed)', err.std()/q, 'max', abs(err).max()/q)
out2=sc.transform(ct,tk,False); g2,ph2=sc.user_decrypt(out2,zn); err2=(ph2-ideal+q//2)%q-q//2
print('noise std (raw) log2', math.log2(err2.std()), 'pred', math.log2(P.noise_std_pred), 'ratio to q', err2.std()/q, (g2==mu).all())

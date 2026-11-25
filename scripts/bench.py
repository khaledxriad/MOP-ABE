import sys, os, json, csv, time, math
import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','src','mopabe'))
import numpy as np
from mop import *
from estimator import bits as lwe_bits

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data')
def wcsv(name, rows, fields):
    with open(f'{OUT}/{name}','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)

def phase_err(P, ph, mu):
    q=P.q; ideal=mu*(q//2); return ((ph-ideal+q//2)%q - q//2)

def elem_bytes(P): return P.d*math.ceil(math.log2(P.q))/8.0

def direct_decrypt(sc, ct, tk_direct, P):
    """baseline: mobile device decrypts the (unpacked) ABE ciphertext itself with the long key (w=0)."""
    ring=sc.ring
    t=time.perf_counter()
    y = ring.dot(ct['c0'], tk_direct['x_n'])
    cP = ct['ci'][tk_direct['Pset'][0]]
    for i in tk_direct['Pset'][1:]: cP = ring.add(cP, ct['ci'][i])
    y = ring.add(y, ring.dot(cP, tk_direct['r2_n']))
    ph = ring.intt(ring.sub(ct['cp'], y))
    bits_ = (((4*ph + ring.q)//(2*ring.q))%2).astype(np.int64)
    return time.perf_counter()-t, bits_

stage = sys.argv[1]
rng0 = np.random.default_rng(2026)

if stage=='params':
    rows=[]; store={}
    for k in (5,6,7,8):
        P=choose_params(k); D=P.describe()
        sc=MOP(P,n_attr=4,seed=100+k); sc.setup()
        z,zn,w=sc.user_keygen(); tk=sc.tk_gen([0,1,2,3],w)      # |P| = 4 = w_max (worst case)
        T=12; raw=[]; pk=[]; ber=0; ncoef=0; maxp=0; uv=[]
        for t in range(T):
            mu=rng0.integers(0,2,size=P.d)
            ct=sc.encrypt([0,1,2,3],mu)
            o_raw=sc.transform(ct,tk,compress_out=False); g,ph=sc.user_decrypt(o_raw,zn); raw.append(phase_err(P,ph,mu)/P.q)
            o=sc.transform(ct,tk); g2,ph2=sc.user_decrypt(o,zn); e=phase_err(P,ph2,mu)/P.q; pk.append(e)
            ber+=int((g2!=mu).sum()); ncoef+=P.d
            if k==7: uv.append((o_raw['cE'].copy(),o_raw['v'].copy(),zn.copy(),mu.copy()))
        raw=np.concatenate(raw); pk=np.concatenate(pk)
        sec=[lwe_bits(k*P.d,P.q,sg)[1] for sg in (3.2,0.8165)]
        rows.append(dict(k=k,n=k*P.d,d=P.d,log2q=round(D['log2q'],2),l=P.l,m=P.m,mp=P.mp,s=round(P.s),sigma_G=round(float(P.gad.sigma),2),
            log2_noise_pred=round(D['log2_noise_std'],2),log2_noise_meas=round(math.log2(raw.std()*P.q),2),
            std_total_pred=round(D['sigma_total_over_q'],4),std_total_meas=round(float(pk.std()),4),max_meas=round(float(np.abs(pk).max()),4),
            margin_pred=round(D['margin'],2),margin_meas=round(0.25/float(pk.std()),2),bit_errors=ber,bits_tested=ncoef,
            sec_cls_gauss=round(sec[0]),sec_cls_tern=round(sec[1]),sec_min=round(min(sec))))
        print(rows[-1],flush=True)
        if k==7:
            import pickle; pickle.dump((P.q,P.d,uv),open(f'{OUT}/k7_outputs.pkl','wb'))
    wcsv('params.csv',rows,list(rows[0].keys()))

elif stage=='scale':      # sizes and timings vs |X| for k=7 and vs k at |X|=4
    rows=[]
    for k,Xs in [(7,[1,2,4,8,16]),(5,[4]),(6,[4]),(8,[4])]:
        P=choose_params(k); eb=elem_bytes(P)
        sc=MOP(P,n_attr=max(Xs),seed=200+k); ts=sc.setup()
        t=time.perf_counter(); z,zn,w=sc.user_keygen(); tuk=time.perf_counter()-t
        Pset=[0,1,2]
        ttk=[]; 
        for _ in range(3):
            t=time.perf_counter(); tk=sc.tk_gen(Pset,w); ttk.append(time.perf_counter()-t)
        tkd=sc.tk_gen(Pset,np.zeros_like(w))
        for nx in Xs:
            X=list(range(nx)) if nx>=3 else list(range(3))[:nx]
            Xe=list(range(max(nx,3)))
            mu=rng0.integers(0,2,size=P.d)
            t=time.perf_counter(); ct=sc.encrypt(Xe,mu); tenc=time.perf_counter()-t
            tt=[];tu=[];td=[]
            for _ in range(20):
                t=time.perf_counter(); o=sc.transform(ct,tk); tt.append(time.perf_counter()-t)
                t=time.perf_counter(); g,ph=sc.user_decrypt(o,zn); tu.append(time.perf_counter()-t)
                dt,gb=direct_decrypt(sc,ct,tkd,P); td.append(dt)
            assert (g==mu).all() and (gb==mu).all()
            ct_bytes=(P.m+len(Xe)*P.mp+1+P.k)*eb
            direct_dl=(P.m+len(Pset)*P.mp+1)*eb                    # components the device needs
            packed=(P.k*P.d*P.du+P.d*P.dv)/8.0
            unpacked_out=(P.k+1)*eb
            tk_bytes=(P.m+P.mp)*P.d*2.0                           # 16-bit signed coefficients (|r| < 2^15)
            maxr=max(abs(tk['x']).max(),abs(tk['r2']).max())
            rows.append(dict(k=k,n=k*P.d,nX=len(Xe),P=len(Pset),setup_s=round(ts,3),userkeygen_ms=round(tuk*1e3,3),
                tkgen_ms=round(1e3*float(np.median(ttk)),2),enc_s=round(tenc,3),transform_ms=round(1e3*float(np.median(tt)),3),
                user_ms=round(1e3*float(np.median(tu)),4),direct_ms=round(1e3*float(np.median(td)),3),
                ct_bytes=round(ct_bytes),direct_dl_bytes=round(direct_dl),packed_bytes=round(packed),unpacked_out_bytes=round(unpacked_out),
                user_key_bytes=P.k*P.d*2/8.0,tk_bytes=round(tk_bytes),direct_key_bytes=round(tk_bytes),user_pk_bytes=round(P.k*eb),
                max_tk_coef=int(maxr)))
            print(rows[-1],flush=True)
    wcsv('scale.csv',rows,list(rows[0].keys()))

elif stage=='psweep':     # |P| sweep at k=7
    P=choose_params(7); sc=MOP(P,n_attr=8,seed=300); sc.setup(); z,zn,w=sc.user_keygen()
    rows=[]; mu=rng0.integers(0,2,size=P.d)
    ct=sc.encrypt(list(range(8)),mu)
    for w_ in (1,2,3,4):
        Ps=list(range(w_)); errs=[]; ttk=[]; tt=[]
        for rep in range(8):
            t=time.perf_counter(); tk=sc.tk_gen(Ps,w); ttk.append(time.perf_counter()-t)
            t=time.perf_counter(); o=sc.transform(ct,tk,False); tt.append(time.perf_counter()-t)
            g,ph=sc.user_decrypt(o,zn); errs.append(phase_err(P,ph,mu)/P.q)
        e=np.concatenate(errs)
        # predicted raw std with this w
        var=SIG_E**2*(P.m*P.d*P.s**2)+SIG_E**2*(P.sig_R**2*w_)*(P.m*P.d)*(P.mp*P.d)*P.s**2
        rows.append(dict(w=w_,tkgen_ms=round(1e3*float(np.median(ttk)),2),transform_ms=round(1e3*float(np.median(tt)),3),
            std_meas=round(float(e.std()),5),std_pred=round(math.sqrt(var)/P.q,5),max_meas=round(float(np.abs(e).max()),5)))
        print(rows[-1],flush=True)
    wcsv('psweep.csv',rows,list(rows[0].keys()))

elif stage=='negative':   # unauthorized attempts
    P=choose_params(6); sc=MOP(P,n_attr=6,seed=400); sc.setup(); ring=sc.ring
    z,zn,w=sc.user_keygen(); z2,zn2,w2=sc.user_keygen()
    rows=[]
    def ber(g,mu): return float((g!=mu).mean())
    N=6
    res={'authorized':[], 'wrong_user_key':[], 'missing_attribute_forged':[], 'collusion_mixed_keys':[], 'unblinded_cloud_view':[]}
    tkA=sc.tk_gen([0,1],w); tkB=sc.tk_gen([2,3],w)
    tkC=sc.tk_gen([0,5],w)     # clause needs attribute 5
    for _ in range(N):
        mu=rng0.integers(0,2,size=P.d)
        ct=sc.encrypt([0,1,2,4],mu)                       # attribute 3 and 5 absent
        o=sc.transform(ct,tkA); g,_=sc.user_decrypt(o,zn); res['authorized'].append(ber(g,mu))
        g,_=sc.user_decrypt(o,zn2); res['wrong_user_key'].append(ber(g,mu))
        # forged component for absent attribute 5: uniform random
        ct2=dict(ct); ct2['ci']=dict(ct['ci']); ct2['ci'][5]=ring.ntt(rng0.integers(0,ring.q,size=(P.mp,P.d)))
        o=sc.transform(ct2,tkC); g,_=sc.user_decrypt(o,zn); res['missing_attribute_forged'].append(ber(g,mu))
        # mixed keys: x from tkA, r2 from tkB used with ciphertext having 0,1,2,3? use a clause-like mix on X={0,1,2,3}
        ct3=sc.encrypt([0,1,2,3],mu)
        mix=dict(x_n=tkA['x_n'],r2_n=tkB['r2_n'],Pset=[2,3])
        o=sc.transform(ct3,mix); g,_=sc.user_decrypt(o,zn); res['collusion_mixed_keys'].append(ber(g,mu))
        # cloud's own view: v - (what cloud knows) -> decode v directly without cE z
        o=sc.transform(ct,tkA,False); ph=o['v']; gb=(((4*ph+ring.q)//(2*ring.q))%2).astype(np.int64); res['unblinded_cloud_view'].append(ber(gb,mu))
    for kx,v in res.items():
        rows.append(dict(test=kx,trials=N,bits=N*P.d,mean_ber=round(float(np.mean(v)),4),min_ber=round(float(np.min(v)),4),max_ber=round(float(np.max(v)),4)))
        print(rows[-1],flush=True)
    wcsv('negative.csv',rows,list(rows[0].keys()))

elif stage=='compress':
    import pickle
    q,d,uv=pickle.load(open(f'{OUT}/k7_outputs.pkl','rb'))
    k=7; rows=[]
    for du in (6,8,9,10,12):
        for dv in (2,3,4,5,6):
            errs=[]; be=0; tot=0
            for cE,v,zn,mu in uv:
                P=choose_params(7); ring=P.ring
                cEc=decompress(compress(cE,q,du),q,du); vc=decompress(compress(v,q,dv),q,dv)
                mask=ring.intt(ring.dot(ring.ntt(cEc),zn)); ph=(vc-mask)%q
                g=(((4*ph+q)//(2*q))%2); be+=int((g!=mu).sum()); tot+=d
                errs.append(phase_err(P,ph,mu)/q)
            e=np.concatenate(errs)
            rows.append(dict(du=du,dv=dv,bytes=round((k*d*du+d*dv)/8.0),std=round(float(e.std()),4),maxabs=round(float(np.abs(e).max()),4),bit_errors=be,bits=tot,
                             margin=round(0.25/float(e.std()),2)))
    wcsv('compress.csv',rows,list(rows[0].keys())); print(len(rows),'rows')

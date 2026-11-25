import math
from estimator import bits

LAMBDA_S = 64
ETA = math.sqrt(math.log(2*(1+2**LAMBDA_S))/math.pi)   # smoothing parameter of Z, eps=2^-64

def derive(k, d=256, w=4, base=2, sig_e=3.2, sig_R=math.sqrt(0.5), tail=8.0, lq_start=30):
    lq = lq_start
    for _ in range(60):
        q = 2.0**lq
        lg = math.ceil(lq/ math.log2(base))
        mp = k*lg            # gadget width m'
        mb = 2*k             # trapdoor-free columns
        m = mb+mp
        sigG = ETA*math.sqrt(base**2+1)
        s1 = sig_R*math.sqrt(w)*(math.sqrt(m*d)+math.sqrt(mp*d)+6)   # largest singular value of R_P (m d x m' d)
        s = 1.05*math.sqrt(sigG**2*(s1**2+1)+ETA**2)
        # noise std per coefficient of y = c0.r1 + cP.r2
        var = sig_e**2*(m*d*s**2) + sig_e**2*(sig_R**2*w)*(m*d)*(mp*d)*s**2
        # extra terms (message noise, user-side s^T e_z - e_E z): small
        var += sig_e**2 + 2*(k*d)*sig_e**4
        std = math.sqrt(var)
        need = math.log2(4*tail*std)+0.0
        if abs(need-lq) < 0.5 and need <= lq:
            break
        lq = max(lq, math.ceil(need))
    return dict(k=k,d=d,w=w,base=base,lq=lq,lg=lg,m=m,mp=mp,s=s,std=std,log2std=math.log2(std),n=k*d)

if __name__=="__main__":
    print("eta",ETA)
    for base in (2,):
        for k in (3,4,5,6,7,8,10):
            p = derive(k, base=base)
            b, c, qn = bits(p['n'], 2.0**p['lq'], 3.2)
            print(f"k={k} n={p['n']} lq={p['lq']} m={p['m']} s={p['s']:.0f} log2std={p['log2std']:.1f} beta={b} classical={c:.0f} quantum={qn:.0f}")

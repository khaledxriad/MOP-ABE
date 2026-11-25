"""MOP-ABE prototype: module-LWE key-policy ABE (conjunctive clauses) with a blinded transformation
key and a packed (Kyber-style) output.  Research prototype: NOT constant-time, NOT hardened."""
import numpy as np, math, time
from ring import Ring, find_primes
from sampling import Gadget, TrapGen, dgauss, ring_mul_small, ETA

SIG_E = 3.2
def sign_dist(rng, shape):
    return np.array([-1,0,0,1])[rng.integers(0,4,size=shape)]

class Params:
    """Parameter set.  `sigma_total` is the standard deviation of the phase error after
    packing (heuristic Gaussian analysis, Theorem 1); `margin` = (q/4)/sigma_total."""
    def __init__(self, k, d=256, prime_bits=(18,18), w_max=4, sig_Rm=3.2, du=10, dv=5, z_var=2.0/3.0):
        self.k, self.d, self.w_max, self.du, self.dv = k, d, w_max, du, dv
        b1, b2 = prime_bits
        p1 = find_primes(d, b1, 1)[0]
        p2 = [p for p in find_primes(d, b2, 2) if p != p1][0]
        self.primes = [p1, p2]
        self.ring = Ring(d, self.primes); self.q = self.ring.q
        self.gad = Gadget(self.q); self.l = self.gad.l
        self.mb = 2*k; self.mp = k*self.l; self.m = self.mb + self.mp
        self.sig_R = math.sqrt(0.5)      # std of the {-1,0,0,1} distribution of the encryption matrices R_i
        self.sig_Rm = sig_Rm             # master trapdoor R: discrete Gaussian width
        s1_sim = self.sig_R*math.sqrt(w_max)*(math.sqrt(self.m*d)+math.sqrt(self.mp*d)+6)
        s1_real = sig_Rm*(math.sqrt(self.mb*d)+math.sqrt(self.mp*d)+6)
        self.s1 = max(s1_sim, s1_real)
        sg = self.gad.sigma
        self.s = 1.05*math.sqrt(sg**2*(self.s1**2+1)+ETA**2)
        w = w_max
        var_y = SIG_E**2*(self.m*d*self.s**2) + SIG_E**2*(self.sig_R**2*w)*(self.m*d)*(self.mp*d)*self.s**2
        var_small = SIG_E**2 + 2*(k*d)*SIG_E**4               # e' and s^T e_z, e_E z (z ternary adds ~ k d z_var sig_e^2)
        var_small += SIG_E**2*k*d*z_var
        var_cE = (self.q/(2**du))**2/12.0 * (k*d*z_var)       # rounding of c_E times ternary z
        var_v = (self.q/(2**dv))**2/12.0                      # rounding of v
        self.var_parts = dict(y=var_y, small=var_small, cE=var_cE, v=var_v)
        self.noise_std_pred = math.sqrt(var_y+var_small)      # unpacked noise
        self.sigma_total = math.sqrt(var_y+var_small+var_cE+var_v)
        self.margin = (self.q/4.0)/self.sigma_total
    def describe(self):
        return dict(k=self.k,d=self.d,n=self.k*self.d,q=self.q,log2q=math.log2(self.q),l=self.l,m=self.m,mp=self.mp,
                    s=self.s,sigma_G=float(self.gad.sigma),log2_noise_std=math.log2(self.noise_std_pred),
                    sigma_total_over_q=self.sigma_total/self.q, margin=self.margin)

def choose_params(k, target_margin=8.4, **kw):
    for b1,b2 in [(18,18),(19,18),(19,19),(20,19),(20,20)]:
        P = Params(k, prime_bits=(b1,b2), **kw)
        if P.margin >= target_margin: return P
    return P

def compress(x, q, bits):
    return ((x.astype(np.int64)*(1 << (bits+1)) + q) // (2*q)) % (1 << bits)
def decompress(y, q, bits):
    return ((y.astype(np.int64)*2*q + (1 << bits)) // (1 << (bits+1)))

class MOP:
    def __init__(self, P, n_attr, seed=1):
        self.P = P; self.ring = P.ring; self.rng = np.random.default_rng(seed); self.ell = n_attr
    # ---------- helpers
    def _const_ntt(self, vals):
        """constant polynomials (ints) -> NTT form, shape (J,)+vals.shape+(d,)"""
        vals = np.asarray(vals, dtype=np.int64)
        out = np.empty((self.ring.J,)+vals.shape+(self.P.d,))
        for j,p in enumerate(self.ring.primes):
            out[j] = np.mod(vals, p).astype(np.float64)[...,None]
        return out
    def _gauss_poly(self, shape, sigma=SIG_E):
        return dgauss(self.rng, sigma, np.zeros(shape+(self.P.d,)))
    # ---------- Setup (authority)
    def setup(self):
        P, ring, rng = self.P, self.ring, self.rng; k,d = P.k,P.d
        t0 = time.perf_counter()
        Ahat = rng.integers(0, ring.q, size=(k,k,d))
        Ahat_n = ring.ntt(Ahat)
        I_n = self._const_ntt(np.eye(k, dtype=np.int64))
        Abar_n = np.concatenate([I_n, Ahat_n], axis=2)                    # (J,k,2k,d)
        R = sign_dist(rng, (P.mb, P.mp, d)) if P.sig_Rm <= 0.71 else dgauss(rng, P.sig_Rm, np.zeros((P.mb,P.mp,d)))
        Rn = ring.ntt(R)                                                  # (J,mb,mp,d)
        AbarR = np.mod(np.einsum('jrcd,jcmd->jrmd', Abar_n, Rn), ring._pcol(4))
        g = np.array([[1 << i for i in range(P.l)]]*k)                    # not used directly
        Gm = np.zeros((k, P.mp), dtype=np.int64)
        for r in range(k):
            for i in range(P.l): Gm[r, r*P.l+i] = 1 << i
        G_n = self._const_ntt(Gm)
        A2 = ring.sub(G_n, AbarR)
        self.A_n = np.concatenate([Abar_n, A2], axis=2)                   # (J,k,m,d)
        self.G_n = G_n
        self.trap = TrapGen(ring, k, d, R, P.gad, P.s, rng)
        self.u_n = ring.ntt(rng.integers(0, ring.q, size=(k,d)))
        # attribute matrices B_i = A R_i - G with short sign matrices R_i that are erased at once.  They are
        # statistically uniform (leftover hash lemma); the form is what lets the security reduction answer key
        # queries without the trapdoor of A (Sec. 6).
        self.B_n = []
        for _ in range(self.ell):
            Ri_n = ring.ntt(sign_dist(rng, (P.m, P.mp, d)))                       # (J,m,mp,d)
            prod = np.empty((ring.J, k, P.mp, d))
            for j,p in enumerate(ring.primes):
                Aj = np.ascontiguousarray(np.transpose(self.A_n[j], (2,0,1)))     # (d,k,m)
                Rj = np.ascontiguousarray(np.transpose(Ri_n[j], (2,0,1)))         # (d,m,mp)
                prod[j] = np.transpose(np.mod(np.matmul(Aj, Rj), p), (1,2,0))
            self.B_n.append(ring.sub(prod, G_n)); del Ri_n
        self.E_n = ring.ntt(rng.integers(0, ring.q, size=(k,k,d)))
        return time.perf_counter()-t0
    # ---------- user key pair
    def user_keygen(self):
        P, ring = self.P, self.ring
        z = self.rng.integers(-1, 2, size=(P.k, P.d))                    # ternary single key
        ez = self._gauss_poly((P.k,))
        zn = ring.ntt(z)
        w_n = ring.add(ring.matvec(self.E_n, zn), ring.ntt(ez))
        return z, zn, w_n
    # ---------- transformation key for clause Pset and user blinding w
    def tk_gen(self, Pset, w_n):
        P, ring, rng = self.P, self.ring, self.rng; k,d = P.k, P.d
        BP = self.B_n[Pset[0]]
        for i in Pset[1:]: BP = ring.add(BP, self.B_n[i])
        r2 = dgauss(rng, P.s, np.zeros((P.mp, d)))
        r2n = ring.ntt(r2)
        v_n = ring.sub(ring.sub(self.u_n, w_n), ring.matvec(BP, r2n))     # target for A
        p1, p2 = self.trap.sample_perturbation(rng)
        pn = ring.ntt(np.concatenate([p1, p2], axis=0))
        vp_n = ring.sub(v_n, ring.matvec(self.A_n, pn))
        vp = ring.intt(vp_n)                                              # (k,d) in [0,q)
        z = P.gad.sample(rng, vp.ravel().astype(np.int64))                # (k*d, l)
        z = z.reshape(k, d, P.l).transpose(0,2,1).reshape(P.mp, d)        # row index r*l+i
        Rz = np.zeros((P.mb, d))
        for a in range(P.mb):
            Rz[a] = ring_mul_small(self.trap.R[a].astype(np.float64), z.astype(np.float64), d).sum(axis=0)
        x1 = p1 + np.rint(Rz).astype(np.int64)
        x2 = p2 + z
        x = np.concatenate([x1, x2], axis=0)                              # (m,d)
        return dict(x_n=ring.ntt(x), r2_n=r2n, x=x, r2=r2, Pset=list(Pset))
    # ---------- encryption (KP: attribute set X)
    def encrypt(self, X, mu_bits):
        P, ring, rng = self.P, self.ring, self.rng; k,d,m,mp = P.k,P.d,P.m,P.mp
        s = self._gauss_poly((k,)); sn = ring.ntt(s)
        e0 = self._gauss_poly((m,)); e0n = ring.ntt(e0)
        ct = {}
        ct['c0'] = ring.add(ring.vecmat(sn, self.A_n), e0n)
        ct['ci'] = {}
        for i in X:
            si = ring.vecmat(sn, self.B_n[i])
            noise = np.zeros((ring.J, mp, d))
            CH = 32
            for t0 in range(0, m, CH):
                t1 = min(m, t0+CH)
                Rc = sign_dist(rng, (t1-t0, mp, d))
                Rn = ring.ntt(Rc)
                noise = np.mod(noise + np.einsum('jtcd,jtd->jcd', Rn, e0n[:, t0:t1, :]), ring._pcol(3))
            ct['ci'][i] = ring.add(si, noise)
        eE = self._gauss_poly((k,)); ep = self._gauss_poly(())
        ct['cE'] = ring.add(ring.vecmat(sn, self.E_n), ring.ntt(eE))
        msg = (np.asarray(mu_bits, dtype=np.int64) * (ring.q//2))
        ct['cp'] = ring.add(ring.add(ring.dot(sn, self.u_n[:, :, :]) if False else np.mod(np.einsum('jcd,jcd->jd', sn, self.u_n), ring._pcol(2)), ring.ntt(ep)), ring.ntt(msg))
        ct['X'] = list(X)
        return ct
    # ---------- cloud: transform + pack
    def transform(self, ct, tk, compress_out=True):
        P, ring = self.P, self.ring
        y = ring.dot(ct['c0'], tk['x_n'])
        cP = ct['ci'][tk['Pset'][0]]
        for i in tk['Pset'][1:]: cP = ring.add(cP, ct['ci'][i])
        y = ring.add(y, ring.dot(cP, tk['r2_n']))
        v_n = ring.sub(ct['cp'], y)
        v = ring.intt(v_n); cE = ring.intt(ct['cE'])
        if compress_out:
            return dict(cE=compress(cE, ring.q, P.du), v=compress(v, ring.q, P.dv), packed=True)
        return dict(cE=cE, v=v, packed=False)
    # ---------- mobile: single-key decryption
    def user_decrypt(self, out, zn):
        P, ring = self.P, self.ring
        if out['packed']:
            cE = decompress(out['cE'], ring.q, P.du); v = decompress(out['v'], ring.q, P.dv)
        else:
            cE, v = out['cE'], out['v']
        mask = ring.intt(ring.dot(ring.ntt(cE), zn))
        ph = (v - mask) % ring.q
        return (((4*ph + ring.q) // (2*ring.q)) % 2).astype(np.int64), ph

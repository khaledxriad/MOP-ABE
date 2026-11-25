"""Discrete Gaussians, gadget-lattice sampler (Klein/GPV), and MP12-style
module trapdoor preimage sampling with a Peikert-rounded perturbation."""
import numpy as np, math

LAMBDA_S = 64
ETA = math.sqrt(math.log(2*(1+2.0**LAMBDA_S))/math.pi)   # smoothing parameter eta_eps(Z)

def dgauss(rng, sigma, center, tau=12.0):
    """vectorised discrete Gaussian D_{Z,sigma,c} (rejection sampling, tail cut at tau*sigma)"""
    c = np.asarray(center, dtype=np.float64); shape = c.shape; c = c.ravel()
    out = np.empty(c.shape, dtype=np.int64); todo = np.arange(c.size)
    T = int(math.ceil(tau*sigma)); K = max(8, int(3*(2*T+1)/(sigma*math.sqrt(2*math.pi)))+1)
    while todo.size:
        base = np.floor(c[todo])
        cand = base[:,None] + rng.integers(-T, T+1, size=(todo.size, K))
        acc = rng.random((todo.size, K)) < np.exp(-(cand - c[todo][:,None])**2/(2*sigma*sigma))
        has = acc.any(axis=1); first = acc.argmax(axis=1)
        sel = todo[has]
        out[sel] = cand[has, first[has]].astype(np.int64)
        todo = todo[~has]
    return out.reshape(shape)

class Gadget:
    """g = (1,2,4,...,2^(l-1)) with q <= 2^l; lattice L = {z : <g,z> = 0 mod q}."""
    def __init__(self, q):
        self.q = int(q); self.l = int(q-1).bit_length()
        l = self.l
        S = np.zeros((l,l))
        for j in range(l-1):
            S[j,j] = 2; S[j+1,j] = -1
        for i in range(l): S[i,l-1] = (self.q >> i) & 1
        self.S = S
        # Gram-Schmidt in column order
        Q = np.zeros((l,l)); G = np.zeros((l,l))
        for j in range(l):
            v = S[:,j].copy()
            for i in range(j):
                v -= (S[:,j] @ Q[:,i]) * Q[:,i]
            G[:,j] = v; Q[:,j] = v/np.linalg.norm(v)
        self.GS = G; self.gsn = np.linalg.norm(G, axis=0)
        self.max_gs = self.gsn.max()
        self.sigma = ETA*self.max_gs

    def sample(self, rng, v):
        """z in Z^l with <g,z> = v mod q, z ~ D_{Lambda_v^perp(g), sigma}; v: int array (N,) in [0,q)."""
        v = np.asarray(v, dtype=np.int64); N = v.size; l = self.l
        z0 = ((v[:,None] >> np.arange(l)) & 1).astype(np.float64)       # binary expansion, <g,z0>=v
        c = -z0.copy()                                                  # centre: -z0
        y = np.zeros((N,l))
        for j in range(l-1, -1, -1):
            gj = self.GS[:,j]
            cj = (c @ gj) / (gj @ gj)
            zj = dgauss(rng, self.sigma/self.gsn[j], cj)
            y += zj[:,None]*self.S[:,j][None,:]
            c = c - zj[:,None]*self.S[:,j][None,:]
        return (z0 + y).astype(np.int64)

class Embedding:
    """canonical embedding of Z[X]/(X^d+1) at the d/2 roots zeta_j = exp(i*pi*(2j+1)/d), j<d/2."""
    def __init__(self, d):
        self.d = d; j = np.arange(d//2)[:,None]; t = np.arange(d)[None,:]
        self.V = np.exp(1j*np.pi*(2*j+1)*t/d)          # (d/2, d): evaluation
        self.Vinv = self.V.conj() * (2.0/d)            # real-part inverse: p_t = Re sum_j p^_j conj(zeta^t)*(2/d)
    def fwd(self, x):          # x (..., d) real -> (..., d/2) complex
        return x @ self.V.T
    def inv(self, X):          # X (..., d/2) complex -> (..., d) real
        return np.real(X @ self.Vinv)

def ring_mul_small(a, b, d):
    """exact-enough negacyclic product of small-integer ring-element arrays via float FFT.
    a: (..., d), b: (..., d) broadcastable -> (..., d) float"""
    n = 2*d
    fa = np.fft.rfft(a, n=n, axis=-1); fb = np.fft.rfft(b, n=n, axis=-1)
    c = np.fft.irfft(fa*fb, n=n, axis=-1)
    return c[..., :d] - c[..., d:]

class TrapGen:
    """Module gadget trapdoor: A = [Abar | G - Abar R], Abar = [I | Ahat] (k x 2k), R in R^{2k x m'} small.
    Preimage sampling follows Micciancio-Peikert (EUROCRYPT 2012, Sec. 4) with the covariance
    Sigma_p = s^2 I - sigma_G^2 [R;I][R;I]^T realised by conditional sampling and Peikert rounding."""
    def __init__(self, ring, k, d, R_coef, gadget, s, rng):
        self.ring = ring; self.k = k; self.d = d; self.gad = gadget; self.s = s
        self.mb = 2*k; self.mp = k*gadget.l
        self.R = R_coef                          # int array (mb, mp, d) coefficients
        self.emb = Embedding(d)
        Rh = self.emb.fwd(R_coef.astype(np.float64))       # (mb, mp, d/2)
        self.Rh = np.transpose(Rh, (2,0,1))                # (d/2, mb, mp)
        sg2 = gadget.sigma**2; s2 = s*s
        RRH = np.einsum('jab,jcb->jac', self.Rh, self.Rh.conj())   # R R^H per slot
        self.c_coef = sg2*s2/(s2 - sg2)
        r2 = ETA**2*1.0001
        C = s2*np.eye(self.mb)[None] - self.c_coef*RRH - r2*np.eye(self.mb)[None]
        self.L = np.linalg.cholesky(C)                      # (d/2, mb, mb)
        self.r_round = math.sqrt(r2)
        self.sg2 = sg2; self.s2 = s2

    def sample_perturbation(self, rng):
        """p = (p1 in Z^{mb x d}, p2 in Z^{mp x d})"""
        mb, mp, d = self.mb, self.mp, self.d
        p2 = dgauss(rng, math.sqrt(self.s2 - self.sg2), np.zeros((mp, d)))
        # mean of p1: -sg2/(s2-sg2) * R p2  (ring product, small x medium -> float FFT)
        Rp2 = np.zeros((mb, d))
        for a in range(mb):
            Rp2[a] = ring_mul_small(self.R[a].astype(np.float64), p2.astype(np.float64), d).sum(axis=0)
        mean = -self.sg2/(self.s2-self.sg2) * Rp2
        # continuous part with covariance C (ring-structured), per embedding slot
        g = (rng.standard_normal((d//2, mb)) + 1j*rng.standard_normal((d//2, mb)))/math.sqrt(2)
        xh = math.sqrt(d) * np.einsum('jab,jb->ja', self.L, g)      # (d/2, mb)
        x = self.emb.inv(xh.T)                                      # (mb, d)
        p1 = dgauss(rng, self.r_round, mean + x)
        return p1, p2

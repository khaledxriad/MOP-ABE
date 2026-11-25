"""Negacyclic ring R_q = Z_q[X]/(X^d+1) with q = p_1 * p_2 * ... (RNS).
All transforms are dense DFT-style matrix products in float64 (exact, because
p < 2^20 and d <= 512 keep every partial sum below 2^53)."""
import numpy as np, math

def is_prime(n):
    if n < 2: return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n % p == 0: return n == p
    r, s = n-1, 0
    while r % 2 == 0: r//=2; s+=1
    for a in (2,3,5,7,11,13,17,19,23,29,31,37):
        x = pow(a, r, n)
        if x in (1, n-1): continue
        for _ in range(s-1):
            x = x*x % n
            if x == n-1: break
        else: return False
    return True

def find_primes(d, bits_each, count):
    """largest `count` distinct primes below 2^bits_each with p = 1 mod 2d"""
    out = []; p = (1 << bits_each) // (2*d) * (2*d) + 1
    while len(out) < count:
        p -= 2*d
        if is_prime(p): out.append(p)
    return out

def prim_root_2d(p, d):
    for a in range(2, p):
        psi = pow(a, (p-1)//(2*d), p)
        if pow(psi, d, p) == p-1:
            return psi
    raise ValueError

class Ring:
    def __init__(self, d, primes):
        self.d = d; self.primes = list(primes); self.J = len(primes)
        self.q = 1
        for p in primes: self.q *= p
        self.F = []; self.Fi = []
        idx = np.arange(d)
        for p in primes:
            psi = prim_root_2d(p, d)
            # evaluation at psi^(2i+1)
            ex = (np.outer(2*idx+1, idx)) % (2*d)
            pw = np.array([pow(psi, int(e), p) for e in range(2*d)], dtype=np.int64)
            F = pw[ex]                                   # F[i,t] = psi^((2i+1)t)
            ipw = np.array([pow(psi, (2*d - e) % (2*d), p) for e in range(2*d)], dtype=np.int64)
            dinv = pow(d, -1, p)
            Fi = (ipw[ex].T * dinv) % p                   # Fi[t,i] = d^-1 psi^{-(2i+1)t}
            self.F.append(F.T.astype(np.float64))        # x @ F.T-> (…,i)
            self.Fi.append(Fi.T.astype(np.float64))      # Y @ Fi.T -> (…,t)
        self.pf = np.array(primes, dtype=np.float64)
        self.inv_p0_mod_p1 = pow(primes[0], -1, primes[1]) if self.J == 2 else None

    # ---- helpers
    def _pcol(self, ndim):
        return self.pf.reshape((self.J,)+(1,)*(ndim-1))

    def ntt(self, x):
        """integer array (..., d) (any sign/size) -> float64 (J, ..., d) residues in NTT domain"""
        x = np.asarray(x)
        out = np.empty((self.J,)+x.shape, dtype=np.float64)
        for j, p in enumerate(self.primes):
            xr = np.mod(x, p).astype(np.float64)
            out[j] = np.mod(xr @ self.F[j], p)
        return out

    def intt(self, X):
        """(J,...,d) NTT-domain -> int64 coefficients in [0,q)"""
        res = []
        for j, p in enumerate(self.primes):
            res.append(np.mod(X[j] @ self.Fi[j], p).astype(np.int64))
        if self.J == 1: return res[0]
        x0, x1 = res[0], res[1]
        t = ((x1 - x0) % self.primes[1]) * self.inv_p0_mod_p1 % self.primes[1]
        return x0 + self.primes[0] * t

    def center(self, x):
        x = np.mod(x, self.q)
        return np.where(x > self.q//2, x - self.q, x)

    # ---- NTT-domain arithmetic
    def add(self, a, b): return np.mod(a + b, self._pcol(a.ndim))
    def sub(self, a, b): return np.mod(a - b, self._pcol(a.ndim))
    def mul(self, a, b): return np.mod(a * b, self._pcol(max(a.ndim, b.ndim)))
    def matvec(self, M, v):      # M (J,r,c,d), v (J,c,d) -> (J,r,d)
        return np.mod(np.einsum('jrcd,jcd->jrd', M, v), self._pcol(3))
    def vecmat(self, v, M):      # v (J,r,d), M (J,r,c,d) -> (J,c,d)
        return np.mod(np.einsum('jrd,jrcd->jcd', v, M), self._pcol(3))
    def dot(self, a, b):         # (J,c,d)x(J,c,d) -> (J,d)
        return np.mod(np.einsum('jcd,jcd->jd', a, b), self._pcol(2))

    def nbytes_elem(self):
        return self.d * math.ceil(math.log2(self.q)) / 8.0

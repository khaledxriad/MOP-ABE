"""Core-SVP primal-attack estimate for (Module-)LWE.  Follows the usual
uSVP analysis (Alkim et al. 2016, Albrecht et al. 2015): the attack succeeds
with BKZ block size beta if  sqrt(beta)*sigma <= delta(beta)^(2*beta-dim-1) * q^(m/dim),
dim = m + n + 1.  Cost model: 0.292*beta (classical sieving), 0.265*beta (quantum).
This is a back-of-the-envelope estimator, NOT a replacement for the lattice-estimator."""
import math

def delta(b):
    return (((math.pi*b)**(1.0/b))*b/(2*math.pi*math.e))**(1.0/(2*(b-1)))

def best_beta(n, q, sigma, m_max=None):
    m_max = m_max or 2*n
    best = None
    for m in range(max(10, n//4), m_max+1, max(1, n//64)):
        dim = m+n+1
        # smallest beta that satisfies the inequality
        lo, hi = 40, dim
        ok = lambda b: math.sqrt(b)*sigma <= delta(b)**(2*b-dim-1) * q**(m/dim)
        if not ok(hi):
            continue
        while lo < hi:
            mid = (lo+hi)//2
            if ok(mid): hi = mid
            else: lo = mid+1
        if best is None or lo < best[0]:
            best = (lo, m)
    return best

def bits(n, q, sigma):
    b, m = best_beta(n, q, sigma)
    return b, 0.292*b, 0.265*b

if __name__ == "__main__":
    # sanity: Kyber512 (n=512,q=3329,sigma=1 (eta=3 -> var 1.5)), Kyber768 (n=768)
    for name, n, q, s in [("Kyber512",512,3329,math.sqrt(1.5)),("Kyber768",768,3329,1.0),("Kyber1024",1024,3329,1.0)]:
        print(name, bits(n,q,s))
    for n,lq in [(1024,27),(1536,40),(2048,54)]:
        print("HES-like", n, lq, bits(n, 2.0**lq, 3.2))

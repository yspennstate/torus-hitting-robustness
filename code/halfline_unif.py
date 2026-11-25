"""Second check of the half-line inequality of Lemma 4.1, f_t(1) f_t(j) >= f_t(0) f_t(j+1) with f_t(j) = P_j(X_t = 0),
for the chain on {0,...,N} with rate 1 up at 0, rate a = (1+e)/2 up and b = (1-e)/2 down inside, and holding at N.
f_t is computed by uniformization, f_t = sum_n Poisson(n; t) P^n e_0, a sum of nonnegative terms, so no matrix
exponential is formed. N = 600, 60 values of t in [0.05, 3000], all j < 300.
    python3 halfline_unif.py
"""
import numpy as np
from scipy.stats import poisson

N = 600
ts = np.geomspace(0.05, 3000, 60)
for e in (0.0, 0.01, 0.05, 0.2):
    a, b = (1 + e) / 2, (1 - e) / 2
    P = np.zeros((N + 1, N + 1)); P[0, 1] = 1.0
    for u in range(1, N):
        P[u, u + 1] = a; P[u, u - 1] = b
    P[N, N - 1] = b; P[N, N] = a
    nmax = int(ts[-1] + 12 * np.sqrt(ts[-1]) + 50)
    W = poisson.pmf(np.arange(nmax)[:, None], ts[None, :])
    v = np.zeros(N + 1); v[0] = 1.0
    F = np.zeros((N + 1, len(ts)))
    for n in range(nmax):
        F += np.outer(v, W[n]); v = P @ v
    if not np.all(np.isfinite(F)):
        raise SystemExit(f'e={e}: non-finite value')
    worst, where = -np.inf, None
    for k, t in enumerate(ts):
        f = F[:, k]
        for j in range(1, 300):
            if f[j + 1] < 1e-200:
                break
            r = (f[0] * f[j + 1] - f[1] * f[j]) / (f[1] * f[j])
            if r > worst:
                worst, where = r, (round(float(t), 2), j)
    print(f'e={e}: max over t, j of (f0 f_(j+1) - f1 f_j)/(f1 f_j) = {worst:.3e} at (t, j) = {where}', flush=True)

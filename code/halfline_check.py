"""Independent check of the half-line (*): p_t(1,0) p_t(j,0) >= p_t(0,0) p_t(j+1,0) for every j, for the chain on
{0,1,2,...} with rate 1 up at 0 and a = (1+e)/2 up, b = (1-e)/2 down inside (drift e away from 0).
 - e = 0: the chain is |SRW| on Z, p_t(j,0) = exp(-t) I_j(t), so (*) is I_{j+1}/I_j <= I_1/I_0 (Turan-type monotonicity
   of Bessel ratios); checked with scipy.special.ive.
 - e > 0: truncated chain on {0..N}, N = 600, t <= 3000 (wall effect <= exp(-N^2/(2t)) ~ e^-60), all j <= 300.
    python halfline_check.py
"""
import numpy as np
from scipy.linalg import expm
from scipy.special import ive

worst = 0.0
for t in np.geomspace(0.01, 500, 200):
    I = ive(np.arange(0, 400), t)
    for j in range(1, 300):
        if I[j + 1] < 1e-280:
            break
        v = (I[0] * I[j + 1] - I[1] * I[j]) / (I[1] * I[j])
        worst = max(worst, v)
print(f'e=0 (Bessel): worst relative violation {worst:.2e}', flush=True)

N = 600
for e in (0.0, 0.01, 0.05, 0.2):
    a, b = (1 + e) / 2, (1 - e) / 2
    G = np.zeros((N + 1, N + 1)); G[0, 1] = 1.0
    for u in range(1, N):
        G[u, u + 1] = a; G[u, u - 1] = b
    G[N, N - 1] = b
    G -= np.diag(G.sum(1))
    worst, where = 0.0, None
    for t in np.geomspace(0.05, 3000, 60):
        f = expm(t * G)[:, 0]
        for j in range(1, 300):
            if f[j + 1] < 1e-200:
                break
            v = (f[0] * f[j + 1] - f[1] * f[j]) / (f[1] * f[j])
            if v > worst:
                worst, where = v, (round(t, 2), j)
    print(f'e={e}: worst relative violation {worst:.2e} at {where}', flush=True)

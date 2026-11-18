"""How far does (*) p_t(1,0) p_t(j,0) >= p_t(0,0) p_t(j+1,0) hold?  Prints the largest J with (*) true for all
1 <= j <= J and all t on the grid, per L and c, and m - J (is the failure a boundary layer at the far wall, or does it
start at a fixed fraction of m?).  Entries below 1e-11 are masked (expm accuracy is relative to the norm).
    python star_range.py
"""
import numpy as np
from scipy.linalg import expm

for L in (16, 32, 64, 128, 256):
    m = L // 2
    for c in (0.5, 1.0, 2.0, 4.0):
        e = c / L
        G = np.zeros((m + 1, m + 1))
        G[0, 1] = 1.0
        for u in range(1, m):
            G[u, u + 1] = (1 + e) / 2
            G[u, u - 1] = (1 - e) / 2
        G[m, m - 1] = 1.0
        G -= np.diag(G.sum(1))
        first_bad, worst_t = m, None
        for t in np.geomspace(0.01, 50.0 * L * L, 300):
            f = expm(t * G)[:, 0]
            j = np.arange(1, m)
            ok = np.minimum(f[j], f[j + 1]) > 1e-11
            v = (f[0] * f[j + 1] - f[1] * f[j]) / np.maximum(f[1] * f[j], 1e-300)
            bad = j[ok & (v > 1e-12)]
            if bad.size and int(bad.min()) < first_bad:
                first_bad, worst_t = int(bad.min()), t
        print(f'L={L} c={c}: (*) holds for j <= {first_bad - 1} (m={m}, m-J={m - first_bad + 1}); first failure at t~{worst_t}',
              flush=True)

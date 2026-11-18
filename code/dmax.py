"""Size and location of the deficit d = B(V_Q) - Q V_Q of the product adversary.
If max d < 1 then E^{P'}_x H_0 <= V_Q(x) + max d * E^{P'}_x H_0 for every eps-adversary P' (supermartingale V_Q(X_t) + t
up to d), so t_hit^+ <= max V_Q / (1 - max d).
    python dmax.py D L1 L2 ...
"""
import sys

import numpy as np

from gate import best_response, killed_solve, q_matrix
from grid_rho import neighbours, solve

if __name__ == '__main__':
    D = int(sys.argv[1])
    for L in (int(a) for a in sys.argv[2:]):
        nb = neighbours(L, D)
        t0, _ = solve(L, D, 0.0, nb=nb)
        for c in (0.5, 1.0, 2.0):
            eps = c / L
            Q = q_matrix(L, D, eps, nb)
            V = killed_solve(Q, np.ones(L ** D))
            d = best_response(V, nb, eps) - Q @ V
            d[0] = 0.0
            i = int(np.argmax(d))
            xi = tuple(int(v) for v in np.unravel_index(i, (L,) * D))
            pos = int((d > 1e-9).sum())
            print(f'D={D} L={L} c={c}: max d={d.max():.4f} at {xi}; sites d>1e-9: {pos}; '
                  f'bound V/(1-dmax)/t = {V.max()/(1-d.max())/t0 if d.max() < 1 else float("inf"):.5f}; V/t={V.max()/t0:.5f}',
                  flush=True)

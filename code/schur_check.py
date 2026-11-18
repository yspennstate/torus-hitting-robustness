"""Comparison lemma check: for k_a = 0 < k_c (folded), is V_Q(k+e_a) <= V_Q(k+e_c)?  And for k_a = m > k_c, is
V_Q(k-e_a) >= V_Q(k-e_c)?  Prints every folded class where either fails (with the size of the failure), so we see whether the
failures stay away from the origin.
    python schur_check.py D L1 L2 ...
"""
import itertools
import sys

import numpy as np

from gate import killed_solve, q_matrix
from grid_rho import neighbours

if __name__ == '__main__':
    D = int(sys.argv[1])
    for L in (int(a) for a in sys.argv[2:]):
        m = L // 2
        nb = neighbours(L, D)
        for c in (1.0, 2.0):
            eps = c / L
            V = killed_solve(q_matrix(L, D, eps, nb), np.ones(L ** D))
            F = V.reshape((L,) * D)[(slice(0, m + 1),) * D]          # folded table V[k], k in {0..m}^D
            bad = []
            for k in itertools.product(range(m + 1), repeat=D):
                if max(k) == 0 or list(k) != sorted(k):             # one representative per permutation class
                    continue
                for a in range(D):
                    for cc in range(D):
                        if k[a] == 0 < k[cc] < m:
                            ka = list(k); ka[a] = 1
                            kc = list(k); kc[cc] += 1
                            gap = F[tuple(ka)] - F[tuple(kc)]
                            if gap > 1e-9 * V.max():
                                bad.append(('0', k, gap))
                        if k[a] == m > k[cc] > 0:
                            ka = list(k); ka[a] = m - 1
                            kc = list(k); kc[cc] -= 1
                            gap = F[tuple(kc)] - F[tuple(ka)]
                            if gap > 1e-9 * V.max():
                                bad.append(('m', k, gap))
            near = [b for b in bad if max(b[1]) <= m // 2]
            print(f'D={D} L={L} c={c}: failures {len(bad)}, with all k <= m/2: {len(near)}; '
                  f'largest gaps {sorted(bad, key=lambda b: -b[2])[:4]}', flush=True)

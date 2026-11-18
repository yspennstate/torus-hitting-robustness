"""Is the optimal eps-adversary on the torus Z_L^D the product chain that, in every coordinate, gives the
neighbour farther from the target (cycle distance) probability (1+eps)/(2D) and the nearer one (1-eps)/(2D)
(1/(2D) each when the coordinate is 0 or L/2)?  Compare its max hitting time of 0 with the exact optimum from
grid_rho.solve (policy iteration), check coordinatewise monotonicity of its value, and compare the ratio with the
reversible-chain prediction ((e^c-1)/c)^D (stationary mass of 0 under the product chain).
    python product_check.py D L1 L2 ...
"""
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from grid_rho import neighbours, solve


def product_value(L, D, eps, nb):
    N = L ** D
    coords = np.array(np.unravel_index(np.arange(N), (L,) * D)).T
    dist = np.minimum(coords, L - coords)                      # cycle distance per coordinate
    P = np.zeros((N, 2 * D))
    for a in range(D):
        for j, s in enumerate((1, -1)):
            c = (coords[:, a] + s) % L
            dn = np.minimum(c, L - c)
            out = dn > dist[:, a]
            inn = dn < dist[:, a]
            p = np.full(N, 1.0 / (2 * D))
            p[out & ~(dist[:, a] == 0)] = (1 + eps) / (2 * D)
            p[inn & ~(2 * dist[:, a] == L)] = (1 - eps) / (2 * D)
            P[:, 2 * a + j] = p
    assert np.allclose(P.sum(1), 1)
    rows = np.repeat(np.arange(N), 2 * D)
    A = sp.csr_matrix((P.ravel(), (rows, nb.ravel())), shape=(N, N))
    keep = np.arange(1, N)
    M = sp.identity(N - 1, format='csr') - A[keep][:, keep]
    V = np.zeros(N); V[keep] = spl.spsolve(M.tocsc(), np.ones(N - 1))
    # coordinatewise monotonicity: V(x) <= V(x') when x' is one step farther in one coordinate
    worst = 0.0
    for a in range(D):
        for s in (1, -1):
            c = coords.copy(); c[:, a] = (c[:, a] + s) % L
            j = np.ravel_multi_index(c.T, (L,) * D)
            farther = np.minimum(c[:, a], L - c[:, a]) > dist[:, a]
            worst = max(worst, np.max((V - V[j])[farther], initial=-np.inf))
    return V.max(), worst


if __name__ == '__main__':
    import os
    PRODONLY = bool(os.environ.get('PRODONLY'))
    D = int(sys.argv[1])
    for L in (int(a) for a in sys.argv[2:]):
        nb = neighbours(L, D)
        base, V0 = solve(L, D, 0.0, nb=nb)
        for c in (0.5, 1.0, 2.0):
            eps = c / L
            opt = solve(L, D, eps, V0=V0, nb=nb)[0] if not PRODONLY else float('nan')
            prod, worst = product_value(L, D, eps, nb)
            pred = ((np.exp(c) - 1) / c) ** D
            print(f'D={D} L={L} c={c}: opt/t={opt/base:.6f} prod/t={prod/base:.6f} rel.gap={(opt-prod)/opt:.2e} '
                  f'max(V(x)-V(farther))={worst:.3e} pred={pred:.4f}', flush=True)

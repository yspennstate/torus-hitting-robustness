"""Robustness of grids (Zuckerman, ITCS 2026, Sec. 1.10). Exact robust max hitting time t_hit^+ on the torus
Z_L^D under the standard eps-adversary (each of the 2D neighbours gets a probability in [(1-eps)/(2D), (1+eps)/(2D)]),
by policy iteration: the best response to a value vector gives every neighbour its lower bound and pours the slack
eps onto the neighbours of highest value. Target 0 (vertex-transitive, so t_hit^+ = max_x V(x)).
Question: at eps = c/L, does t_hit^+/t_hit converge as L grows (rho_hit = Theta(1/L) = Theta(1/diam))
or grow without bound?
    python grid_rho.py [D]
"""
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def neighbours(L, D):
    N = L ** D
    idx = np.arange(N)
    coords = np.array(np.unravel_index(idx, (L,) * D)).T
    nb = []
    for a in range(D):
        for s in (1, -1):
            c = coords.copy(); c[:, a] = (c[:, a] + s) % L
            nb.append(np.ravel_multi_index(c.T, (L,) * D))
    return np.stack(nb, axis=1)                      # N x 2D


def solve(L, D, eps, V0=None, nb=None):
    nb = neighbours(L, D) if nb is None else nb
    N, k = nb.shape
    lo, hi = (1 - eps) / k, (1 + eps) / k
    V = np.zeros(N) if V0 is None else V0.copy()
    for it in range(200):
        order = np.argsort(-V[nb], axis=1, kind='stable')
        room = hi - lo
        add = np.minimum(room, np.maximum(0.0, eps - room * np.arange(k)))     # slack poured in value order
        P = np.full((N, k), lo)
        np.put_along_axis(P, order, lo + add[None, :], axis=1)
        rows = np.repeat(np.arange(N), k)
        A = sp.csr_matrix((P.ravel(), (rows, nb.ravel())), shape=(N, N))
        keep = np.arange(1, N)                                                  # drop the target
        M = sp.identity(N - 1, format='csr') - A[keep][:, keep]
        Vn = np.zeros(N); Vn[keep] = spl.spsolve(M.tocsc(), np.ones(N - 1))
        if np.max(np.abs(Vn - V)) < 1e-9 * Vn.max():
            V = Vn
            break
        V = Vn
    return V.max(), V


if __name__ == '__main__':
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    Ls = tuple(int(a) for a in sys.argv[2:]) or ((8, 12, 16, 24, 32, 48) if D == 2 else (4, 6, 8, 10, 12))
    print(f'D={D}  ratio t_hit^+(c/L)/t_hit for c = 0.5, 1, 2, 4')
    for L in Ls:
        nb = neighbours(L, D)
        base, V0 = solve(L, D, 0.0, nb=nb)
        out = []
        for c in (0.5, 1.0, 2.0, 4.0):
            t, _ = solve(L, D, c / L, V0=V0, nb=nb)
            out.append(t / base)
        print(L, f't_hit={base:.1f}', ' '.join(f'{r:.4f}' for r in out), flush=True)

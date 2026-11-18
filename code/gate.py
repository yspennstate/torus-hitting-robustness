"""Upper-bound gate. Q = product chain (outward (1+eps)/2D, inward (1-eps)/2D, symmetric at coordinate 0 or L/2).
B(U)(x) = best response of the eps-adversary = sum lo*U(y) + slack on the top neighbours (= PU + eps/(2D)(top D - bottom D)).
Deficit d = B(V_Q) - Q V_Q >= 0 (zero off the special hyperplanes if V_Q is coordinatewise monotone).
Candidate supersolution U = V_Q + W, W = (I - Q_killed)^{-1} (kappa d).  If 1 + B(U) - U <= 0 off the target, then every
eps-adversary has E_x H_0 <= U(x), so t_hit^+ <= max U.  Print max residual, max W / max V_Q, max U / t_hit.
    python gate.py D kappa L1 L2 ...
"""
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from grid_rho import neighbours, solve


def q_matrix(L, D, eps, nb):
    N = L ** D
    coords = np.array(np.unravel_index(np.arange(N), (L,) * D)).T
    dist = np.minimum(coords, L - coords)
    P = np.zeros((N, 2 * D))
    for a in range(D):
        for j, s in enumerate((1, -1)):
            c = (coords[:, a] + s) % L
            dn = np.minimum(c, L - c)
            p = np.full(N, 1.0 / (2 * D))
            p[(dn > dist[:, a]) & (dist[:, a] != 0)] = (1 + eps) / (2 * D)
            p[(dn < dist[:, a]) & (2 * dist[:, a] != L)] = (1 - eps) / (2 * D)
            P[:, 2 * a + j] = p
    rows = np.repeat(np.arange(N), 2 * D)
    return sp.csr_matrix((P.ravel(), (rows, nb.ravel())), shape=(N, N))


def best_response(U, nb, eps):
    k = nb.shape[1]
    lo = (1 - eps) / k
    vals = -np.sort(-U[nb], axis=1)                    # descending
    return lo * vals.sum(1) + (2 * eps / k) * vals[:, :k // 2].sum(1)


def killed_solve(A, rhs):
    N = A.shape[0]
    keep = np.arange(1, N)
    M = sp.identity(N - 1, format='csr') - A[keep][:, keep]
    x = np.zeros(N); x[keep] = spl.spsolve(M.tocsc(), rhs[keep])
    return x


if __name__ == '__main__':
    D, kappa = int(sys.argv[1]), float(sys.argv[2])
    for L in (int(a) for a in sys.argv[3:]):
        nb = neighbours(L, D)
        t0, _ = solve(L, D, 0.0, nb=nb)
        for c in (0.5, 1.0, 2.0):
            eps = c / L
            Q = q_matrix(L, D, eps, nb)
            V = killed_solve(Q, np.ones(L ** D))
            d = best_response(V, nb, eps) - Q @ V
            d[0] = 0.0
            W = killed_solve(Q, kappa * d)
            U = V + W
            R = 1 + best_response(U, nb, eps) - U
            R[0] = -np.inf
            i = int(np.argmax(R))
            xi = np.unravel_index(i, (L,) * D)
            bad = int((R > 1e-8).sum())
            print(f'   argmax R at {tuple(int(v) for v in xi)}; sites with R>1e-8: {bad}; d there {d[i]:.3e}', flush=True)
            print(f'D={D} L={L} c={c} kappa={kappa}: min d={d[1:].min():.2e} max R={R.max():.3e} '
                  f'W/V={W.max()/V.max():.4f} V/t={V.max()/t0:.5f} U/t={U.max()/t0:.5f}', flush=True)

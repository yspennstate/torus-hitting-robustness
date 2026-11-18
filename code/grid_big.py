"""Larger L for grid_rho.py (D=2, c = 1, 2). Writes grid_big.out."""
from grid_rho import neighbours, solve
with open('grid_big.out', 'a') as f:
    for L in (64, 96, 128):
        nb = neighbours(L, 2)
        base, V0 = solve(L, 2, 0.0, nb=nb)
        r = [solve(L, 2, c / L, V0=V0, nb=nb)[0] / base for c in (1.0, 2.0)]
        print(L, f't_hit={base:.1f}', ' '.join(f'{x:.4f}' for x in r), file=f, flush=True)

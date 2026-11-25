# Hitting-time robustness of the discrete torus

Yitzchak Shmalo, Einstein Institute of Mathematics, The Hebrew University of Jerusalem.

Zuckerman (Markov chain robustness, ITCS 2026, LIPIcs 362, Paper 118, doi:10.4230/LIPIcs.ITCS.2026.118) defined the hitting-time robustness of a random walk. It is the largest eps for which an adversary may change every transition probability by a factor in [1-eps, 1+eps], choosing the change from the whole history, and still not raise the maximum expected hitting time by more than a constant factor. He listed grids among the graphs whose robustness is open.

For the simple random walk on the torus Z_L^D with D >= 3 and eps = c/L, the strongest such adversary raises the maximum expected hitting time by a factor tending to ((e^c - 1)/c)^D. An adversary that pushes every coordinate away from the target independently is optimal up to a factor 1 + o(1). So on these tori the robustness is of order one over the diameter. The upper bound O(1/L) already follows from the adversary constructed in the proof of Zuckerman's Theorem 4.5; the limit constant and the matching lower bound are proved here. Zuckerman conjectured that the robustness of every graph is at least of order one over its diameter (Conjecture 1.16); counterexamples on other graphs have been proposed in doi:10.5281/zenodo.23575572.

## Contents

- `paper/main.tex`, `paper/main.pdf`: the paper.
- `code/grid_rho.py`: the exact worst-case hitting time on Z_L^D by policy iteration, and the ratio to the plain walk (Section 7). `python3 grid_rho.py 2 8 16 32` for D = 2; `python3 grid_rho.py 3 12 16 20 24 32` for D = 3. Needs numpy and scipy.
- `code/grid_big.py`: the D = 2 runs at L = 64 and 128 (the table in Section 7); output `output/grid_big.out`.
- `code/product_check.py`: the value of the product adversary Q against the exact optimum; `PRODONLY=1 python3 product_check.py 3 16 24 32 40` gives the D = 3 values of Q at L = 16 to 40 (output `output/d3_product.out`; with PRODONLY set the exact optimum is not computed, so its columns opt/t and rel.gap read nan). Even L only.
- `code/dmax.py`: the deficit d = BV - QV of Section 2, its maximum and where it is attained; output `output/dmax.out`.
- `code/gate.py`, `code/schur_check.py`, `code/star_range.py`: checks of the comparison steps of Sections 2 to 4 (supersolution gate, the order of the values V(k + e_a), the range of the kernel inequality on the folded chain); outputs `output/gate_*.out`, `output/schur.out`.
- `code/halfline_check.py`: the kernel inequality of Lemma 4.1 checked by Bessel ratios at eps = 0 and on a truncated chain with 600 states for eps in {0.01, 0.05, 0.2}.
- `code/halfline_unif.py`: the same inequality on the same chain with the transition probabilities computed by uniformization, a sum of nonnegative terms, so that no matrix exponential is formed; output `output/halfline_unif.out`.
- `output/d3_small.out`, `output/d3_big.out`: the D = 3 ratios t_hit^+(c/L)/t_hit for c = 0.5, 1, 2, 4. At L = 4 the value c = 4 gives eps = 1, outside the model, and that entry is not used.
- `SHA256SUMS`: sha256 of each file above.

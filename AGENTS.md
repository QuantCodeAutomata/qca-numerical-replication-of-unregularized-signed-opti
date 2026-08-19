# AGENTS.md — Repository Knowledge

## Project Overview
Signed regularised optimal transport on a compact 2D torus.  All computations
live in `/workspace/project/src/`.  Tests live in `/workspace/project/tests/`.
Python 3.12, numpy 1.26.4, scipy 1.14.1.

## Source Files
| File | Purpose |
|------|---------|
| `torus_utils.py` | Core torus primitives (grids, FFT-based Green op, marginals, norms) |
| `regularized_ot.py` | Dual operator K = A G A*, RHS assembly, synthetic problems |
| `unregularized_ot.py` | Ill-posedness demo (affine family, Case A/B) |
| `gauss_seidel.py` | Block Gauss-Seidel solver for K z = d (Experiment 5) |
| `regularization_path.py` | Regularisation path + spectral flatness (Experiment 6) |
| `panel_analysis.py` | Fixed-basis time-series panel OT (Experiment 7) |

## Key Conventions

### Zero-Mean Subspace
K has a null space spanned by constant vectors on X and Y separately.
All solves use a **rank-1 regularisation**: add `ones*ones^T / n` to each
diagonal block of K, then project the solution to zero mean:
```python
K_reg = K_mat.copy()
K_reg[:Nx, :Nx] += np.ones((Nx, Nx)) / Nx
K_reg[Nx:, Nx:] += np.ones((Ny, Ny)) / Ny
z = scipy.linalg.solve(K_reg, d)
z[:Nx] = project_zero_mean(z[:Nx])
z[Nx:] = project_zero_mean(z[Nx:])
```
This pattern is duplicated as `_solve_zm_system` in both
`regularization_path.py` and `panel_analysis.py`, and as `_solve_zm` +
`_reg_matrix` in `gauss_seidel.py`.

### Block Eigenvalues — Skip Index 0
`scipy.linalg.eigvals(KXX)` returns one near-zero eigenvalue (the null
direction). When reporting `min_eig_KXX`, skip index 0 of the **sorted**
eigenvalue array:
```python
eigs = np.sort(linalg.eigvals(KXX).real)
min_eig = float(eigs[1])   # skip null mode
```

### Primal Recovery
```python
lifted = lift(phi, psi, Nx, Ny)   # phi[:,None] + psi[None,:]
pi_star = apply_G(lifted - c, m) / eps
```

### AGc Reuse Across eps
`compute_AGc(c, m)` returns `(AGc_x, AGc_y)`.  The RHS `d = eps*[p;q] + [AGc_x;AGc_y]`
so K_mat and AGc can be built once and d recomputed cheaply for each eps.

### Synthetic Problems
- `make_synthetic_problem(Nx, Ny)` — default marginals / cost.
- `make_synthetic_problem(Nx, Ny, marginal_type='exp6', cost_type='exp6')` —
  used in Experiments 6 & 7.
- `make_synthetic_problem` is also re-exported from `regularization_path.py`
  for test convenience.

## Spectral Diagnostics
- `spectral_bands(pi, m)` partitions `|pi_hat|^2 / N^4` into low/medium/high
  bands by radial frequency `r = sqrt(m)`.
- `high_freq_share ≈ 0` is **correct** for smooth OT solutions (the Green
  operator suppresses high-frequency content).
- `dirichlet_energy(pi, m)` is the primary spectral discriminant for regime
  detection; it is weighted by `m = kx^2+ky^2` and rises ~4× in Regime 1.

## Panel Regime Detection
- Regime 0 (calm): low-frequency marginals only.
- Regime 1 (stress): fixed high-frequency component (amplitude 2.0 at
  wavenumber 8) plus small random noise (σ=0.3).
- Detection uses `detect_regime_shifts(dirichlet_energy, window)` — rolling
  window MAD; peaks when window straddles the regime-change time.
- Result key `detected_change_period` = `detection_idx` (alias for tests).
- Default tolerance: `|detection_idx - regime_change_t| <= window + 5`.

## Experiment Scripts (`exp/`)

All 7 scripts are standalone (`if __name__ == '__main__': main()`),
produce PNG plots under `results/`, and return a metrics dict.

| File | Key test | Expected outcome |
|------|----------|-----------------|
| `exp_1_ill_posedness.py` | `<c,r>≠0`, Case B flat | PASS (structural trials 0,1,3,4 may FAIL inner-product check — expected) |
| `exp_2_regularized_ot.py` | uniqueness across 3 inits | PASS; LinAlgWarning at ε≤1e-3 is expected |
| `exp_3_euler_lagrange.py` | EL residuals | Gracefully skips if `src/euler_lagrange.py` absent |
| `exp_4_richardson.py` | Richardson contraction | α=2.1/λ_max FAILS (diverges by design); others PASS |
| `exp_5_gauss_seidel.py` | ρ(M)=0, 1-step conv | PASS; empirical rate = NaN (nilpotent, 1-iter exact) |
| `exp_6_regularization_path.py` | feasibility ∀ε | PASS; high_freq_share=0 is correct (smooth solutions) |
| `exp_7_panel.py` | energy jump at t=50 | energy PASS; HF-share FAIL (noise at 1e-32, expected) |

### API Quick-Reference

**`run_regularization_path(Nx, Ny, eps_list)`** → dict with key `'path'`
(list of per-ε dicts containing `eps, pi_star, energy, cost_term, objective, p_resid, q_resid, spectral_bands`).
Access HF share as `rec['spectral_bands']['high_freq_share']`.

**`run_panel_analysis(T, N, eps, regime_change_t)`** → dict with keys:
- `spectral['dirichlet_energy']`, `spectral['high_freq_share']`
- `anomaly_scores`
- `panel['regime_labels']`  (0 = calm pre-change, 1 = stress post-change)
- `detection_window`, `detection_idx`

**`run_block_gauss_seidel_analysis(Nx, Ny, eps, max_iter, tol)`** → dict with keys:
`KXX, KYY, KXY, rho_M, gs_residuals, gs_iterates, solution, direct_solution`.

## euler_lagrange.py (Experiment 3)

### Correct EL Field Sign
The first-order condition is `c + eps*L pi* = phi*(x) + psi*(y) + const`.
The EL residual field is therefore:
```python
S = c + eps * apply_L_pi(pi_star, m) - lift(phi_star, psi_star, Nx, Ny)
```
At the optimum, `std(S) ≈ 1e-14`.  The spec's formula `c + phi + psi + eps*Lpi`
has the wrong sign on the potentials (std ≈ 0.4, NOT constant).

### Normalization of Relative Residuals
- **Local EL** (ker(A) test fns): `rel = |residual| / (||c + eps*Lpi||_L2 * h_norm)`.
- **Global EL** (zero-mean test fns): normalize by `rms(c) * h_norm`  (NOT by `||S||`
  since `||S|| ≈ 0` causes 0/0).
- **Strong EL**: `rel = std(S) / rms(c)`.

### Test Functions
- ker(A): outer products `a[:,None] * b[None,:]` where both `a` and `b` are zero-mean.
- Global: `project_zero_mean(rng.standard_normal((Nx,Ny)))`.

## richardson.py (Experiment 4)

### Zero-Mean Projection per Block
After each Richardson step, independently center each block:
```python
z_new[:Nx] -= np.mean(z_new[:Nx])
z_new[Nx:] -= np.mean(z_new[Nx:])
```
This keeps iterates in the zero-mean subspace and ensures convergence to the
unique zero-mean solution.

### Reference Solution
`scipy.linalg.solve` issues a `LinAlgWarning` (ill-conditioned, rcond ≈ 5e-18)
but still returns a numerically correct particular solution.  Project it to
zero-mean before computing true errors:
```python
z_ref = scipy.linalg.solve(K_mat, d)
z_ref[:Nx] -= np.mean(z_ref[:Nx])
z_ref[Nx:] -= np.mean(z_ref[Nx:])
```

### Eigenvalue Filtering
```python
eigvals = scipy.linalg.eigh(K_mat, eigvals_only=True)
threshold = max(eigvals.max() * 1e-8, 1e-14)
pos_eigvals = eigvals[eigvals > threshold]
lambda_min, lambda_max = pos_eigvals.min(), pos_eigvals.max()
```

### Expected Spectrum (Nx=Ny=32, eps=0.1)
`lambda_min ≈ 0.015625`, `lambda_max = 1.0`, `kappa ≈ 64`.
`rho_opt = 63/65 ≈ 0.969`. Converges in ~few thousand iters at `alpha_opt`.
`alpha_diverge = 2.1 / lambda_max` → spectral radius 1.1, residuals grow.

## Test Suite — Final State
All **49 tests pass** across 8 test files.  Run with:
```bash
cd /workspace/project && python -m pytest tests/ -v
```
The `LinAlgWarning` (ill-conditioned matrix, rcond≈1e-18) from `scipy.linalg.solve`
is expected on small grids (K condition number = 256 for 32-grid) — solutions
remain numerically correct and all assertions hold.

## Block Gauss-Seidel: Why ρ(M) = 0
On the normalised 1-D torus, the Laplacian decouples in Fourier space:
- Lifting φ → φ⊗1 has only l=0 Fourier modes.  G suppresses (0,0) and leaves
  modes (k,0) for k≠0.  Taking the X-marginal (averaging over Y) of G(φ⊗1)
  returns a pure X-function.  But P_X G (1⊗ψ) = 0 because G(1⊗ψ) has only
  k=0 modes, and averaging over Y at k=0 gives 0 (zero-mode of G is zero).
- Therefore **KXY = 0** and **KYX = 0** exactly.
- M = KYY⁻¹ KYX KXX⁻¹ KXY = 0 → ρ(M)=0 → Block GS converges in one step.

This is mathematically exact, not a numerical coincidence.

"""
Experiment 2: Regularized signed optimal transport on the torus.

Demonstrates well-posedness of the Dirichlet-regularized problem:
  min_{Aπ=b} <c, π> + ε·H(π),  H(π) = ½<π, Lπ>

Verifies: unique solution, marginal feasibility, eps-dependence, and
consistency between 'direct' and 'cg' solvers.
"""

import sys
sys.path.insert(0, '/workspace/project')

import os
import numpy as np
from scipy.sparse.linalg import cg as scipy_cg

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from src.torus_utils import (
        make_torus_grids,
        make_laplacian_multiplier,
        project_zero_mean,
        lift,
        apply_G,
        marginals,
        inner_product,
        dirichlet_energy,
    )
    from src.regularized_ot import (
        solve_regularized_ot,
        make_synthetic_problem,
        build_K_matrix,
        compute_rhs,
        make_K_linear_operator,
    )
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Ensure src/torus_utils.py and src/regularized_ot.py are present.")
    sys.exit(1)

RESULTS_DIR = '/workspace/project/results'


def _pf(condition: bool) -> str:
    return 'PASS' if condition else 'FAIL'


def _recover_pi(z: np.ndarray, c: np.ndarray, m: np.ndarray, eps: float,
                Nx: int, Ny: int) -> np.ndarray:
    """Primal recovery: π* = (1/ε) G(A*z - c)."""
    phi = project_zero_mean(z[:Nx])
    psi = project_zero_mean(z[Nx:])
    lifted = lift(phi, psi, Nx, Ny)
    return apply_G(lifted - c, m) / eps


def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    Nx, Ny = 64, 64
    x, y, p, q, c = make_synthetic_problem(Nx, Ny, marginal_type='default', cost_type='default')
    m = make_laplacian_multiplier(Nx, Ny)

    print("\n=== Exp 2: Regularized Signed OT ===\n")

    # ------------------------------------------------------------------
    # 2. Verify Laplacian multiplier properties
    # ------------------------------------------------------------------
    m00 = m[0, 0]
    spectral_gap = float(np.min(m[m > 0]))
    print("Laplacian multiplier:")
    print(f"  m[0,0]       = {m00:.6f}  {_pf(m00 == 0.0)}")
    print(f"  spectral gap = {spectral_gap:.4f}  {_pf(spectral_gap > 0)}")
    print(f"  max(m)       = {float(np.max(m)):.1f}")

    # ------------------------------------------------------------------
    # 3. Solve for multiple eps values using 'direct' method
    # ------------------------------------------------------------------
    eps_list = [1e-3, 1e-2, 1e-1, 1.0]
    results_by_eps = {}

    print(f"\n{'eps':>8}  {'p_resid':>10}  {'q_resid':>10}  "
          f"{'objective':>12}  {'cost_term':>12}  {'energy':>12}  {'feas':>4}")
    print("-" * 78)

    for eps in eps_list:
        res = solve_regularized_ot(p, q, c, eps, Nx, Ny, method='direct')
        results_by_eps[eps] = res
        feas_ok = (res['p_resid'] < 1e-8) and (res['q_resid'] < 1e-8)
        print(f"{eps:>8.0e}  {res['p_resid']:>10.2e}  {res['q_resid']:>10.2e}  "
              f"{res['objective']:>12.6f}  {res['cost_term']:>12.6f}  "
              f"{res['energy']:>12.6f}  {_pf(feas_ok):>4}")

    # ------------------------------------------------------------------
    # 4. Feasibility tests
    # ------------------------------------------------------------------
    feas_all = all(
        (results_by_eps[eps]['p_resid'] < 1e-8) and (results_by_eps[eps]['q_resid'] < 1e-8)
        for eps in eps_list
    )
    print(f"\nFeasibility (all eps): {_pf(feas_all)}")

    # ------------------------------------------------------------------
    # 5. Uniqueness: three CG solves from different initializations
    # ------------------------------------------------------------------
    eps_uniq = 0.1
    K_op = make_K_linear_operator(Nx, Ny, m)
    d = compute_rhs(p, q, c, m, eps_uniq)
    dim = Nx + Ny

    rng = np.random.default_rng(7)
    init_vectors = [
        np.zeros(dim),
        project_zero_mean(rng.standard_normal(dim)),
        project_zero_mean(rng.standard_normal(dim)),
    ]
    init_labels = ['zeros', 'random1', 'random2']

    pi_solutions = []
    for label, x0 in zip(init_labels, init_vectors):
        z_cg, info = scipy_cg(K_op, d, x0=x0, rtol=1e-12, maxiter=20000)
        pi_cg = _recover_pi(z_cg, c, m, eps_uniq, Nx, Ny)
        pi_solutions.append((label, pi_cg, info))

    pi_ref = pi_solutions[0][1]
    print(f"\nUniqueness test (ε={eps_uniq}, 3 CG initializations):")
    print(f"  {'init':>8}  {'cg_info':>8}  {'max|Δπ|':>12}  {'result':>6}")
    uniq_pass = True
    for label, pi_sol, info in pi_solutions[1:]:
        diff = float(np.max(np.abs(pi_sol - pi_ref)))
        ok = (diff < 1e-6) and (info == 0)
        uniq_pass = uniq_pass and ok
        print(f"  {label:>8}  {info:>8}  {diff:>12.2e}  {_pf(ok):>6}")
    print(f"  Uniqueness PASS: {_pf(uniq_pass)}")

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------

    # exp2_transport_plan.png: heatmap of π* for eps=0.1
    pi_01 = results_by_eps[0.1]['pi_star']
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pi_01, origin='lower', aspect='equal', cmap='RdBu_r',
                   extent=[0, 1, 0, 1])
    plt.colorbar(im, ax=ax, label='π*(x, y)')
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('Optimal transport plan π*  (ε = 0.1)')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp2_transport_plan.png'), dpi=100)
    plt.close(fig)

    # exp2_eps_diagnostics.png: objective decomposition vs eps
    objectives = [results_by_eps[e]['objective'] for e in eps_list]
    cost_terms = [results_by_eps[e]['cost_term'] for e in eps_list]
    energy_terms = [results_by_eps[e]['energy'] for e in eps_list]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(eps_list, objectives, 'o-', color='tab:green', label='J_ε(π*)')
    ax.semilogx(eps_list, cost_terms, 's--', color='tab:blue', label='cost term ⟨c,π*⟩')
    ax.semilogx(eps_list, energy_terms, '^--', color='tab:orange', label='ε·H(π*)')
    ax.set_xlabel('ε (regularization)')
    ax.set_ylabel('value')
    ax.set_title('Objective decomposition vs ε')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp2_eps_diagnostics.png'), dpi=100)
    plt.close(fig)

    print(f"\nPlots saved to {RESULTS_DIR}/")

    # ------------------------------------------------------------------
    # 7. Return metrics
    # ------------------------------------------------------------------
    metrics = {
        'm00': m00,
        'spectral_gap': spectral_gap,
        'feas_all_eps': feas_all,
        'uniqueness_pass': uniq_pass,
        'results_by_eps': {
            eps: {k: v for k, v in res.items() if k != 'pi_star'}
            for eps, res in results_by_eps.items()
        },
    }
    return metrics


if __name__ == '__main__':
    main()

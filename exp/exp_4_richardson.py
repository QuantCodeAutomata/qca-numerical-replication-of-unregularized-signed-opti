"""
Experiment 4: Richardson iteration for the dual operator K.

Analyzes convergence of stationary Richardson iterations  z_{k+1} = z_k + α(d - Kz_k)
as a function of step size α relative to the spectrum of K.  Four step sizes are
tested: the optimal α*, 1/λ_max (safe), 1.9/λ_max (near-critical), and
2.1/λ_max (super-critical / diverging).
"""

import sys
sys.path.insert(0, '/workspace/project')

import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from src.torus_utils import (
        make_torus_grids,
        make_laplacian_multiplier,
        project_zero_mean,
    )
    from src.regularized_ot import (
        make_synthetic_problem,
        build_K_matrix,
        compute_rhs,
    )
except ImportError as e:
    print(f"[ERROR] Core import failed: {e}")
    sys.exit(1)

_RICHARDSON_AVAILABLE = True
_RICHARDSON_ERROR = ''
try:
    from src.richardson import run_richardson_analysis, richardson_iteration
except ImportError as e:
    _RICHARDSON_AVAILABLE = False
    _RICHARDSON_ERROR = str(e)

RESULTS_DIR = '/workspace/project/results'


def _pf(condition: bool) -> str:
    return 'PASS' if condition else 'FAIL'


def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n=== Exp 4: Richardson Iteration Analysis ===\n")

    # ------------------------------------------------------------------
    # 1. Setup (always runs, used for spectral plot)
    # ------------------------------------------------------------------
    Nx, Ny = 32, 32
    eps = 0.1
    x, y, p, q, c = make_synthetic_problem(Nx, Ny, marginal_type='default', cost_type='default')
    m = make_laplacian_multiplier(Nx, Ny)

    # Assemble K explicitly (dim = 64x64 for Nx=Ny=32)
    K_mat = build_K_matrix(Nx, Ny, m)
    d_vec = compute_rhs(p, q, c, m, eps)

    # Eigenvalues of K (restrict to nonzero modes via SVD for stability)
    eigvals = np.linalg.eigvalsh(K_mat)
    eigvals_pos = eigvals[eigvals > 1e-14]
    lambda_min = float(np.min(eigvals_pos))
    lambda_max = float(np.max(eigvals_pos))
    kappa = lambda_max / lambda_min

    print(f"K matrix shape: {K_mat.shape}")
    print(f"λ_min (positive) = {lambda_min:.6e}")
    print(f"λ_max            = {lambda_max:.6e}")
    print(f"Condition number κ = {kappa:.4f}")

    # Step sizes
    alpha_opt = 2.0 / (lambda_min + lambda_max)
    alpha_safe = 1.0 / lambda_max
    alpha_near = 1.9 / lambda_max
    alpha_super = 2.1 / lambda_max
    alpha_values = [alpha_opt, alpha_safe, alpha_near, alpha_super]
    alpha_labels = ['α_opt', '1/λ_max', '1.9/λ_max', '2.1/λ_max (diverge)']

    # Theoretical contraction factors
    def contraction(alpha: float) -> float:
        return max(abs(1.0 - alpha * lambda_min), abs(1.0 - alpha * lambda_max))

    rho_opt = contraction(alpha_opt)
    rho_safe = contraction(alpha_safe)
    rho_near = contraction(alpha_near)
    rho_super = contraction(alpha_super)
    rho_values = [rho_opt, rho_safe, rho_near, rho_super]

    print(f"\nTheoretical contraction factors ρ(α):")
    for label, alpha, rho in zip(alpha_labels, alpha_values, rho_values):
        converges = rho < 1.0
        print(f"  {label:22s}: α={alpha:.4e}  ρ={rho:.6f}  {_pf(converges)}")

    # ------------------------------------------------------------------
    # 2. exp4_spectral.png: eigenvalue distribution (always available)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(sorted(eigvals), 'o', markersize=3, color='tab:blue')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Index (sorted)')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalues of K  (Nx=Ny=32)')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(range(len(eigvals_pos)), sorted(eigvals_pos), 'o', markersize=3,
                color='tab:green')
    ax.axhline(lambda_min, color='tab:red', linestyle='--', label=f'λ_min={lambda_min:.2e}')
    ax.axhline(lambda_max, color='tab:orange', linestyle='--', label=f'λ_max={lambda_max:.2e}')
    ax.set_xlabel('Index (sorted positive)')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title(f'Positive eigenvalues of K  (κ={kappa:.1f})')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp4_spectral.png'), dpi=100)
    plt.close(fig)
    print(f"\nexp4_spectral.png saved.")

    # ------------------------------------------------------------------
    # 3. Richardson convergence analysis
    # ------------------------------------------------------------------
    if not _RICHARDSON_AVAILABLE:
        print(f"\n[WARNING] src/richardson.py not available: {_RICHARDSON_ERROR}")
        print("Skipping Richardson convergence analysis.  exp4_convergence.png not generated.")
        return {
            'richardson_available': False,
            'error': _RICHARDSON_ERROR,
            'lambda_min': lambda_min,
            'lambda_max': lambda_max,
            'kappa': kappa,
            'rho_opt': rho_opt,
        }

    # Compute reference solution for true-error tracking
    z_ref = np.linalg.solve(K_mat, d_vec)
    z_ref[:Nx] -= np.mean(z_ref[:Nx])
    z_ref[Nx:] -= np.mean(z_ref[Nx:])
    z0 = np.zeros_like(d_vec)

    # Run richardson_iteration for each step size
    rich_results = [
        richardson_iteration(K_mat, d_vec, alpha, z0, max_iter=5000, tol=1e-8,
                             z_ref=z_ref, Nx=Nx)
        for alpha in alpha_values
    ]

    # ------------------------------------------------------------------
    # 4. Report theoretical vs empirical contraction
    # ------------------------------------------------------------------
    conv_tol = 1e-8
    print(f"\n{'Step size':22s}  {'ρ_theory':>10}  {'ρ_empirical':>12}  "
          f"{'iters':>8}  {'final_resid':>12}  {'result':>6}")
    print("-" * 80)

    all_conv_pass = True
    conv_metrics = {}
    for label, alpha, rho_th, run in zip(
        alpha_labels, alpha_values, rho_values, rich_results
    ):
        rho_emp = run.get('empirical_contraction', float('nan'))
        iters   = run.get('n_iter', -1)            # key from richardson_iteration
        final_r = run.get('final_residual', float('nan'))
        converged = run.get('converged', False)
        expected_conv = rho_th < 1.0
        test_ok = (converged == expected_conv)
        all_conv_pass = all_conv_pass and test_ok
        print(f"  {label:22s}  {rho_th:>10.6f}  {rho_emp:>12.6f}  "
              f"{iters:>8}  {final_r:>12.2e}  {_pf(test_ok):>6}")
        conv_metrics[label] = run

    print(f"\nAll convergence tests PASS: {_pf(all_conv_pass)}")

    # ------------------------------------------------------------------
    # 5. exp4_convergence.png
    # ------------------------------------------------------------------
    colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:red']
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, run, color in zip(alpha_labels, rich_results, colors):
        residuals = run.get('residuals', [])        # key from richardson_iteration
        if len(residuals) > 0:
            ax.semilogy(residuals, label=label, color=color, linewidth=1.5)
    ax.axhline(conv_tol, color='grey', linestyle=':', linewidth=1, label=f'tol={conv_tol:.0e}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('‖Kz - d‖ (residual)')
    ax.set_title('Richardson iteration convergence  (Nx=Ny=32)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp4_convergence.png'), dpi=100)
    plt.close(fig)

    print(f"\nPlots saved to {RESULTS_DIR}/")

    # ------------------------------------------------------------------
    # 6. Return metrics
    # ------------------------------------------------------------------
    metrics = {
        'richardson_available': True,
        'lambda_min': lambda_min,
        'lambda_max': lambda_max,
        'kappa': kappa,
        'rho_opt': rho_opt,
        'rho_safe': rho_safe,
        'rho_near': rho_near,
        'rho_super': rho_super,
        'all_conv_pass': all_conv_pass,
        'conv_metrics': conv_metrics,
    }
    return metrics


if __name__ == '__main__':
    main()

"""
Experiment 3: Euler-Lagrange conditions for regularized signed OT.

Verifies that the optimal π* satisfies the Euler-Lagrange (first-order
optimality) conditions both weakly (against test functions) and strongly
(pointwise), and that (π*, z*) forms a saddle point of the Lagrangian.
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
        inner_product,
    )
    from src.regularized_ot import make_synthetic_problem, solve_regularized_ot
except ImportError as e:
    print(f"[ERROR] Core import failed: {e}")
    sys.exit(1)

_EL_AVAILABLE = True
_EL_ERROR = ''
try:
    from src.euler_lagrange import (
        run_el_verification,
        local_el_residual,
        global_el_residual,
        strong_el_residual,
        lagrangian,
        saddle_point_check,
    )
except ImportError as e:
    _EL_AVAILABLE = False
    _EL_ERROR = str(e)

RESULTS_DIR = '/workspace/project/results'


def _pf(condition: bool) -> str:
    return 'PASS' if condition else 'FAIL'


def _make_kernel_test_functions(x: np.ndarray, y: np.ndarray, n: int,
                                 rng: np.random.Generator) -> list:
    """Generate n functions in ker(A): outer products of zero-mean sinusoids."""
    test_fns = []
    for _ in range(n):
        kx = rng.integers(1, 6)
        ky = rng.integers(1, 6)
        amp = rng.uniform(0.5, 1.5)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        a = project_zero_mean(np.sin(2 * np.pi * kx * x + phase_x))
        b = project_zero_mean(np.cos(2 * np.pi * ky * y + phase_y))
        test_fns.append(amp * np.outer(a, b))
    return test_fns


def _make_zero_mean_test_functions(x: np.ndarray, y: np.ndarray, n: int,
                                    rng: np.random.Generator) -> list:
    """Generate n arbitrary zero-mean functions on T^2."""
    Nx, Ny = len(x), len(y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    test_fns = []
    for _ in range(n):
        f = np.zeros((Nx, Ny))
        for _ in range(3):
            kx = rng.integers(1, 8)
            ky = rng.integers(1, 8)
            amp = rng.uniform(-1, 1)
            f += amp * np.sin(2 * np.pi * kx * X + rng.uniform(0, 2 * np.pi)) * \
                       np.cos(2 * np.pi * ky * Y + rng.uniform(0, 2 * np.pi))
        test_fns.append(project_zero_mean(f))
    return test_fns


def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n=== Exp 3: Euler-Lagrange Conditions ===\n")

    if not _EL_AVAILABLE:
        print(f"[WARNING] src/euler_lagrange.py not available: {_EL_ERROR}")
        print("Skipping EL-specific tests.  Run after implementing src/euler_lagrange.py.")
        return {'el_available': False, 'error': _EL_ERROR}

    # ------------------------------------------------------------------
    # 1. Setup: build problem and solve regularized OT
    # ------------------------------------------------------------------
    Nx, Ny = 64, 64
    eps = 0.1
    x, y, p, q, c = make_synthetic_problem(Nx, Ny, marginal_type='exp3', cost_type='exp3')
    m = make_laplacian_multiplier(Nx, Ny)

    sol = solve_regularized_ot(p, q, c, eps, Nx, Ny, method='direct')
    pi_star  = sol['pi_star']
    phi_star = sol['phi_star']   # (Nx,)
    psi_star = sol['psi_star']   # (Ny,)

    # ------------------------------------------------------------------
    # 2. run_el_verification: full self-contained verification
    # ------------------------------------------------------------------
    el_full = run_el_verification(Nx=Nx, Ny=Ny, eps=eps)
    print("Full EL verification (run_el_verification):")
    for key, val in el_full.items():
        if isinstance(val, float):
            print(f"  {key:30s}: {val:.4e}")
        elif isinstance(val, bool):
            print(f"  {key:30s}: {_pf(val)}")

    # ------------------------------------------------------------------
    # 3. Local weak EL: 10 test functions in ker(A)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(2024)
    kernel_fns = _make_kernel_test_functions(x, y, 10, rng)
    # local_el_residual(pi_star, c, m, eps, test_functions) -> list of dicts
    local_results = local_el_residual(pi_star, c, m, eps, kernel_fns)
    local_resids  = [d['residual'] for d in local_results]
    local_pass    = all(abs(r) < 1e-8 for r in local_resids)
    print(f"\nLocal weak EL (10 ker(A) test fns):")
    print(f"  max |residual| = {max(abs(r) for r in local_resids):.4e}  {_pf(local_pass)}")

    # ------------------------------------------------------------------
    # 4. Global weak EL: 10 arbitrary zero-mean test functions
    # ------------------------------------------------------------------
    zmean_fns = _make_zero_mean_test_functions(x, y, 10, rng)
    # global_el_residual(pi_star, phi_star, psi_star, c, m, eps, test_functions)
    global_results = global_el_residual(pi_star, phi_star, psi_star, c, m, eps, zmean_fns)
    global_resids  = [d['residual'] for d in global_results]
    global_pass    = all(abs(r) < 1e-8 for r in global_resids)
    print(f"\nGlobal weak EL (10 zero-mean test fns):")
    print(f"  max |residual| = {max(abs(r) for r in global_resids):.4e}  {_pf(global_pass)}")

    # ------------------------------------------------------------------
    # 5. Strong EL condition
    # ------------------------------------------------------------------
    # strong_el_residual(pi_star, phi_star, psi_star, c, m, eps, Nx, Ny) -> dict
    strong_dict  = strong_el_residual(pi_star, phi_star, psi_star, c, m, eps, Nx, Ny)
    strong_resid = strong_dict['rel_residual']
    strong_pass  = strong_resid < 1e-6
    print(f"\nStrong EL rel. residual: {strong_resid:.4e}  {_pf(strong_pass)}")

    # ------------------------------------------------------------------
    # 6. Saddle-point check
    # ------------------------------------------------------------------
    # saddle_point_check(pi_star, phi_star, psi_star, c, p, q, m, eps, Nx, Ny)
    sp_results    = saddle_point_check(pi_star, phi_star, psi_star, c, p, q, m, eps, Nx, Ny,
                                        n_random=20, delta=0.01, rng=rng)
    duality_gap   = sp_results['duality_gap']
    # Verify saddle property: primal L values >= saddle_value
    saddle_val    = sp_results['saddle_value']
    primal_gaps   = [lv - saddle_val for lv in sp_results['primal_perturbations']]
    min_primal_gap = min(primal_gaps)
    # Dual perturbations change L by feasibility residual ≈ 0
    dual_gaps     = [abs(lv - saddle_val) for lv in sp_results['dual_perturbations']]
    max_dual_gap  = max(dual_gaps)
    sp_pass       = (duality_gap < 1e-10) and (min_primal_gap >= -1e-8) and (max_dual_gap < 1e-8)
    print(f"\nSaddle-point check (20 perturbations):")
    print(f"  duality gap      : {duality_gap:.4e}")
    print(f"  min primal gap   : {min_primal_gap:.4e}")
    print(f"  max dual gap     : {max_dual_gap:.4e}  {_pf(sp_pass)}")

    # ------------------------------------------------------------------
    # 7. Sensitivity over eps
    # ------------------------------------------------------------------
    eps_list = [1e-3, 1e-2, 1e-1, 1.0]
    eps_metrics = []
    for e in eps_list:
        s = solve_regularized_ot(p, q, c, e, Nx, Ny, method='direct')
        sd = strong_el_residual(s['pi_star'], s['phi_star'], s['psi_star'],
                                c, m, e, Nx, Ny)
        eps_metrics.append({'eps': e, 'strong_rel': sd['rel_residual'],
                            'obj': s['objective']})

    print(f"\n{'eps':>8}  {'strong_rel':>12}  {'objective':>12}")
    print("-" * 38)
    for em in eps_metrics:
        print(f"  {em['eps']:>6.0e}  {em['strong_rel']:>12.4e}  {em['obj']:>12.4f}")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    from matplotlib.patches import Patch

    # 8a. EL residual bar chart
    labels_local  = [f'ker#{i+1}' for i in range(len(local_resids))]
    labels_global = [f'zm#{i+1}'  for i in range(len(global_resids))]
    all_labels    = labels_local + labels_global
    all_resids    = [abs(r) for r in local_resids] + [abs(r) for r in global_resids]
    colors        = ['tab:blue'] * len(local_resids) + ['tab:orange'] * len(global_resids)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(all_labels)), all_resids, color=colors)
    ax.axhline(1e-8, color='red', linestyle='--', linewidth=1, label='tol = 1e-8')
    ax.set_yscale('log')
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('|EL residual|')
    ax.set_title('Euler-Lagrange residuals  (ε = 0.1, exp3 problem)')
    legend_elements = [Patch(facecolor='tab:blue',   label='ker(A) test fns'),
                       Patch(facecolor='tab:orange', label='zero-mean test fns')]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp3_el_residuals.png'), dpi=100)
    plt.close(fig)

    # 8b. Strong EL map
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    strong_S = strong_dict['S']
    im0 = axes[0].imshow(strong_S, origin='lower', cmap='RdBu_r', aspect='auto')
    axes[0].set_title(r'$S = c + \varepsilon L\pi^* - \phi^* - \psi^*$')
    axes[0].set_xlabel('y'); axes[0].set_ylabel('x')
    plt.colorbar(im0, ax=axes[0])
    S_centered = strong_S - strong_dict['C']
    im1 = axes[1].imshow(S_centered, origin='lower', cmap='RdBu_r', aspect='auto')
    axes[1].set_title(r'$S - C$ (should be $\approx 0$)')
    axes[1].set_xlabel('y'); axes[1].set_ylabel('x')
    plt.colorbar(im1, ax=axes[1])
    fig.suptitle(f'Strong EL field  (ε = 0.1)  rel.residual = {strong_resid:.2e}')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp3_strong_el_field.png'), dpi=100)
    plt.close(fig)

    print(f"\nPlots saved to {RESULTS_DIR}/")

    # ------------------------------------------------------------------
    # 9. Return metrics
    # ------------------------------------------------------------------
    metrics = {
        'el_available'    : True,
        'local_pass'      : local_pass,
        'global_pass'     : global_pass,
        'strong_pass'     : strong_pass,
        'saddle_point_pass': sp_pass,
        'max_local_resid' : max(abs(r) for r in local_resids),
        'max_global_resid': max(abs(r) for r in global_resids),
        'strong_resid'    : strong_resid,
        'duality_gap'     : duality_gap,
        'min_primal_gap'  : min_primal_gap,
        'el_full'         : el_full,
    }
    return metrics


if __name__ == '__main__':
    main()

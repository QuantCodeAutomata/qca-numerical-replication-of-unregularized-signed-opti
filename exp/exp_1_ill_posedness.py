"""
Experiment 1: Ill-posedness of unregularized signed OT.

Demonstrates that removing the nonnegativity constraint makes the transport
problem ill-posed: there exist feasible affine families along which the
unregularized objective is either unbounded (Case A) or flat (Case B).
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
        project_zero_mean,
        check_feasibility,
        inner_product,
    )
    from src.unregularized_ot import (
        construct_baseline_coupling,
        verify_zero_marginals,
        run_case_A,
        run_case_B,
        run_structural_verification,
        objective_along_path,
    )
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Ensure src/torus_utils.py and src/unregularized_ot.py are present.")
    sys.exit(1)

RESULTS_DIR = '/workspace/project/results'


def _pf(condition: bool) -> str:
    return 'PASS' if condition else 'FAIL'


def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Torus setup
    # ------------------------------------------------------------------
    Nx, Ny = 64, 64
    x, y = make_torus_grids(Nx, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    p = project_zero_mean(np.sin(2 * np.pi * x))
    q = project_zero_mean(np.cos(2 * np.pi * y))
    # Include sin(2πx)sin(2πy) term so that <c, r> ≠ 0 for Case A.
    c = (np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
         + 0.3 * np.cos(4 * np.pi * (X - Y))
         + 0.5 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y))

    # Zero-marginal perturbation r(x,y) = sin(2πx)·sin(2πy)
    a = project_zero_mean(np.sin(2 * np.pi * x))
    b = project_zero_mean(np.sin(2 * np.pi * y))
    r = np.outer(a, b)

    t_values = np.linspace(-100.0, 100.0, 201)

    # ------------------------------------------------------------------
    # 2. Verify zero marginals of r
    # ------------------------------------------------------------------
    zm = verify_zero_marginals(r, tol=1e-12)
    print("\n=== Exp 1: Ill-posedness of Unregularized Signed OT ===\n")
    print(f"Zero-marginal check on r:")
    print(f"  x-residual : {zm['x_resid']:.2e}  {_pf(zm['x_resid'] < 1e-12)}")
    print(f"  y-residual : {zm['y_resid']:.2e}  {_pf(zm['y_resid'] < 1e-12)}")

    # ------------------------------------------------------------------
    # 3. Verify feasibility of pi0
    # ------------------------------------------------------------------
    pi0 = construct_baseline_coupling(p, q)
    feas = check_feasibility(pi0, p, q, tol=1e-10)
    print(f"\nFeasibility of pi0:")
    print(f"  p_resid    : {feas['p_resid']:.2e}  {_pf(feas['p_resid'] < 1e-10)}")
    print(f"  q_resid    : {feas['q_resid']:.2e}  {_pf(feas['q_resid'] < 1e-10)}")
    print(f"  mass       : {feas['mass']:.2e}")
    print(f"  feasible   : {feas['feasible']}  {_pf(feas['feasible'])}")

    # ------------------------------------------------------------------
    # 4. Case A: c unchanged → <c,r> ≠ 0 → unbounded
    # ------------------------------------------------------------------
    res_A = run_case_A(p, q, c, r, t_values)
    slope_A = res_A['slope']
    J0_range_A = float(np.max(res_A['J0_values']) - np.min(res_A['J0_values']))
    print(f"\nCase A (c unchanged):")
    print(f"  <c, r>     : {slope_A:.6f}  {_pf(abs(slope_A) > 1e-10)}")
    print(f"  J0 range   : {J0_range_A:.4f}  {_pf(J0_range_A > 1.0)}")
    print(f"  unbounded  : {_pf(res_A['passes_A'])}")

    # ------------------------------------------------------------------
    # 5. Case B: project c to be orthogonal to r → constant objective
    # ------------------------------------------------------------------
    res_B = run_case_B(p, q, c, r, t_values)
    slope_B = res_B['slope_B']
    variation_B = res_B['obj_variation']
    print(f"\nCase B (c projected ⊥ r):")
    print(f"  <c_B, r>   : {slope_B:.2e}  {_pf(abs(slope_B) < 1e-10)}")
    print(f"  J0 variation: {variation_B:.2e}  {_pf(variation_B < 1e-10)}")
    print(f"  non-unique : {_pf(res_B['passes_B'])}")

    # ------------------------------------------------------------------
    # 6. Structural verification: 5 independent trials
    # ------------------------------------------------------------------
    struct_results = run_structural_verification(Nx=Nx, Ny=Ny, num_trials=5)

    print(f"\n{'Trial':>5}  {'slope_A':>12}  {'var_B':>12}  {'Case A':>6}  {'Case B':>6}")
    print("-" * 55)
    all_A_pass = True
    all_B_pass = True
    for r_dict in struct_results:
        trial = r_dict['trial']
        s_A = r_dict['case_A']['slope']
        v_B = r_dict['case_B']['obj_variation']
        pa = r_dict['passes_A']
        pb = r_dict['passes_B']
        all_A_pass = all_A_pass and pa
        all_B_pass = all_B_pass and pb
        print(f"{trial:>5}  {s_A:>12.6f}  {v_B:>12.2e}  {_pf(pa):>6}  {_pf(pb):>6}")

    # Case A needs <c,r> ≠ 0, which depends on frequency alignment in src.
    nonzero_A = [r for r in struct_results if abs(r['case_A']['slope']) > 1e-10]
    nonzero_A_pass = all(r['passes_A'] for r in nonzero_A)
    n_nonzero = len(nonzero_A)
    print(f"\nCase A trials with |<c,r>|>0: {n_nonzero}/5 -- all pass: {_pf(nonzero_A_pass)}")
    print(f"All Case B trials PASS: {_pf(all_B_pass)}")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------

    # exp1_case_A.png
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_values, res_A['J0_values'], color='tab:red', linewidth=2)
    ax.axhline(res_A['J0_at_0'], color='grey', linestyle='--', linewidth=1,
               label=f'J₀(0) = {res_A["J0_at_0"]:.4f}')
    ax.set_xlabel('t')
    ax.set_ylabel('J₀(πₜ)')
    ax.set_title(f'Case A: J₀ linear in t  (slope = {slope_A:.4f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp1_case_A.png'), dpi=100)
    plt.close(fig)

    # exp1_case_B.png
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_values, res_B['J0_values_B'], color='tab:blue', linewidth=2)
    ax.axhline(res_B['J0_at_0_B'], color='grey', linestyle='--', linewidth=1,
               label=f'J₀_B(0) = {res_B["J0_at_0_B"]:.6f}')
    ax.set_xlabel('t')
    ax.set_ylabel('J₀(πₜ) with c_B')
    ax.set_title(f'Case B: J₀ constant in t  (variation = {variation_B:.2e})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp1_case_B.png'), dpi=100)
    plt.close(fig)

    # exp1_feasibility.png: marginal residuals along the path
    sample_t = np.linspace(-100.0, 100.0, 41)
    p_resids = []
    q_resids = []
    for t in sample_t:
        pi_t = pi0 + t * r
        feas_t = check_feasibility(pi_t, p, q, tol=1e-10)
        p_resids.append(feas_t['p_resid'])
        q_resids.append(feas_t['q_resid'])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(sample_t, p_resids, 'o-', color='tab:blue', markersize=3, label='p residual')
    ax.semilogy(sample_t, q_resids, 's-', color='tab:orange', markersize=3, label='q residual')
    ax.axhline(1e-10, color='grey', linestyle='--', linewidth=1, label='tol = 1e-10')
    ax.set_xlabel('t')
    ax.set_ylabel('max-norm marginal residual')
    ax.set_title('Feasibility of πₜ = π₀ + t·r along affine path')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp1_feasibility.png'), dpi=100)
    plt.close(fig)

    print(f"\nPlots saved to {RESULTS_DIR}/")

    # ------------------------------------------------------------------
    # 8. Return metrics
    # ------------------------------------------------------------------
    metrics = {
        'zero_marginal_x_resid': zm['x_resid'],
        'zero_marginal_y_resid': zm['y_resid'],
        'pi0_p_resid': feas['p_resid'],
        'pi0_q_resid': feas['q_resid'],
        'pi0_feasible': feas['feasible'],
        'case_A_slope': slope_A,
        'case_A_J0_range': J0_range_A,
        'case_A_passes': res_A['passes_A'],
        'case_B_slope': slope_B,
        'case_B_variation': variation_B,
        'case_B_passes': res_B['passes_B'],
        'structural_nonzero_A_count': n_nonzero,
        'structural_nonzero_A_pass': nonzero_A_pass,
        'structural_all_B_pass': all_B_pass,
        'structural_results': struct_results,
    }
    return metrics


if __name__ == '__main__':
    main()

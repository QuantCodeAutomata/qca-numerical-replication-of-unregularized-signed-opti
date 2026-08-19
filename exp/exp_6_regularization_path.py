"""
Experiment 6: Regularization path — how π*(ε) evolves as ε varies.

Solves the regularized OT problem for a logarithmically spaced sequence of
regularization strengths and tracks the energy, spectral content, and objective
decomposition as ε → 0 and ε → ∞.
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
        dirichlet_energy,
        inner_product,
    )
    from src.regularized_ot import (
        make_synthetic_problem,
        solve_regularized_ot,
    )
except ImportError as e:
    print(f"[ERROR] Core import failed: {e}")
    sys.exit(1)

_RP_AVAILABLE = True
_RP_ERROR = ''
try:
    from src.regularization_path import run_regularization_path
except ImportError as e:
    _RP_AVAILABLE = False
    _RP_ERROR = str(e)

RESULTS_DIR = '/workspace/project/results'

EPS_LIST = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]


def _pf(condition: bool) -> str:
    return 'PASS' if condition else 'FAIL'


def _compute_high_freq_share(pi: np.ndarray, m: np.ndarray, freq_threshold_sq: float) -> float:
    """
    Fraction of Dirichlet energy in modes with m(k,l) > freq_threshold_sq.
    """
    Nx, Ny = pi.shape
    pi_hat = np.fft.fft2(pi)
    N2 = (Nx * Ny) ** 2
    energy_spectrum = m * np.abs(pi_hat) ** 2 / N2      # per-mode energy (unnormalized)
    total = float(np.sum(energy_spectrum))
    if total < 1e-30:
        return 0.0
    high = float(np.sum(energy_spectrum[m > freq_threshold_sq]))
    return high / total


def _run_fallback_path(Nx: int, Ny: int, p: np.ndarray, q: np.ndarray,
                        c: np.ndarray, m: np.ndarray) -> list:
    """
    Solve for each eps using solve_regularized_ot (fallback when
    src/regularization_path.py is not available).
    """
    freq_threshold_sq = (Nx // 4) ** 2
    records = []
    for eps in EPS_LIST:
        sol = solve_regularized_ot(p, q, c, eps, Nx, Ny, method='cg',
                                    tol=1e-10, max_iter=10000)
        hfs = _compute_high_freq_share(sol['pi_star'], m, freq_threshold_sq)
        records.append({
            'eps': eps,
            'pi_star': sol['pi_star'],
            'energy': sol['energy'] / eps if eps > 0 else 0.0,  # H(π*) without eps factor
            'raw_energy': sol['energy'],
            'cost_term': sol['cost_term'],
            'objective': sol['objective'],
            'p_resid': sol['p_resid'],
            'q_resid': sol['q_resid'],
            'high_freq_share': hfs,
        })
    return records


def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n=== Exp 6: Regularization Path ===\n")

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    Nx, Ny = 128, 128
    x, y, p, q, c = make_synthetic_problem(Nx, Ny, marginal_type='exp6', cost_type='exp6')
    m = make_laplacian_multiplier(Nx, Ny)
    freq_threshold_sq = (Nx // 4) ** 2
    print(f"Grid: {Nx}×{Ny},  marginals/cost: exp6")
    # Actual bands used by run_regularization_path: r1=3, r2=6 (m > 36 = high)
    print(f"High-freq threshold (src solver): r > 6  (m(k,l) > 36)\n")

    # ------------------------------------------------------------------
    # 2. Run regularization path
    # ------------------------------------------------------------------
    if _RP_AVAILABLE:
        # run_regularization_path builds its own exp6 problem internally;
        # it returns a dict whose 'path' key is the per-eps list of dicts.
        rp_result = run_regularization_path(Nx=Nx, Ny=Ny, eps_list=EPS_LIST)
        raw_records = rp_result['path']
        # Normalise to a flat dict matching the fallback schema.
        records = []
        for r in raw_records:
            records.append({
                'eps': r['eps'],
                'pi_star': r['pi_star'],
                'energy': r['energy'],
                'raw_energy': r['energy'],
                'cost_term': r['cost_term'],
                'objective': r['objective'],
                'p_resid': r['p_resid'],
                'q_resid': r['q_resid'],
                'high_freq_share': r['spectral_bands']['high_freq_share'],
            })
        print("Source: src/regularization_path.run_regularization_path()")
    else:
        print(f"[WARNING] src/regularization_path.py not available: {_RP_ERROR}")
        print("Using fallback: solve_regularized_ot for each ε.\n")
        records = _run_fallback_path(Nx, Ny, p, q, c, m)

    # ------------------------------------------------------------------
    # 3. Report per-eps diagnostics
    # ------------------------------------------------------------------
    print(f"{'eps':>8}  {'energy':>12}  {'hf_share':>10}  "
          f"{'p_resid':>10}  {'q_resid':>10}  {'feas':>4}")
    print("-" * 62)

    all_feas = True
    for rec in records:
        eps = rec['eps']
        energy = rec.get('raw_energy', rec.get('energy', float('nan')))
        hfs = rec.get('high_freq_share', float('nan'))
        pr = rec.get('p_resid', float('nan'))
        qr = rec.get('q_resid', float('nan'))
        feas_ok = (pr < 1e-6) and (qr < 1e-6)
        all_feas = all_feas and feas_ok
        print(f"{eps:>8.1e}  {energy:>12.6f}  {hfs:>10.4f}  "
              f"{pr:>10.2e}  {qr:>10.2e}  {_pf(feas_ok):>4}")

    print(f"\nAll ε feasible: {_pf(all_feas)}")

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------

    eps_arr = np.array([rec['eps'] for rec in records])
    energy_arr = np.array([rec.get('raw_energy', rec.get('energy', 0.0)) for rec in records])
    hfs_arr = np.array([rec.get('high_freq_share', 0.0) for rec in records])
    cost_arr = np.array([rec.get('cost_term', 0.0) for rec in records])
    obj_arr = np.array([rec.get('objective', 0.0) for rec in records])

    # exp6_heatmaps.png: pi* for eps = 1e-4, 1e-2, 1
    target_eps = [1e-4, 1e-2, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, tgt in zip(axes, target_eps):
        # Find record closest to tgt
        idx = min(range(len(records)), key=lambda i: abs(records[i]['eps'] - tgt))
        pi_plot = records[idx].get('pi_star', np.zeros((Nx, Ny)))
        eps_plot = records[idx]['eps']
        vmax = float(np.max(np.abs(pi_plot)))
        if vmax < 1e-14:
            vmax = 1.0
        im = ax.imshow(pi_plot, origin='lower', aspect='equal', cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax, extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'ε = {eps_plot:.0e}')
        ax.set_xlabel('y')
        ax.set_ylabel('x')
    fig.suptitle('Optimal transport plan π*(x,y) along regularization path', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp6_heatmaps.png'), dpi=100,
                bbox_inches='tight')
    plt.close(fig)

    # exp6_spectral_path.png: high_freq_share vs eps
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(eps_arr, hfs_arr, 'o-', color='tab:purple', linewidth=2, markersize=6)
    ax.set_xlabel('ε')
    ax.set_ylabel('High-frequency energy share')
    ax.set_title('Spectral content vs regularization  (exp6, 128×128)')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp6_spectral_path.png'), dpi=100)
    plt.close(fig)

    # exp6_objective_decomp.png: cost_term and energy vs eps
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(eps_arr, cost_arr, 's-', color='tab:blue', linewidth=2,
                markersize=6, label='cost term ⟨c, π*⟩')
    ax.semilogx(eps_arr, energy_arr, '^-', color='tab:orange', linewidth=2,
                markersize=6, label='ε·H(π*)')
    ax.semilogx(eps_arr, obj_arr, 'o--', color='tab:green', linewidth=1.5,
                markersize=5, label='total J_ε(π*)')
    ax.set_xlabel('ε')
    ax.set_ylabel('Value')
    ax.set_title('Objective decomposition vs ε')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp6_objective_decomp.png'), dpi=100)
    plt.close(fig)

    print(f"\nPlots saved to {RESULTS_DIR}/")

    # ------------------------------------------------------------------
    # 5. Return metrics
    # ------------------------------------------------------------------
    metrics = {
        'rp_available': _RP_AVAILABLE,
        'all_feas': all_feas,
        'eps_list': EPS_LIST,
        'energy_by_eps': dict(zip(EPS_LIST, energy_arr.tolist())),
        'hfs_by_eps': dict(zip(EPS_LIST, hfs_arr.tolist())),
        'cost_by_eps': dict(zip(EPS_LIST, cost_arr.tolist())),
    }
    return metrics


if __name__ == '__main__':
    main()

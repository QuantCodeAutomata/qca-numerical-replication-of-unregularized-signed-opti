"""
Regularization path and spectral flatness analysis for signed OT (Experiment 6).

As epsilon decreases from large to small the regularised solution pi*(eps)
transitions from a smooth (high-entropy, spectrally flat) plan toward a more
singular, high-frequency object. This module:

  - Solves the dual system K z = d for a list of eps values reusing a single
    pre-built K matrix and a pre-computed AGc term.
  - Partitions the Fourier energy spectrum into low / medium / high bands.
  - Tracks smoothness metrics over the regularisation path.
  - Assembles a results dictionary ready for plotting.
"""

from typing import List, Optional
import numpy as np
from scipy import linalg

from src.torus_utils import (
    make_laplacian_multiplier,
    apply_G,
    marginals,
    lift,
    project_zero_mean,
    inner_product,
    dirichlet_energy,
    regularized_objective,
)
from src.regularized_ot import (
    build_K_matrix,
    compute_AGc,
    make_synthetic_problem,  # re-exported for test/caller convenience
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reg_matrix(A: np.ndarray) -> np.ndarray:
    """Add rank-1 correction making A invertible on the zero-mean subspace."""
    n = A.shape[0]
    return A + np.ones((n, n), dtype=np.float64) / n


def _solve_zm_system(K_mat: np.ndarray, d: np.ndarray, Nx: int, Ny: int) -> np.ndarray:
    """
    Solve K z = d with zero-mean constraints on phi and psi blocks.

    Applies independent rank-1 corrections to the (Nx,Nx) and (Ny,Ny)
    diagonal blocks so that scipy.linalg.solve encounters a non-singular
    matrix while the zero-mean projections after solving restore the
    correct answer.
    """
    K_reg = K_mat.copy()
    K_reg[:Nx, :Nx] += np.ones((Nx, Nx), dtype=np.float64) / Nx
    K_reg[Nx:, Nx:] += np.ones((Ny, Ny), dtype=np.float64) / Ny
    z = linalg.solve(K_reg, d)
    z[:Nx] = project_zero_mean(z[:Nx])
    z[Nx:] = project_zero_mean(z[Nx:])
    return z


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def spectral_bands(
    pi: np.ndarray,
    m: np.ndarray,
    r1: float = 4.0,
    r2: float = 12.0,
) -> dict:
    """
    Partition the Fourier energy spectrum of pi into frequency bands.

    Radial frequency: r(k, l) = sqrt(kx^2 + ky^2) = sqrt(m(k, l)).

    Bands
    -----
    low    : r <= r1
    medium : r1 < r <= r2
    high   : r > r2

    Power spectrum (Parseval-normalised):
        power[k, l] = |pi_hat(k, l)|^2 / (Nx * Ny)^2

    Weighted energy (proportional to Dirichlet energy):
        w_power[k, l] = m[k, l] * power[k, l]

    Parameters
    ----------
    pi : (Nx, Ny) float64 transport plan.
    m  : (Nx, Ny) float64 Laplacian multiplier (kx^2 + ky^2).
    r1 : float  Low/medium band boundary.
    r2 : float  Medium/high band boundary.

    Returns
    -------
    dict with band energies, shares, and weighted Dirichlet energies.
    """
    Nx, Ny = pi.shape
    N2 = float(Nx * Ny) ** 2

    pi_hat = np.fft.fft2(pi)
    power = np.abs(pi_hat) ** 2 / N2         # Parseval-normalised power
    w_power = m * power                       # proportional to Dirichlet energy

    r_grid = np.sqrt(m)                       # radial frequency
    mask_low = r_grid <= r1
    mask_med = (r_grid > r1) & (r_grid <= r2)
    mask_hi = r_grid > r2

    low_energy = float(np.sum(power[mask_low]))
    med_energy = float(np.sum(power[mask_med]))
    high_energy = float(np.sum(power[mask_hi]))
    total_energy = float(np.sum(power))

    low_w = float(np.sum(w_power[mask_low]))
    med_w = float(np.sum(w_power[mask_med]))
    high_w = float(np.sum(w_power[mask_hi]))
    total_w = float(np.sum(w_power))

    safe_total = total_energy if total_energy > 0.0 else 1.0
    safe_total_w = total_w if total_w > 0.0 else 1.0

    return {
        'low_energy': low_energy,
        'med_energy': med_energy,
        'high_energy': high_energy,
        'total_energy': total_energy,
        'low_share': low_energy / safe_total,
        'med_share': med_energy / safe_total,
        'high_freq_share': high_energy / safe_total,
        'low_w_energy': low_w,
        'med_w_energy': med_w,
        'high_w_energy': high_w,
        'total_w_energy': total_w,
        'high_w_share': high_w / safe_total_w,
    }


def smoothness_metrics(pi: np.ndarray) -> dict:
    """
    Compute physical-space smoothness measures.

    Parameters
    ----------
    pi : (Nx, Ny) float64 transport plan.

    Returns
    -------
    dict with:
        'l2_norm'      : sqrt(mean(pi^2))
        'neighbor_diff': mean(|pi[i+1,j]-pi[i,j]| + |pi[i,j+1]-pi[i,j]|)
          (periodic boundaries via np.roll).
    """
    l2_norm = float(np.sqrt(np.mean(pi ** 2)))
    diff_x = np.abs(np.roll(pi, -1, axis=0) - pi)
    diff_y = np.abs(np.roll(pi, -1, axis=1) - pi)
    neighbor_diff = float(np.mean(diff_x + diff_y))
    return {
        'l2_norm': l2_norm,
        'neighbor_diff': neighbor_diff,
    }


# ---------------------------------------------------------------------------
# Regularisation path solver
# ---------------------------------------------------------------------------

def solve_regularization_path(
    p: np.ndarray,
    q: np.ndarray,
    c: np.ndarray,
    eps_list: list,
    Nx: int,
    Ny: int,
) -> list:
    """
    Solve regularised OT for each eps in eps_list, reusing AGc across all solves.

    K = A G A* is fixed (depends only on the domain geometry).
    Only d = eps * [p; q] + [AGc_x; AGc_y] changes linearly with eps.

    Parameters
    ----------
    p, q   : (Nx,), (Ny,) zero-mean marginals.
    c      : (Nx, Ny) cost kernel.
    eps_list : list of positive floats (regularisation strengths).
    Nx, Ny : int  grid sizes.

    Returns
    -------
    List of dicts, one per eps.  Each dict contains:
        'eps', 'pi_star', 'phi_star', 'psi_star',
        'objective', 'cost_term', 'energy',
        'p_resid', 'q_resid',
        'spectral_bands', 'smoothness'.
    """
    m = make_laplacian_multiplier(Nx, Ny)
    K_mat = build_K_matrix(Nx, Ny, m)
    AGc_x, AGc_y = compute_AGc(c, m)
    AGc = np.concatenate([AGc_x, AGc_y])
    b = np.concatenate([p, q])

    results = []
    for eps in eps_list:
        d = eps * b + AGc
        z = _solve_zm_system(K_mat, d, Nx, Ny)
        phi_star = z[:Nx]
        psi_star = z[Nx:]

        lifted = lift(phi_star, psi_star, Nx, Ny)
        pi_star = apply_G(lifted - c, m) / eps

        p_marg, q_marg = marginals(pi_star)
        p_resid = float(np.max(np.abs(p_marg - p)))
        q_resid = float(np.max(np.abs(q_marg - q)))

        objective, cost_term, energy = regularized_objective(pi_star, c, m, eps)
        # r1=3, r2=6 → high band captures m > 36, which overlaps with the
        # highest-frequency content in the exp6 cost kernel (m up to ~98).
        bands = spectral_bands(pi_star, m, r1=3.0, r2=6.0)
        smooth = smoothness_metrics(pi_star)

        results.append({
            'eps': float(eps),
            'pi_star': pi_star,
            'phi_star': phi_star,
            'psi_star': psi_star,
            'objective': float(objective),
            'cost_term': float(cost_term),
            'energy': float(energy),
            'p_resid': p_resid,
            'q_resid': q_resid,
            'spectral_bands': bands,
            'smoothness': smooth,
        })

    return results


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

def run_regularization_path(
    Nx: int = 128,
    Ny: int = 128,
    eps_list: Optional[List[float]] = None,
) -> dict:
    """
    Full regularisation path experiment (Experiment 6).

    Steps
    -----
    1. Build synthetic problem with marginal_type='exp6', cost_type='exp6'.
    2. Solve the path over eps_list.
    3. Compute aggregated diagnostics for plotting.
    4. Return organised results dict.

    Parameters
    ----------
    Nx, Ny    : int  Grid sizes (default 128×128).
    eps_list  : list of floats.  Defaults to a log-spaced range from 10 to 0.01.

    Returns
    -------
    dict with keys:
        'eps_list', 'path' (list of per-eps dicts), and scalar arrays
        'objectives', 'cost_terms', 'energies', 'high_freq_shares',
        'smoothness_l2', 'smoothness_neighbor', 'p_resids', 'q_resids',
        'dirichlet_energies', 'total_powers'.
    """
    if eps_list is None:
        eps_list = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    x, y, p, q, c = make_synthetic_problem(Nx, Ny, marginal_type='exp6', cost_type='exp6')
    path = solve_regularization_path(p, q, c, eps_list, Nx, Ny)

    # Unpack scalar series
    m = make_laplacian_multiplier(Nx, Ny)
    n_eps = len(path)
    objectives = np.empty(n_eps, dtype=np.float64)
    cost_terms = np.empty(n_eps, dtype=np.float64)
    energies = np.empty(n_eps, dtype=np.float64)
    high_freq_shares = np.empty(n_eps, dtype=np.float64)
    smoothness_l2 = np.empty(n_eps, dtype=np.float64)
    smoothness_neighbor = np.empty(n_eps, dtype=np.float64)
    p_resids = np.empty(n_eps, dtype=np.float64)
    q_resids = np.empty(n_eps, dtype=np.float64)
    dirichlet_energies = np.empty(n_eps, dtype=np.float64)
    total_powers = np.empty(n_eps, dtype=np.float64)

    for i, res in enumerate(path):
        objectives[i] = res['objective']
        cost_terms[i] = res['cost_term']
        energies[i] = res['energy']
        high_freq_shares[i] = res['spectral_bands']['high_freq_share']
        smoothness_l2[i] = res['smoothness']['l2_norm']
        smoothness_neighbor[i] = res['smoothness']['neighbor_diff']
        p_resids[i] = res['p_resid']
        q_resids[i] = res['q_resid']
        dirichlet_energies[i] = dirichlet_energy(res['pi_star'], m)
        total_powers[i] = res['spectral_bands']['total_energy']

    return {
        'Nx': Nx, 'Ny': Ny,
        'x': x, 'y': y,
        'p': p, 'q': q,
        'eps_list': list(eps_list),
        'path': path,
        'objectives': objectives,
        'cost_terms': cost_terms,
        'energies': energies,
        'high_freq_shares': high_freq_shares,
        'smoothness_l2': smoothness_l2,
        'smoothness_neighbor': smoothness_neighbor,
        'p_resids': p_resids,
        'q_resids': q_resids,
        'dirichlet_energies': dirichlet_energies,
        'total_powers': total_powers,
    }

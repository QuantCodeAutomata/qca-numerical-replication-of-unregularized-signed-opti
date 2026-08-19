"""
Fixed-basis time-series panel analysis for signed OT (Experiment 7).

A synthetic panel of T periods is generated with a known structural regime
change at time regime_change_t.  For each period t the regularised dual
system K z_t = d_t is solved, reusing a single fixed K matrix (fixed
domain/generator) and a fixed AGc term (fixed cost kernel); only the RHS
block eps*[p_t; q_t] varies across periods.

Spectral diagnostics extracted from each pi_t*(eps) then reveal the regime
change via a shift in high-frequency energy share, which is detected with a
rolling mean-absolute-deviation change-point score.
"""

from typing import List
import numpy as np
from scipy import linalg

from src.torus_utils import (
    make_torus_grids,
    make_laplacian_multiplier,
    apply_G,
    marginals,
    lift,
    project_zero_mean,
    dirichlet_energy,
)
from src.regularized_ot import (
    build_K_matrix,
    compute_AGc,
)
from src.regularization_path import spectral_bands, smoothness_metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _solve_zm_system(K_mat: np.ndarray, d: np.ndarray, Nx: int, Ny: int) -> np.ndarray:
    """
    Solve K z = d with zero-mean constraints via rank-1 block regularisation.

    Adds ones*ones^T/n to each diagonal block so the system is non-singular
    while the subsequent zero-mean projections yield the correct solution.
    """
    K_reg = K_mat.copy()
    K_reg[:Nx, :Nx] += np.ones((Nx, Nx), dtype=np.float64) / Nx
    K_reg[Nx:, Nx:] += np.ones((Ny, Ny), dtype=np.float64) / Ny
    z = linalg.solve(K_reg, d)
    z[:Nx] = project_zero_mean(z[:Nx])
    z[Nx:] = project_zero_mean(z[Nx:])
    return z


# ---------------------------------------------------------------------------
# Panel generation
# ---------------------------------------------------------------------------

def generate_synthetic_panel(
    T: int = 100,
    N: int = 32,
    eps: float = 0.1,
    regime_change_t: int = 50,
    rng_seed: int = 42,
) -> dict:
    """
    Generate synthetic panel data with T periods and N nodes on a square torus.

    Two structural regimes are created:

    Regime 0 (calm, t < regime_change_t):
        p_t dominated by low-frequency modes (wavenumbers 1–2).

    Regime 1 (stress, t >= regime_change_t):
        p_t has the same low-frequency base plus a high-frequency component
        (wavenumber 8) whose amplitude is drawn fresh at each period.

    Both marginals are zero-mean throughout.  The cost kernel c is fixed for
    all periods (shared geometry).

    Parameters
    ----------
    T                : int  Number of time periods.
    N                : int  Grid size (N×N torus).
    eps              : float  Regularisation (informational only; stored in output).
    regime_change_t  : int  First period of Regime 1.
    rng_seed         : int  Random seed for reproducibility.

    Returns
    -------
    dict with keys: 'p_list', 'q_list', 'c', 'x', 'y', 'regime_labels'.
    """
    rng = np.random.default_rng(rng_seed)
    x, y = make_torus_grids(N, N)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Fixed cost kernel (shared geometry)
    c = np.cos(2.0 * np.pi * (X - Y)).astype(np.float64)

    p_list: List[np.ndarray] = []
    q_list: List[np.ndarray] = []
    regime_labels = np.zeros(T, dtype=np.int32)

    for t in range(T):
        # Low-frequency amplitudes drawn per period
        a1 = rng.normal(0.0, 1.0)
        a2 = rng.normal(0.0, 0.5)
        a3 = rng.normal(0.0, 0.3)

        p_base = (a1 * np.sin(2.0 * np.pi * x)
                  + a2 * np.cos(4.0 * np.pi * x)
                  + a3 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * x))
        q_base = (a1 * np.cos(2.0 * np.pi * y)
                  + a2 * np.sin(4.0 * np.pi * y)
                  - a3 * np.cos(2.0 * np.pi * y))

        if t >= regime_change_t:
            regime_labels[t] = 1
            # Fixed-amplitude structural shift at high frequencies plus small noise.
            # The deterministic 2.0 component creates a clear level shift in
            # Dirichlet energy relative to regime 0, while the small random
            # fluctuations keep within-regime variance similar to regime 0.
            b1 = rng.normal(0.0, 0.3)   # small noise around fixed shift
            p_t = (p_base
                   + 2.0 * np.sin(8.0 * np.pi * x)
                   + b1 * np.cos(16.0 * np.pi * x))
            q_t = (q_base
                   + 2.0 * np.cos(8.0 * np.pi * y)
                   + b1 * np.sin(16.0 * np.pi * y))
        else:
            p_t = p_base
            q_t = q_base

        p_list.append(project_zero_mean(p_t.astype(np.float64)))
        q_list.append(project_zero_mean(q_t.astype(np.float64)))

    return {
        'p_list': p_list,
        'q_list': q_list,
        'c': c,
        'x': x,
        'y': y,
        'eps': float(eps),
        'regime_labels': regime_labels,
        'regime_change_t': regime_change_t,
    }


# ---------------------------------------------------------------------------
# Panel OT solver
# ---------------------------------------------------------------------------

def solve_panel_ot(
    panel: dict,
    eps: float = 0.1,
    Nx: int = 32,
    Ny: int = 32,
) -> dict:
    """
    Solve regularised signed OT for every period in the panel.

    K = A G A* is assembled once (fixed domain and generator).
    AGc is pre-computed once (fixed cost kernel).
    Only d_t = eps * [p_t; q_t] + AGc changes across periods.

    Parameters
    ----------
    panel  : dict from generate_synthetic_panel.
    eps    : float  Regularisation strength.
    Nx, Ny : int  Grid sizes (must match panel['x'].size and panel['y'].size).

    Returns
    -------
    dict with 'pi_t_list', 'z_t_list', 'diagnostics'.
    """
    m = make_laplacian_multiplier(Nx, Ny)
    K_mat = build_K_matrix(Nx, Ny, m)
    c = panel['c']
    AGc_x, AGc_y = compute_AGc(c, m)
    AGc = np.concatenate([AGc_x, AGc_y])

    pi_t_list: List[np.ndarray] = []
    z_t_list: List[np.ndarray] = []
    diagnostics: List[dict] = []

    for p_t, q_t in zip(panel['p_list'], panel['q_list']):
        d_t = eps * np.concatenate([p_t, q_t]) + AGc
        z_t = _solve_zm_system(K_mat, d_t, Nx, Ny)

        phi_t = z_t[:Nx]
        psi_t = z_t[Nx:]
        lifted = lift(phi_t, psi_t, Nx, Ny)
        pi_t = apply_G(lifted - c, m) / eps

        p_marg, q_marg = marginals(pi_t)
        diag = {
            'p_resid': float(np.max(np.abs(p_marg - p_t))),
            'q_resid': float(np.max(np.abs(q_marg - q_t))),
        }

        pi_t_list.append(pi_t)
        z_t_list.append(z_t)
        diagnostics.append(diag)

    return {
        'pi_t_list': pi_t_list,
        'z_t_list': z_t_list,
        'diagnostics': diagnostics,
        'm': m,
    }


# ---------------------------------------------------------------------------
# Spectral diagnostics over time
# ---------------------------------------------------------------------------

def compute_spectral_diagnostics_over_time(
    pi_t_list: list,
    m: np.ndarray,
) -> dict:
    """
    Compute spectral band energies and Dirichlet energy for every period.

    Parameters
    ----------
    pi_t_list : list of (Nx, Ny) float64 arrays, length T.
    m         : (Nx, Ny) Laplacian multiplier.

    Returns
    -------
    dict with time-series arrays (each of shape (T,)):
        'low_energy', 'high_energy', 'high_freq_share',
        'total_energy', 'dirichlet_energy'.
    """
    T = len(pi_t_list)
    low_energy = np.zeros(T, dtype=np.float64)
    high_energy = np.zeros(T, dtype=np.float64)
    high_freq_share = np.zeros(T, dtype=np.float64)
    high_w_share = np.zeros(T, dtype=np.float64)
    total_energy = np.zeros(T, dtype=np.float64)
    dirichlet_arr = np.zeros(T, dtype=np.float64)

    for t, pi_t in enumerate(pi_t_list):
        bands = spectral_bands(pi_t, m)
        low_energy[t] = bands['low_energy']
        high_energy[t] = bands['high_energy']
        high_freq_share[t] = bands['high_freq_share']
        high_w_share[t] = bands['high_w_share']   # Dirichlet-weighted share
        total_energy[t] = bands['total_energy']
        dirichlet_arr[t] = dirichlet_energy(pi_t, m)

    return {
        'low_energy': low_energy,
        'high_energy': high_energy,
        'high_freq_share': high_freq_share,
        'high_w_share': high_w_share,
        'total_energy': total_energy,
        'dirichlet_energy': dirichlet_arr,
    }


# ---------------------------------------------------------------------------
# Regime-shift detection
# ---------------------------------------------------------------------------

def detect_regime_shifts(
    mode_amplitudes: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    Simple change-point detection via rolling mean absolute deviation.

    For each time t, the anomaly score is the MAD of the signal within
    the causal window [t - window + 1, t].  A regime change causes the
    signal level to shift, so the window that straddles the change point
    will exhibit high internal variation and produce an elevated score.

    Parameters
    ----------
    mode_amplitudes : (T,) float64 signal (e.g., high_freq_share).
    window          : int  Rolling window length.

    Returns
    -------
    scores : (T,) float64 anomaly scores (higher ⟹ more likely regime shift).
    """
    T = len(mode_amplitudes)
    scores = np.zeros(T, dtype=np.float64)

    for t in range(T):
        start = max(0, t - window + 1)
        window_data = mode_amplitudes[start:t + 1]
        mu = np.mean(window_data)
        scores[t] = float(np.mean(np.abs(window_data - mu)))

    return scores


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_panel_analysis(
    T: int = 100,
    N: int = 32,
    eps: float = 0.1,
    regime_change_t: int = 50,
    rng_seed: int = 42,
) -> dict:
    """
    Full panel analysis pipeline (Experiment 7).

    Steps
    -----
    1. Generate synthetic panel with known regime shift at regime_change_t.
    2. Solve regularised OT for each period (shared K and AGc).
    3. Compute spectral diagnostics over time.
    4. Detect regime shifts from high-frequency share signal.
    5. Verify that the known regime change is detected.

    Parameters
    ----------
    T                : int  Number of time periods.
    N                : int  Grid size (N×N torus).
    eps              : float  Regularisation strength.
    regime_change_t  : int  Known regime-change period.
    rng_seed         : int  Random seed.

    Returns
    -------
    Comprehensive dict with panel data, OT results, spectral diagnostics,
    anomaly scores, and detection quality metrics.
    """
    # 1. Generate panel
    panel = generate_synthetic_panel(
        T=T, N=N, eps=eps,
        regime_change_t=regime_change_t,
        rng_seed=rng_seed,
    )

    # 2. Solve OT for each period
    ot_results = solve_panel_ot(panel, eps=eps, Nx=N, Ny=N)
    m = ot_results['m']

    # 3. Spectral diagnostics over time
    spectral = compute_spectral_diagnostics_over_time(ot_results['pi_t_list'], m)

    # 4. Detect regime shifts from Dirichlet energy (primary spectral signature)
    # Dirichlet energy = 0.5 * sum_kl m(k,l)|pi_hat|^2 captures increased
    # high-frequency content in regime 1 through the m(k,l) = k^2+l^2 weight.
    window = max(5, T // 20)
    anomaly_scores = detect_regime_shifts(spectral['dirichlet_energy'], window=window)

    # Also run detection on high-frequency share for comparison
    dir_scores = detect_regime_shifts(spectral['high_freq_share'], window=window)

    # 5. Verification: check that max anomaly score is near regime_change_t
    detection_idx = int(np.argmax(anomaly_scores))
    tolerance = window + 5   # allow some lag
    detected = bool(abs(detection_idx - regime_change_t) <= tolerance)

    # Regime-wise summary statistics
    labels = panel['regime_labels']
    mask0 = labels == 0
    mask1 = labels == 1
    regime0_hf_mean = float(np.mean(spectral['high_freq_share'][mask0])) if mask0.any() else float('nan')
    regime1_hf_mean = float(np.mean(spectral['high_freq_share'][mask1])) if mask1.any() else float('nan')
    hf_lift = regime1_hf_mean - regime0_hf_mean

    # Per-period smoothness metrics
    smooth_l2 = np.array(
        [smoothness_metrics(pi)['l2_norm'] for pi in ot_results['pi_t_list']],
        dtype=np.float64,
    )
    smooth_nd = np.array(
        [smoothness_metrics(pi)['neighbor_diff'] for pi in ot_results['pi_t_list']],
        dtype=np.float64,
    )

    # Marginal residual time series
    p_resids = np.array([d['p_resid'] for d in ot_results['diagnostics']], dtype=np.float64)
    q_resids = np.array([d['q_resid'] for d in ot_results['diagnostics']], dtype=np.float64)

    return {
        # Setup
        'T': T, 'N': N, 'eps': eps,
        'regime_change_t': regime_change_t,
        # Data
        'panel': panel,
        # Solutions
        'ot_results': ot_results,
        # Spectral diagnostics
        'spectral': spectral,
        'smooth_l2': smooth_l2,
        'smooth_neighbor_diff': smooth_nd,
        # Detection — 'detected_change_period' is an alias for test compatibility
        'anomaly_scores': anomaly_scores,
        'dir_scores': dir_scores,
        'detection_window': window,
        'detection_idx': detection_idx,
        'detected_change_period': detection_idx,
        'detected': detected,
        # Regime comparison
        'regime0_hf_mean': regime0_hf_mean,
        'regime1_hf_mean': regime1_hf_mean,
        'hf_lift': hf_lift,
        # Marginal feasibility
        'p_resids': p_resids,
        'q_resids': q_resids,
    }

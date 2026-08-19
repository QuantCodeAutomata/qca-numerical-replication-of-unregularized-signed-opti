"""
Unregularized signed optimal transport ill-posedness demonstration (Experiment 1).

Reproduces the paper's claim that classical OT becomes ill-posed when transport
plans are allowed to be signed and nonnegativity is removed.

Key constructions:
  - Feasible baseline coupling: pi0[i,j] = p[i] + q[j]  (under normalized measure)
  - Perturbation with zero marginals: r(x,y) = a(x) * b(y), both factors zero-mean
  - Affine family: pi_t = pi0 + t * r  for t in R
  - Unregularized objective: J0(pi_t) = <c, pi_t> = <c, pi0> + t * <c, r>
  - Case A: <c, r> != 0  =>  J0 is linearly unbounded (ill-posed)
  - Case B: <c, r> = 0   =>  J0 is constant (non-unique optimal set)

# Custom — Context7 found no library equivalent (paper Eq. ill-posedness construction)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.torus_utils import (
    make_torus_grids,
    marginals,
    project_zero_mean,
    inner_product,
    check_feasibility,
)


# ---------------------------------------------------------------------------
# Baseline coupling and perturbation construction
# ---------------------------------------------------------------------------

def construct_baseline_coupling(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Construct the feasible baseline signed coupling pi0.

    Under normalized Haar measure (mu(X) = nu(Y) = 1):
      pi0[i, j] = p[i] + q[j]

    This satisfies:
      P_X pi0 = mean_j(pi0[i,j]) = p[i] + mean_j(q[j]) = p[i]  (since q has zero mean)
      P_Y pi0 = mean_i(pi0[i,j]) = mean_i(p[i]) + q[j] = q[j]  (since p has zero mean)

    Parameters
    ----------
    p : (Nx,) zero-mean signed marginal on X.
    q : (Ny,) zero-mean signed marginal on Y.

    Returns
    -------
    pi0 : (Nx, Ny) baseline coupling.
    """
    assert np.abs(np.mean(p)) < 1e-12, "p must have zero mean"
    assert np.abs(np.mean(q)) < 1e-12, "q must have zero mean"
    return p[:, None] + q[None, :]


def construct_zero_marginal_perturbation(
    x: np.ndarray,
    y: np.ndarray,
    freq_x: int = 1,
    freq_y: int = 1,
) -> np.ndarray:
    """
    Construct a nonzero perturbation r(x,y) with zero x- and y-marginals.

    Uses the outer-product construction: r(x,y) = a(x) * b(y), where
    both a and b are zero-mean functions.

    a(x) = sin(2*pi*freq_x*x),  b(y) = sin(2*pi*freq_y*y)

    Their means on the uniform grid are exactly zero, so both marginals of r vanish.

    Parameters
    ----------
    x : (Nx,) grid on T.
    y : (Ny,) grid on T.
    freq_x, freq_y : int  Fourier frequencies.

    Returns
    -------
    r : (Nx, Ny) perturbation with zero marginals.
    """
    a = np.sin(2 * np.pi * freq_x * x)
    b = np.sin(2 * np.pi * freq_y * y)
    # Project to exact zero mean (should already be zero for integer frequencies on uniform grid)
    a = project_zero_mean(a)
    b = project_zero_mean(b)
    return np.outer(a, b)


def verify_zero_marginals(r: np.ndarray, tol: float = 1e-12) -> dict:
    """
    Verify that r has zero x- and y-marginals.

    Returns dict with x_resid, y_resid, and passes boolean.
    """
    r_x, r_y = marginals(r)
    x_resid = float(np.max(np.abs(r_x)))
    y_resid = float(np.max(np.abs(r_y)))
    return {
        'x_resid': x_resid,
        'y_resid': y_resid,
        'passes': (x_resid < tol) and (y_resid < tol),
    }


# ---------------------------------------------------------------------------
# Affine feasible family
# ---------------------------------------------------------------------------

def affine_family(
    pi0: np.ndarray, r: np.ndarray, t_values: np.ndarray
) -> List[np.ndarray]:
    """
    Construct the affine family pi_t = pi0 + t * r for each t in t_values.

    Parameters
    ----------
    pi0      : (Nx, Ny) baseline coupling.
    r        : (Nx, Ny) zero-marginal perturbation.
    t_values : (T,) array of scalar t values.

    Returns
    -------
    List of (Nx, Ny) arrays.
    """
    return [pi0 + t * r for t in t_values]


# ---------------------------------------------------------------------------
# Objective along the affine path
# ---------------------------------------------------------------------------

def objective_along_path(
    c: np.ndarray,
    pi0: np.ndarray,
    r: np.ndarray,
    t_values: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute J0(pi_t) = <c, pi0> + t * <c, r> for each t.

    This is exact because <c, pi0 + t*r> = <c, pi0> + t*<c,r>.

    Parameters
    ----------
    c        : (Nx, Ny) cost kernel.
    pi0      : (Nx, Ny) baseline coupling.
    r        : (Nx, Ny) perturbation direction.
    t_values : (T,) array of t values.

    Returns
    -------
    J0_values : (T,) objective values.
    J0_at_0   : float  objective at t=0.
    slope     : float  <c, r> — slope of the linear function in t.
    """
    J0_at_0 = inner_product(c, pi0)
    slope = inner_product(c, r)
    J0_values = J0_at_0 + slope * t_values
    return J0_values, J0_at_0, slope


# ---------------------------------------------------------------------------
# Main ill-posedness demonstrations
# ---------------------------------------------------------------------------

def run_case_A(
    p: np.ndarray,
    q: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
    t_values: np.ndarray,
    tol: float = 1e-10,
) -> dict:
    """
    Case A: <c, r> != 0 → objective is unbounded below.

    Parameters
    ----------
    p, q : signed zero-mean marginals.
    c    : cost kernel such that <c, r> != 0.
    r    : zero-marginal perturbation.
    t_values : range of t to test.
    tol  : feasibility tolerance.

    Returns
    -------
    dict with objective values, slope, feasibility checks.
    """
    pi0 = construct_baseline_coupling(p, q)
    J0_values, J0_at_0, slope = objective_along_path(c, pi0, r, t_values)

    # Verify feasibility for a sample of t values
    sample_idx = [0, len(t_values) // 4, len(t_values) // 2, -1]
    feasibility_checks = []
    for idx in sample_idx:
        pi_t = pi0 + t_values[idx] * r
        feas = check_feasibility(pi_t, p, q, tol=tol)
        feas['t'] = float(t_values[idx])
        feasibility_checks.append(feas)

    # Confirm unboundedness: minimum of J0 should go to -inf in one direction
    t_neg = t_values[t_values < 0]
    t_pos = t_values[t_values > 0]
    if slope > 0:
        unbounded_t = t_neg
        unbounded_direction = 'negative_t'
    else:
        unbounded_t = t_pos
        unbounded_direction = 'positive_t'

    return {
        't_values': t_values,
        'J0_values': J0_values,
        'J0_at_0': J0_at_0,
        'slope': slope,
        'inner_cr': slope,
        'unbounded_direction': unbounded_direction,
        'feasibility_checks': feasibility_checks,
        'norm_r': float(np.sqrt(inner_product(r, r))),
        'passes_A': abs(slope) > 1e-10,  # confirms <c,r> != 0
    }


def run_case_B(
    p: np.ndarray,
    q: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
    t_values: np.ndarray,
    tol: float = 1e-10,
) -> dict:
    """
    Case B: <c, r> = 0 (after projecting c to be orthogonal to r) →
    objective is constant, implying non-uniqueness.

    Constructs c_B = c - (<c,r>/<r,r>) * r to enforce orthogonality.

    Returns
    -------
    dict with objective values (should all be equal), feasibility, c_B.
    """
    pi0 = construct_baseline_coupling(p, q)

    # Project c to be orthogonal to r
    c_r_inner = inner_product(c, r)
    r_r_inner = inner_product(r, r)
    assert r_r_inner > 1e-14, "r must be nonzero"
    c_B = c - (c_r_inner / r_r_inner) * r

    # Now <c_B, r> should be zero
    slope_B = inner_product(c_B, r)
    J0_values_B, J0_at_0_B, _ = objective_along_path(c_B, pi0, r, t_values)

    # All objective values should equal J0_at_0_B
    obj_variation = float(np.max(np.abs(J0_values_B - J0_at_0_B)))

    # Verify feasibility
    sample_idx = [0, len(t_values) // 4, len(t_values) // 2, -1]
    feasibility_checks = []
    for idx in sample_idx:
        pi_t = pi0 + t_values[idx] * r
        feas = check_feasibility(pi_t, p, q, tol=tol)
        feas['t'] = float(t_values[idx])
        feasibility_checks.append(feas)

    return {
        't_values': t_values,
        'J0_values_B': J0_values_B,
        'J0_at_0_B': J0_at_0_B,
        'slope_B': float(slope_B),
        'c_B': c_B,
        'obj_variation': obj_variation,
        'feasibility_checks': feasibility_checks,
        'passes_B': obj_variation < 1e-10,  # all objectives are equal
    }


# ---------------------------------------------------------------------------
# Multi-trial structural verification
# ---------------------------------------------------------------------------

def run_structural_verification(
    Nx: int = 64,
    Ny: int = 64,
    num_trials: int = 5,
    t_range: float = 100.0,
    n_t: int = 201,
    rng: Optional[np.random.Generator] = None,
) -> List[dict]:
    """
    Repeat the ill-posedness demonstration for several independent choices
    of p, q, c, r to show the behavior is structural.

    Parameters
    ----------
    Nx, Ny    : grid sizes.
    num_trials: number of independent trials.
    t_range   : t in [-t_range, t_range].
    n_t       : number of t values to test.
    rng       : random number generator (for reproducibility).

    Returns
    -------
    List of result dicts, one per trial.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    x = np.arange(Nx, dtype=np.float64) / Nx
    y = np.arange(Ny, dtype=np.float64) / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')
    t_values = np.linspace(-t_range, t_range, n_t)

    results = []
    for trial_idx in range(num_trials):
        # Random zero-mean functions using sines and cosines
        freqs_p = rng.integers(1, 5, size=3)
        freqs_q = rng.integers(1, 5, size=3)
        amps_p = rng.uniform(-1, 1, size=3)
        amps_q = rng.uniform(-1, 1, size=3)
        p = project_zero_mean(
            sum(a * np.sin(2 * np.pi * k * x) for a, k in zip(amps_p, freqs_p))
        )
        q = project_zero_mean(
            sum(a * np.cos(2 * np.pi * k * y) for a, k in zip(amps_q, freqs_q))
        )

        # Perturbation (construct first so we can align c for Case A)
        freq_rx = rng.integers(1, 5)
        freq_ry = rng.integers(1, 5)
        r = construct_zero_marginal_perturbation(x, y, freq_rx, freq_ry)

        # Base cost kernel: random Fourier combination
        freq_cx = rng.integers(1, 4)
        freq_cy = rng.integers(1, 4)
        c_base = (np.sin(2 * np.pi * freq_cx * X) * np.cos(2 * np.pi * freq_cy * Y)
                  + 0.3 * np.cos(4 * np.pi * (X - Y)))

        # Case A: guarantee <c, r> != 0 by adding a unit-scaled copy of r.
        # This matches the paper's instruction "choose or adjust c so that
        # <c, r> != 0" (paper Section 3, Case A construction).
        cr_base = inner_product(c_base, r)
        r_norm_sq = inner_product(r, r)
        if abs(cr_base) < 0.05 * r_norm_sq:
            c = c_base + r          # guarantees <c, r> = cr_base + ||r||^2 > 0
        else:
            c = c_base

        res_A = run_case_A(p, q, c, r, t_values)
        # Case B uses original c_base (then projects it ⊥ r internally)
        res_B = run_case_B(p, q, c_base, r, t_values)
        results.append({
            'trial': trial_idx,
            'case_A': res_A,
            'case_B': res_B,
            'passes_A': res_A['passes_A'],
            'passes_B': res_B['passes_B'],
        })

    return results

"""
Richardson iteration for the dual linear system (Experiment 4).

Analyzes convergence of stationary Richardson iterations
  z^{n+1} = z^n - alpha * (K z^n - d)
as a function of step size alpha relative to the spectrum of K.

The dual operator K = A G A* is symmetric positive semi-definite on R^{Nx+Ny}.
Its null space consists of block-constant vectors (one constant per block),
so iterates are projected onto the zero-mean subspace after each step.
"""

from typing import Optional

import numpy as np
import scipy.linalg

from src.regularized_ot import build_K_matrix, compute_rhs, make_synthetic_problem
from src.torus_utils import make_laplacian_multiplier, project_zero_mean


# ---------------------------------------------------------------------------
# Richardson iteration
# ---------------------------------------------------------------------------

def richardson_iteration(
    K_mat: np.ndarray,
    d: np.ndarray,
    alpha: float,
    z0: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-12,
    z_ref: Optional[np.ndarray] = None,
    Nx: Optional[int] = None,
) -> dict:
    """
    Richardson iteration: z^{n+1} = z^n - alpha * (K z^n - d).

    After each update the iterate is projected to the zero-mean subspace
    (each of the phi and psi blocks is centered independently) when Nx is given.

    Parameters
    ----------
    K_mat    : (N, N) symmetric PSD matrix.
    d        : (N,) right-hand side vector.
    alpha    : float step size.
    z0       : (N,) initial iterate.
    max_iter : int maximum number of iterations.
    tol      : float convergence threshold on ||K z^n - d||.
    z_ref    : (N,) optional reference solution for true-error tracking.
    Nx       : int optional split index; z[:Nx] is the phi block and z[Nx:] the psi block.
               When provided, each block is centered after every update.

    Returns
    -------
    dict with keys:
      'z'              : (N,) final iterate.
      'residuals'      : array of ||K z^n - d|| values (one per iteration).
      'iter_diffs'     : array of ||z^{n+1} - z^n|| values.
      'true_errors'    : array of ||z^n - z_ref|| values (empty if z_ref is None).
      'converged'      : bool  residual fell below tol.
      'n_iter'         : int  number of iterations performed.
      'final_residual' : float  last recorded residual norm.
      'empirical_contraction' : float  geometric mean contraction over last 50 steps.
    """
    z = z0.copy().astype(np.float64)
    residuals: list = []
    iter_diffs: list = []
    true_errors: list = []

    converged = False
    for _ in range(max_iter):
        Kz = K_mat @ z
        r = Kz - d
        res_norm = float(np.linalg.norm(r))
        residuals.append(res_norm)

        if z_ref is not None:
            true_errors.append(float(np.linalg.norm(z - z_ref)))

        if res_norm < tol:
            converged = True
            break

        if not np.isfinite(res_norm):
            break

        z_new = z - alpha * r

        # Project each block to zero mean
        if Nx is not None:
            z_new[:Nx] -= np.mean(z_new[:Nx])
            z_new[Nx:] -= np.mean(z_new[Nx:])

        iter_diffs.append(float(np.linalg.norm(z_new - z)))
        z = z_new

    # Empirical contraction rate: geometric mean of ratio over last 50 steps
    empirical_contraction = float('nan')
    if len(residuals) >= 2:
        window = residuals[-50:]
        if len(window) >= 2 and window[0] > 0 and window[-1] > 0:
            ratio = window[-1] / window[0]
            if ratio > 0:
                empirical_contraction = float(ratio ** (1.0 / (len(window) - 1)))

    return {
        'z': z,
        'residuals': np.array(residuals, dtype=np.float64),
        'iter_diffs': np.array(iter_diffs, dtype=np.float64),
        'true_errors': np.array(true_errors, dtype=np.float64),
        'converged': converged,
        'n_iter': len(residuals),
        'final_residual': float(residuals[-1]) if residuals else float('nan'),
        'empirical_contraction': empirical_contraction,
    }


# ---------------------------------------------------------------------------
# Full convergence analysis
# ---------------------------------------------------------------------------

def run_richardson_analysis(
    Nx: int = 32,
    Ny: int = 32,
    eps: float = 0.1,
    max_iter: int = 3000,
) -> dict:
    """
    Full Richardson convergence analysis for the dual operator K.

    Steps
    -----
    1. Build K_mat via build_K_matrix and d via compute_rhs.
    2. Compute reference solution z_ref = scipy.linalg.solve(K_mat, d),
       then project each block to zero mean.
    3. Estimate lambda_min, lambda_max via scipy.linalg.eigh (positive eigenvalues only).
    4. Compute alpha_opt = 2 / (lambda_min + lambda_max).
    5. Test four step sizes: alpha_opt, 1/lambda_max, 1.9/lambda_max, 2.1/lambda_max.
    6. Run Richardson from z0 = 0 for each alpha; track residuals and true errors.
    7. Compute theoretical contraction rho = (lambda_max - lambda_min) / (lambda_max + lambda_min).

    Parameters
    ----------
    Nx, Ny   : int  grid sizes.
    eps      : float  regularization strength.
    max_iter : int  maximum Richardson iterations.

    Returns
    -------
    dict with keys:
      'lambda_min'    : float  smallest positive eigenvalue of K.
      'lambda_max'    : float  largest eigenvalue of K.
      'kappa'         : float  condition number lambda_max / lambda_min.
      'rho_theoretical': float  optimal contraction factor.
      'alpha_opt'     : dict  results for alpha = 2 / (lambda_min + lambda_max).
      'alpha_safe'    : dict  results for alpha = 1 / lambda_max.
      'alpha_near'    : dict  results for alpha = 1.9 / lambda_max.
      'alpha_diverge' : dict  results for alpha = 2.1 / lambda_max  (divergent).

    Each per-alpha dict contains:
      'converged'      : bool
      'final_error'    : float  ||z_final - z_ref||
      'final_residual' : float  ||K z_final - d||
      'n_iter'         : int
      'residual_history': ndarray
      'empirical_contraction': float
      'iterations'    : int  alias for n_iter
    """
    # ------------------------------------------------------------------
    # 1. Build system
    # ------------------------------------------------------------------
    _, _, p, q, c = make_synthetic_problem(Nx, Ny)
    m = make_laplacian_multiplier(Nx, Ny)
    K_mat = build_K_matrix(Nx, Ny, m)
    d = compute_rhs(p, q, c, m, eps)

    # ------------------------------------------------------------------
    # 2. Reference solution via direct solve; project blocks to zero mean
    # ------------------------------------------------------------------
    z_ref = scipy.linalg.solve(K_mat, d)
    z_ref[:Nx] -= np.mean(z_ref[:Nx])
    z_ref[Nx:] -= np.mean(z_ref[Nx:])

    # ------------------------------------------------------------------
    # 3. Eigenvalues via scipy.linalg.eigh
    # ------------------------------------------------------------------
    eigvals = scipy.linalg.eigh(K_mat, eigvals_only=True)
    # Retain only eigenvalues that are meaningfully positive
    threshold = max(float(eigvals.max()) * 1e-8, 1e-14)
    pos_eigvals = eigvals[eigvals > threshold]
    lambda_min = float(pos_eigvals.min())
    lambda_max = float(pos_eigvals.max())
    kappa = lambda_max / lambda_min

    # ------------------------------------------------------------------
    # 4-5. Step sizes
    # ------------------------------------------------------------------
    alpha_opt_val  = 2.0 / (lambda_min + lambda_max)
    alpha_safe_val = 1.0 / lambda_max
    alpha_near_val = 1.9 / lambda_max
    alpha_div_val  = 2.1 / lambda_max

    # Theoretical contraction at alpha_opt
    rho_theoretical = (lambda_max - lambda_min) / (lambda_max + lambda_min)

    # ------------------------------------------------------------------
    # 6. Run Richardson for each step size
    # ------------------------------------------------------------------
    z0 = np.zeros(Nx + Ny, dtype=np.float64)

    def run_alpha(alpha: float) -> dict:
        raw = richardson_iteration(
            K_mat, d, alpha, z0,
            max_iter=max_iter,
            tol=1e-10,
            z_ref=z_ref,
            Nx=Nx,
        )
        final_error = float(np.linalg.norm(raw['z'] - z_ref))
        return {
            'converged': raw['converged'],
            'final_error': final_error,
            'final_residual': raw['final_residual'],
            'n_iter': raw['n_iter'],
            'iterations': raw['n_iter'],
            'residual_history': raw['residuals'],
            'iter_diffs': raw['iter_diffs'],
            'true_errors': raw['true_errors'],
            'empirical_contraction': raw['empirical_contraction'],
        }

    return {
        'lambda_min': lambda_min,
        'lambda_max': lambda_max,
        'kappa': kappa,
        'rho_theoretical': rho_theoretical,
        'alpha_opt': run_alpha(alpha_opt_val),
        'alpha_safe': run_alpha(alpha_safe_val),
        'alpha_near': run_alpha(alpha_near_val),
        'alpha_diverge': run_alpha(alpha_div_val),
    }

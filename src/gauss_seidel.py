"""
Block Gauss-Seidel iteration for the regularized signed OT dual system (Experiment 5).

The dual system K z = d has the block structure
    K = [[KXX, KXY], [KYX, KYY]]
where z = [phi; psi] and d = [dX; dY].

Block Gauss-Seidel sweeps:
    phi_new = KXX^{-1} (dX - KXY psi)
    psi_new = KYY^{-1} (dY - KYX phi_new)

The error propagation operator for psi is M = KYY^{-1} KYX KXX^{-1} KXY,
and convergence rate is governed by rho(M) = max |eigenvalues(M)|.

All block solves are performed on the zero-mean subspace via a rank-1
correction that pins the null direction (constant vectors) without
affecting zero-mean solutions.
"""

from typing import Tuple
import numpy as np
from scipy import linalg

from src.torus_utils import (
    make_laplacian_multiplier,
    make_freq_grids,
    marginals,
    lift,
    project_zero_mean,
    inner_product,
    dirichlet_energy,
)
from src.regularized_ot import (
    build_K_matrix,
    compute_rhs,
    compute_AGc,
    make_synthetic_problem,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reg_matrix(A: np.ndarray) -> np.ndarray:
    """
    Return A + ones*ones^T / n  so that the matrix is invertible on R^n
    while its restriction to the zero-mean subspace is unchanged.

    When the RHS is zero-mean and A v = 0 implies v is constant, the
    solution of (A + ones*ones^T/n) x = b is the unique zero-mean solution
    of A x = b.
    """
    n = A.shape[0]
    return A + np.ones((n, n), dtype=np.float64) / n


def _solve_zm(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b on the zero-mean subspace via rank-1 regularisation."""
    x = linalg.solve(_reg_matrix(A), b)
    return project_zero_mean(x)


def _dirichlet_energy_1d(phi: np.ndarray, kx2: np.ndarray) -> float:
    """
    1-D Dirichlet energy  H_X(phi) = 0.5 / Nx^2 * sum_k kx^2 |phi_hat_k|^2.

    Parameters
    ----------
    phi : (Nx,) real array (zero-mean).
    kx2 : (Nx,) squared integer wavenumbers from make_freq_grids.

    Returns
    -------
    float
    """
    Nx = phi.shape[0]
    phi_hat = np.fft.fft(phi)
    return 0.5 * float(np.sum(kx2 * np.abs(phi_hat) ** 2)) / Nx ** 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble_blocks(K_mat: np.ndarray, Nx: int, Ny: int) -> dict:
    """
    Extract block decomposition of K = [[KXX, KXY], [KYX, KYY]].

    KXX = K_mat[:Nx, :Nx],   KXY = K_mat[:Nx, Nx:]
    KYX = K_mat[Nx:, :Nx],   KYY = K_mat[Nx:, Nx:]

    Also verifies KXY.T ≈ KYX and that KXX, KYY are symmetric.

    Parameters
    ----------
    K_mat : (Nx+Ny, Nx+Ny) float64 K matrix from build_K_matrix.
    Nx, Ny : int  Grid sizes.

    Returns
    -------
    dict with keys 'KXX', 'KXY', 'KYX', 'KYY',
    'sym_KXX_residual', 'sym_KYY_residual', 'sym_KXY_KYX_residual'.
    """
    KXX = K_mat[:Nx, :Nx].copy()
    KXY = K_mat[:Nx, Nx:].copy()
    KYX = K_mat[Nx:, :Nx].copy()
    KYY = K_mat[Nx:, Nx:].copy()

    sym_KXX = float(np.max(np.abs(KXX - KXX.T)))
    sym_KYY = float(np.max(np.abs(KYY - KYY.T)))
    sym_KXY_KYX = float(np.max(np.abs(KXY.T - KYX)))

    return {
        'KXX': KXX,
        'KXY': KXY,
        'KYX': KYX,
        'KYY': KYY,
        'sym_KXX_residual': sym_KXX,
        'sym_KYY_residual': sym_KYY,
        'sym_KXY_KYX_residual': sym_KXY_KYX,
    }


def gauss_seidel_step(
    blocks: dict,
    dX: np.ndarray,
    dY: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One block Gauss-Seidel step.

    Computes:
        phi_new = KXX^{-1} (dX - KXY psi)
        psi_new = KYY^{-1} (dY - KYX phi_new)

    Both outputs are projected to zero mean.

    Parameters
    ----------
    blocks : dict from assemble_blocks (must contain 'KXX','KXY','KYX','KYY').
    dX : (Nx,) RHS block for phi.
    dY : (Ny,) RHS block for psi.
    phi : (Nx,) current phi iterate (zero-mean).
    psi : (Ny,) current psi iterate (zero-mean).

    Returns
    -------
    (phi_new, psi_new) — both zero-mean float64 arrays.
    """
    KXX = blocks['KXX']
    KXY = blocks['KXY']
    KYX = blocks['KYX']
    KYY = blocks['KYY']

    phi_new = _solve_zm(KXX, dX - KXY @ psi)
    psi_new = _solve_zm(KYY, dY - KYX @ phi_new)
    return phi_new, psi_new


def compute_error_propagation_operator(blocks: dict) -> np.ndarray:
    """
    Compute M = KYY^{-1} KYX KXX^{-1} KXY.

    The psi-component error satisfies e_psi^{n+1} = M e_psi^n,
    so convergence rate is rho(M) = max |eigenvalues(M)|.

    Parameters
    ----------
    blocks : dict from assemble_blocks.

    Returns
    -------
    M : (Ny, Ny) float64 error propagation matrix.
    """
    KXX = blocks['KXX']
    KXY = blocks['KXY']
    KYX = blocks['KYX']
    KYY = blocks['KYY']

    Ny = KYY.shape[0]
    KXX_reg = _reg_matrix(KXX)
    KYY_reg = _reg_matrix(KYY)

    M = np.zeros((Ny, Ny), dtype=np.float64)
    for j in range(Ny):
        # Step 1: w = KXX^{-1} KXY[:,j]
        w = linalg.solve(KXX_reg, KXY[:, j])
        w = project_zero_mean(w)
        # Step 2: v = KYX w
        v = KYX @ w
        # Step 3: M[:,j] = KYY^{-1} v
        u = linalg.solve(KYY_reg, v)
        M[:, j] = project_zero_mean(u)

    return M


def lifting_norm_check(
    phi: np.ndarray,
    psi: np.ndarray,
    m: np.ndarray,
    Nx: int,
    Ny: int,
) -> dict:
    """
    Verify lifting norm identities for the normalised torus (mu(X) = nu(Y) = 1).

    For phi on X:
        ||PX* phi||_{E0}  =  dirichlet_energy(lift(phi, 0_{Ny}), m_XY)
        ||phi||_{EX^0}    =  H_X(phi)  = 0.5/Nx^2 * sum_k kx^2 |phi_hat_k|^2
    These must be equal (identity holds exactly).

    Analogously for psi on Y.

    Parameters
    ----------
    phi : (Nx,) zero-mean potential.
    psi : (Ny,) zero-mean potential.
    m   : (Nx, Ny) Laplacian multiplier from make_laplacian_multiplier.
    Nx, Ny : int  grid sizes.

    Returns
    -------
    dict with 'norm_phi_lifted', 'norm_phi_X', 'diff_phi',
              'norm_psi_lifted', 'norm_psi_Y', 'diff_psi'.
    """
    kx, ky = make_freq_grids(Nx, Ny)
    kx2 = kx ** 2
    ky2 = ky ** 2

    zeros_y = np.zeros(Ny, dtype=np.float64)
    zeros_x = np.zeros(Nx, dtype=np.float64)

    norm_phi_lifted = dirichlet_energy(lift(phi, zeros_y, Nx, Ny), m)
    norm_phi_X = _dirichlet_energy_1d(phi, kx2)

    norm_psi_lifted = dirichlet_energy(lift(zeros_x, psi, Nx, Ny), m)
    norm_psi_Y = _dirichlet_energy_1d(psi, ky2)

    return {
        'norm_phi_lifted': float(norm_phi_lifted),
        'norm_phi_X': float(norm_phi_X),
        'diff_phi': float(abs(norm_phi_lifted - norm_phi_X)),
        'norm_psi_lifted': float(norm_psi_lifted),
        'norm_psi_Y': float(norm_psi_Y),
        'diff_psi': float(abs(norm_psi_lifted - norm_psi_Y)),
    }


def run_block_gauss_seidel_analysis(
    Nx: int = 32,
    Ny: int = 32,
    eps: float = 0.1,
    max_iter: int = 2000,
    tol: float = 1e-12,
    rng_seed: int = 42,
) -> dict:
    """
    Full Gauss-Seidel analysis pipeline.

    Steps
    -----
    1. Build synthetic problem; assemble K_mat (Nx+Ny, Nx+Ny).
    2. Compute reference solution z_ref via direct (regularised) solve.
    3. Extract blocks KXX, KXY, KYX, KYY and verify symmetry/structure.
    4. Verify block properties (symmetry residuals, min eigenvalue > 0).
    5. Compute M = KYY^{-1} KYX KXX^{-1} KXY and spectral radius rho(M).
    6. Run Gauss-Seidel from psi^0 = 0 and psi^0 = random.
    7. Track full residual ||K z^n - d||_2, block residuals, and true errors.
    8. Compare empirical per-step rate with rho(M).
    9. Run lifting norm checks on the reference potentials.

    Parameters
    ----------
    Nx, Ny    : int   Grid sizes.
    eps       : float Regularisation strength.
    max_iter  : int   Maximum GS iterations.
    tol       : float Early-stop residual threshold.
    rng_seed  : int   Random seed for random initialisation.

    Returns
    -------
    dict with all analysis results.
    """
    rng = np.random.default_rng(rng_seed)

    # 1. Build problem
    x, y, p, q, c = make_synthetic_problem(Nx, Ny)
    m = make_laplacian_multiplier(Nx, Ny)
    K_mat = build_K_matrix(Nx, Ny, m)
    d = compute_rhs(p, q, c, m, eps)
    dX, dY = d[:Nx], d[Nx:]

    # 2. Reference solution via direct zero-mean solve
    dim = Nx + Ny
    K_reg = K_mat.copy()
    K_reg[:Nx, :Nx] += np.ones((Nx, Nx), dtype=np.float64) / Nx
    K_reg[Nx:, Nx:] += np.ones((Ny, Ny), dtype=np.float64) / Ny
    z_ref = linalg.solve(K_reg, d)
    phi_ref = project_zero_mean(z_ref[:Nx])
    psi_ref = project_zero_mean(z_ref[Nx:])
    z_ref = np.concatenate([phi_ref, psi_ref])

    ref_residual = float(np.linalg.norm(K_mat @ z_ref - d))

    # 3. Block decomposition
    blocks = assemble_blocks(K_mat, Nx, Ny)

    # 4. Block eigenvalue properties
    eigs_KXX = np.sort(linalg.eigvals(blocks['KXX']).real)
    eigs_KYY = np.sort(linalg.eigvals(blocks['KYY']).real)

    # 5. Error propagation operator and spectral radius
    M = compute_error_propagation_operator(blocks)
    eigs_M = linalg.eigvals(M)
    rho_M = float(np.max(np.abs(eigs_M)))

    # 6 & 7. GS iterations — helper closure
    def _run_gs(psi_init: np.ndarray, label: str) -> dict:
        phi = np.zeros(Nx, dtype=np.float64)
        psi = psi_init.copy()
        residuals = []
        block_resid_X = []
        block_resid_Y = []
        err_phi = []
        err_psi = []

        for _ in range(max_iter):
            phi, psi = gauss_seidel_step(blocks, dX, dY, phi, psi)
            z = np.concatenate([phi, psi])

            full_resid = float(np.linalg.norm(K_mat @ z - d))
            residuals.append(full_resid)

            KXX, KXY = blocks['KXX'], blocks['KXY']
            KYX, KYY = blocks['KYX'], blocks['KYY']
            br_X = float(np.linalg.norm(KXX @ phi + KXY @ psi - dX))
            br_Y = float(np.linalg.norm(KYX @ phi + KYY @ psi - dY))
            block_resid_X.append(br_X)
            block_resid_Y.append(br_Y)

            err_phi.append(float(np.linalg.norm(phi - phi_ref)))
            err_psi.append(float(np.linalg.norm(psi - psi_ref)))

            if full_resid < tol:
                break

        residuals = np.array(residuals, dtype=np.float64)
        err_psi_arr = np.array(err_psi, dtype=np.float64)

        # Empirical per-step convergence rate (ratio of successive psi errors)
        safe = err_psi_arr > 1e-15
        if np.sum(safe) >= 2:
            rates = err_psi_arr[1:][safe[1:]] / np.maximum(err_psi_arr[:-1][safe[:-1]], 1e-300)
            empirical_rate = float(np.median(rates[rates < 2.0])) if len(rates) > 0 else float('nan')
        else:
            empirical_rate = float('nan')

        return {
            'label': label,
            'phi_final': phi,
            'psi_final': psi,
            'residuals': residuals,
            'block_resid_X': np.array(block_resid_X, dtype=np.float64),
            'block_resid_Y': np.array(block_resid_Y, dtype=np.float64),
            'err_phi': np.array(err_phi, dtype=np.float64),
            'err_psi': err_psi_arr,
            'n_iters': len(residuals),
            'final_residual': float(residuals[-1]),
            'empirical_rate': empirical_rate,
        }

    gs_zeros = _run_gs(np.zeros(Ny, dtype=np.float64), 'zero_init')
    gs_random = _run_gs(
        project_zero_mean(rng.standard_normal(Ny).astype(np.float64)),
        'random_init',
    )

    # 8. Compare empirical rate vs rho(M)
    rate_comparison = {
        'rho_M': rho_M,
        'empirical_rate_zeros': gs_zeros['empirical_rate'],
        'empirical_rate_random': gs_random['empirical_rate'],
    }

    # 9. Lifting norm checks on reference potentials
    lift_check = lifting_norm_check(phi_ref, psi_ref, m, Nx, Ny)

    # Recover primal solution for completeness
    from src.regularized_ot import apply_K as _apply_K  # noqa: F401 (local import ok)
    from src.torus_utils import apply_G
    lifted_ref = lift(phi_ref, psi_ref, Nx, Ny)
    pi_ref = apply_G(lifted_ref - c, m) / eps
    p_marg, q_marg = marginals(pi_ref)

    converged = (
        gs_zeros['final_residual'] < tol * 1e4
        or gs_zeros['n_iters'] < max_iter
    )

    return {
        # Problem setup
        'Nx': Nx, 'Ny': Ny, 'eps': eps,
        'x': x, 'y': y, 'p': p, 'q': q,
        # Reference solution
        'z_ref': z_ref,
        'phi_ref': phi_ref,
        'psi_ref': psi_ref,
        'pi_ref': pi_ref,
        'ref_residual': ref_residual,
        'p_resid': float(np.max(np.abs(p_marg - p))),
        'q_resid': float(np.max(np.abs(q_marg - q))),
        # Block structure — top-level aliases for test compatibility
        'kxy_kyx_sym_residual': blocks['sym_KXY_KYX_residual'],
        'blocks': blocks,
        'eigs_KXX': eigs_KXX,
        'eigs_KYY': eigs_KYY,
        'min_eig_KXX': float(eigs_KXX[1]),   # skip near-zero null mode
        'min_eig_KYY': float(eigs_KYY[1]),
        # Error propagation
        'M': M,
        'eigs_M': eigs_M,
        'rho_M': rho_M,
        # GS runs
        'gs_zeros': gs_zeros,
        'gs_random': gs_random,
        'converged': converged,
        'rate_comparison': rate_comparison,
        # Lifting norm identities — top-level aliases for test compatibility
        'lifting_norm_check': lift_check,
        'lifting_norm_diff_phi': lift_check['diff_phi'],
        'lifting_norm_diff_psi': lift_check['diff_psi'],
    }

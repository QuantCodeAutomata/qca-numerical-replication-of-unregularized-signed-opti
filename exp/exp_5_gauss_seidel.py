"""
Experiment 5: Block Gauss-Seidel solver for the dual system K z = d.

The (Nx+Ny)×(Nx+Ny) dual operator K has a natural 2×2 block structure
  K = [[K_xx, K_xy], [K_yx, K_yy]]
and block Gauss-Seidel alternates between updating φ and ψ.  This
experiment analyzes the block properties, spectral radius of the error-
propagation operator, and convergence rate vs the direct solver.
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

_GS_AVAILABLE = True
_GS_ERROR = ''
try:
    from src.gauss_seidel import run_block_gauss_seidel_analysis, assemble_blocks
except ImportError as e:
    _GS_AVAILABLE = False
    _GS_ERROR = str(e)

RESULTS_DIR = '/workspace/project/results'


def _pf(condition: bool) -> str:
    return 'PASS' if condition else 'FAIL'


def _run_block_gs_inline(K_xx: np.ndarray, K_xy: np.ndarray, K_yx: np.ndarray,
                          K_yy: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                          tol: float = 1e-10, max_iter: int = 2000) -> dict:
    """
    Inline block Gauss-Seidel for K z = d.  Falls back when gauss_seidel.py missing.

    Block GS:
      K_xx φ_{k+1} = dx - K_xy ψ_k
      K_yy ψ_{k+1} = dy - K_yx φ_{k+1}
    """
    Nx = K_xx.shape[0]
    Ny = K_yy.shape[0]
    K_full = np.block([[K_xx, K_xy], [K_yx, K_yy]])
    d_full = np.concatenate([dx, dy])

    # Regularize degenerate blocks (zero-mean subspace)
    def safe_solve(A, b):
        eigvals, eigvecs = np.linalg.eigh(A)
        thr = 1e-12 * np.max(np.abs(eigvals))
        active = np.abs(eigvals) > thr
        Vt_b = eigvecs[:, active].T @ b           # (n_active,)
        coeffs_act = Vt_b / eigvals[active]        # (n_active,)
        return eigvecs[:, active] @ coeffs_act

    phi = np.zeros(Nx)
    psi = np.zeros(Ny)
    residuals = []
    iters = 0

    for k in range(max_iter):
        phi = safe_solve(K_xx, dx - K_xy @ psi)
        phi = project_zero_mean(phi)
        psi = safe_solve(K_yy, dy - K_yx @ phi)
        psi = project_zero_mean(psi)
        z = np.concatenate([phi, psi])
        resid = float(np.linalg.norm(K_full @ z - d_full))
        residuals.append(resid)
        iters += 1
        if resid < tol:
            break

    return {
        'z': np.concatenate([phi, psi]),
        'residual_history': residuals,
        'iterations': iters,
        'final_residual': residuals[-1] if residuals else float('nan'),
        'converged': (residuals[-1] < tol) if residuals else False,
    }


def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n=== Exp 5: Block Gauss-Seidel Analysis ===\n")

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    Nx, Ny = 32, 32
    eps = 0.1
    x, y, p, q, c = make_synthetic_problem(Nx, Ny, marginal_type='default', cost_type='default')
    m = make_laplacian_multiplier(Nx, Ny)

    K_mat = build_K_matrix(Nx, Ny, m)
    d_vec = compute_rhs(p, q, c, m, eps)

    # Extract blocks
    K_xx = K_mat[:Nx, :Nx]
    K_xy = K_mat[:Nx, Nx:]
    K_yx = K_mat[Nx:, :Nx]
    K_yy = K_mat[Nx:, Nx:]
    dx = d_vec[:Nx]
    dy = d_vec[Nx:]

    print(f"K_xx shape: {K_xx.shape},  K_yy shape: {K_yy.shape}")

    # ------------------------------------------------------------------
    # 2. Block matrix properties
    # ------------------------------------------------------------------
    sym_resid_xx = float(np.max(np.abs(K_xx - K_xx.T)))
    sym_resid_yy = float(np.max(np.abs(K_yy - K_yy.T)))
    sym_resid_xy = float(np.max(np.abs(K_xy - K_yx.T)))

    eigvals_xx = np.linalg.eigvalsh(K_xx)
    eigvals_yy = np.linalg.eigvalsh(K_yy)
    min_eig_xx = float(np.min(eigvals_xx[eigvals_xx > 1e-12]))
    min_eig_yy = float(np.min(eigvals_yy[eigvals_yy > 1e-12]))

    print("Block matrix properties:")
    print(f"  sym residual K_xx : {sym_resid_xx:.2e}  {_pf(sym_resid_xx < 1e-10)}")
    print(f"  sym residual K_yy : {sym_resid_yy:.2e}  {_pf(sym_resid_yy < 1e-10)}")
    print(f"  sym residual K_xy : {sym_resid_xy:.2e}  (K_xy = K_yx^T)")
    print(f"  min eig K_xx (>0) : {min_eig_xx:.4e}  {_pf(min_eig_xx > 0)}")
    print(f"  min eig K_yy (>0) : {min_eig_yy:.4e}  {_pf(min_eig_yy > 0)}")

    # ------------------------------------------------------------------
    # 3. Spectral radius of error propagation operator M = (D+L)^{-1} U
    #    D = block-diag(K_xx, K_yy), L = [[0,0],[K_yx,0]], U = [[0,K_xy],[0,0]]
    # ------------------------------------------------------------------
    def block_diag_inv_solve(D_inv_xx, D_inv_yy, v):
        """Apply (D+L)^{-1} to v using forward substitution."""
        vx = v[:Nx]
        vy = v[Nx:]
        phi_out = D_inv_xx @ vx
        psi_out = D_inv_yy @ (vy - K_yx @ phi_out)
        return np.concatenate([phi_out, psi_out])

    def pinv_safe(A):
        eigvals, eigvecs = np.linalg.eigh(A)
        thr = 1e-12 * np.max(np.abs(eigvals))
        inv_eigvals = np.where(np.abs(eigvals) > thr, 1.0 / eigvals, 0.0)
        return eigvecs @ np.diag(inv_eigvals) @ eigvecs.T

    Kxx_pinv = pinv_safe(K_xx)
    Kyy_pinv = pinv_safe(K_yy)

    U = np.block([[np.zeros((Nx, Nx)), K_xy],
                   [np.zeros((Ny, Nx)), np.zeros((Ny, Ny))]])

    # M = (D+L)^{-1} U applied to each standard basis vector
    dim = Nx + Ny
    M_mat = np.zeros((dim, dim))
    for j in range(dim):
        ej = np.zeros(dim)
        ej[j] = 1.0
        Uej = U @ ej
        M_mat[:, j] = block_diag_inv_solve(Kxx_pinv, Kyy_pinv, Uej)

    eigvals_M = np.linalg.eigvals(M_mat)
    rho_M = float(np.max(np.abs(eigvals_M)))
    print(f"\nSpectral radius ρ(M) = {rho_M:.6f}  {_pf(rho_M < 1.0)}")

    # ------------------------------------------------------------------
    # 4. Direct solve reference
    # ------------------------------------------------------------------
    eigvals_K, eigvecs_K = np.linalg.eigh(K_mat)
    thr = 1e-12 * np.max(np.abs(eigvals_K))
    active = np.abs(eigvals_K) > thr
    Vt_d = eigvecs_K[:, active].T @ d_vec      # (n_active,)
    coeffs = Vt_d / eigvals_K[active]           # (n_active,)
    z_direct = eigvecs_K[:, active] @ coeffs    # (dim,)
    resid_direct = float(np.linalg.norm(K_mat @ z_direct - d_vec))
    print(f"\nDirect solve residual  : {resid_direct:.4e}")

    # ------------------------------------------------------------------
    # 5. Gauss-Seidel convergence (from src or inline fallback)
    # ------------------------------------------------------------------
    if _GS_AVAILABLE:
        # run_block_gauss_seidel_analysis builds its own synthetic problem
        # internally, so we only pass grid sizes and solver parameters.
        gs_analysis = run_block_gauss_seidel_analysis(
            Nx=Nx, Ny=Ny, eps=eps, max_iter=2000, tol=1e-10
        )

        # Zero-init run is the primary reference
        gs_run = gs_analysis['gs_zeros']
        gs_residuals = list(gs_run['residuals'])
        gs_iters = gs_run['n_iters']
        gs_final = gs_run['final_residual']
        gs_converged = gs_final < 1e-10

        # Lifting-norm identity
        lc = gs_analysis['lifting_norm_check']
        lifting_pass = lc['diff_phi'] < 1e-10 and lc['diff_psi'] < 1e-10
        print(f"\n  lifting norm diff φ : {lc['diff_phi']:.2e}  {_pf(lc['diff_phi'] < 1e-10)}")
        print(f"  lifting norm diff ψ : {lc['diff_psi']:.2e}  {_pf(lc['diff_psi'] < 1e-10)}")

        # Empirical vs theoretical rate
        emp_rate_zeros = gs_analysis['rate_comparison']['empirical_rate_zeros']
        emp_rate_rand = gs_analysis['rate_comparison']['empirical_rate_random']
        print(f"\nRate comparison:")
        print(f"  theoretical ρ(M)       : {gs_analysis['rho_M']:.6f}")
        print(f"  empirical (zero init)  : {emp_rate_zeros:.6f}")
        print(f"  empirical (rand init)  : {emp_rate_rand:.6f}")

        # Also grab the random-init convergence curve for the plot
        gs_run_rand = gs_analysis['gs_random']
        gs_residuals_rand = list(gs_run_rand['residuals'])
    else:
        print(f"\n[WARNING] src/gauss_seidel.py not available: {_GS_ERROR}")
        print("Using inline fallback for Gauss-Seidel convergence.")
        gs_result = _run_block_gs_inline(K_xx, K_xy, K_yx, K_yy, dx, dy,
                                         tol=1e-10, max_iter=2000)
        gs_residuals = gs_result['residual_history']
        gs_iters = gs_result['iterations']
        gs_final = gs_result['final_residual']
        gs_converged = gs_result['converged']
        lifting_pass = None
        gs_residuals_rand = []
        emp_rate_zeros = float('nan')

    print(f"\nBlock Gauss-Seidel convergence (zero init):")
    print(f"  iterations     : {gs_iters}")
    print(f"  final residual : {gs_final:.4e}")
    print(f"  converged      : {_pf(gs_converged)}")
    if lifting_pass is not None:
        print(f"  lifting norm id: {_pf(lifting_pass)}")

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    if gs_residuals:
        ax.semilogy(gs_residuals, color='tab:blue', linewidth=1.5,
                    label=f'Block GS zero-init (ρ(M)={rho_M:.4f})')
    if gs_residuals_rand:
        ax.semilogy(gs_residuals_rand, color='tab:orange', linewidth=1.5,
                    linestyle='--', label='Block GS rand-init')
    ax.axhline(resid_direct, color='tab:green', linestyle='--', linewidth=1.5,
               label=f'Direct solve ({resid_direct:.1e})')
    ax.axhline(1e-10, color='grey', linestyle=':', linewidth=1, label='tol=1e-10')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('‖Kz - d‖')
    ax.set_title(f'Block Gauss-Seidel convergence  (Nx=Ny={Nx}, ε={eps})')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp5_gauss_seidel_convergence.png'), dpi=100)
    plt.close(fig)

    print(f"\nPlot saved to {RESULTS_DIR}/")

    # ------------------------------------------------------------------
    # 7. Return metrics
    # ------------------------------------------------------------------
    metrics = {
        'gs_available': _GS_AVAILABLE,
        'sym_resid_xx': sym_resid_xx,
        'sym_resid_yy': sym_resid_yy,
        'min_eig_xx': min_eig_xx,
        'min_eig_yy': min_eig_yy,
        'rho_M': rho_M,
        'gs_converged': gs_converged,
        'gs_iters': gs_iters,
        'gs_final_resid': gs_final,
        'lifting_pass': lifting_pass,
        'empirical_rate': emp_rate_zeros,
    }
    return metrics


if __name__ == '__main__':
    main()

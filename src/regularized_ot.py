"""
Regularized signed optimal transport solver on compact tori.

Implements the framework from Experiment 2:
  min_{pi: A pi = b}  <c, pi> + eps * H(pi)
where H(pi) = 0.5 * <pi, L pi> is the Dirichlet energy and L is the torus Laplacian.

Dual operator equation: K z = d
  K = A G A*,   d = eps * b + A G c,   G = L^{-1} on nonzero modes.
Primal recovery: pi* = (1/eps) * G (A* z - c).

All operators live on the zero-mean (zero-mass) subspace enforced by setting
G_hat(0,0) = 0 and projecting potentials to zero mean.

# Custom — Context7 found no library equivalent (paper regularized signed OT framework)
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.linalg import solve as dense_solve
from scipy.sparse.linalg import LinearOperator, cg

from src.torus_utils import (
    make_torus_grids,
    make_laplacian_multiplier,
    apply_G,
    apply_G_hat,
    marginals,
    lift,
    project_zero_mean,
    inner_product,
    dirichlet_energy,
)


# ---------------------------------------------------------------------------
# Dual operator K = A G A* (matrix-free)
# ---------------------------------------------------------------------------

def apply_K(
    phi: np.ndarray,
    psi: np.ndarray,
    m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the dual operator K = A G A* to potentials (phi, psi).

    Steps:
      1. Lift: f = A*(phi, psi) = phi(x) + psi(y)
      2. Apply G in Fourier domain: Gf = G f
      3. Take marginals: (phi_out, psi_out) = A (Gf)

    Parameters
    ----------
    phi : (Nx,) zero-mean potential on X.
    psi : (Ny,) zero-mean potential on Y.
    m   : (Nx, Ny) Laplacian multiplier.

    Returns
    -------
    phi_out : (Nx,)
    psi_out : (Ny,)
    """
    Nx = phi.shape[0]
    Ny = psi.shape[0]
    f = lift(phi, psi, Nx, Ny)            # (Nx, Ny)
    Gf = apply_G(f, m)                    # (Nx, Ny)
    phi_out, psi_out = marginals(Gf)      # (Nx,), (Ny,)
    return phi_out, psi_out


def build_K_matrix(Nx: int, Ny: int, m: np.ndarray) -> np.ndarray:
    """
    Explicitly assemble the K matrix by applying K to each basis vector.

    The state vector z concatenates phi (Nx elements) and psi (Ny elements).
    Both are constrained to zero mean; the matrix is (Nx+Ny) x (Nx+Ny) but
    operates on the zero-mean subspace.

    Parameters
    ----------
    Nx, Ny : int
    m      : (Nx, Ny) Laplacian multiplier.

    Returns
    -------
    K_mat : (Nx + Ny, Nx + Ny) float64 symmetric PSD matrix on the zero-mean subspace.
    """
    dim = Nx + Ny
    K_mat = np.zeros((dim, dim), dtype=np.float64)
    for col in range(dim):
        e = np.zeros(dim, dtype=np.float64)
        e[col] = 1.0
        phi_e = e[:Nx]
        psi_e = e[Nx:]
        # Project to zero mean before applying K
        phi_e = project_zero_mean(phi_e)
        psi_e = project_zero_mean(psi_e)
        phi_out, psi_out = apply_K(phi_e, psi_e, m)
        K_mat[:Nx, col] = phi_out
        K_mat[Nx:, col] = psi_out
    return K_mat


def make_K_linear_operator(Nx: int, Ny: int, m: np.ndarray) -> LinearOperator:
    """
    Return a scipy LinearOperator representing K on the concatenated [phi; psi] vector.

    Parameters
    ----------
    Nx, Ny : int
    m      : (Nx, Ny) Laplacian multiplier.

    Returns
    -------
    K_op : scipy LinearOperator of shape (Nx+Ny, Nx+Ny).
    """
    dim = Nx + Ny

    def matvec(z: np.ndarray) -> np.ndarray:
        phi = project_zero_mean(z[:Nx])
        psi = project_zero_mean(z[Nx:])
        phi_out, psi_out = apply_K(phi, psi, m)
        return np.concatenate([phi_out, psi_out])

    return LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)


# ---------------------------------------------------------------------------
# RHS construction
# ---------------------------------------------------------------------------

def compute_AGc(c: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute A G c: apply G to c then take marginals.

    Parameters
    ----------
    c : (Nx, Ny) cost kernel.
    m : (Nx, Ny) Laplacian multiplier.

    Returns
    -------
    AGc_x : (Nx,) x-marginal of G c.
    AGc_y : (Ny,) y-marginal of G c.
    """
    Gc = apply_G(c, m)
    return marginals(Gc)


def compute_rhs(
    p: np.ndarray,
    q: np.ndarray,
    c: np.ndarray,
    m: np.ndarray,
    eps: float,
) -> np.ndarray:
    """
    Compute the right-hand side d = eps * (p, q) + A G c.

    Parameters
    ----------
    p   : (Nx,) zero-mean x-marginal.
    q   : (Ny,) zero-mean y-marginal.
    c   : (Nx, Ny) cost kernel.
    m   : (Nx, Ny) Laplacian multiplier.
    eps : float  regularization strength.

    Returns
    -------
    d : (Nx + Ny,) concatenated RHS vector.
    """
    AGc_x, AGc_y = compute_AGc(c, m)
    d = eps * np.concatenate([p, q]) + np.concatenate([AGc_x, AGc_y])
    return d


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_regularized_ot(
    p: np.ndarray,
    q: np.ndarray,
    c: np.ndarray,
    eps: float,
    Nx: int,
    Ny: int,
    method: str = 'direct',
    tol: float = 1e-12,
    max_iter: int = 10000,
) -> dict:
    """
    Solve the regularized signed OT problem on the compact torus.

    Dual system: K z = d, then primal: pi* = (1/eps) G (A* z - c).

    Parameters
    ----------
    p      : (Nx,) zero-mean signed marginal on X.
    q      : (Ny,) zero-mean signed marginal on Y.
    c      : (Nx, Ny) cost kernel.
    eps    : float  regularization strength (> 0).
    Nx, Ny : int  grid sizes.
    method : 'direct' (assemble K and solve) or 'cg' (conjugate gradient).
    tol    : float  solver tolerance (for CG).
    max_iter : int  max CG iterations.

    Returns
    -------
    dict with keys:
      'pi_star' : (Nx, Ny) optimal signed transport plan.
      'phi_star': (Nx,) dual potential on X.
      'psi_star': (Ny,) dual potential on Y.
      'z'       : (Nx+Ny,) dual solution vector.
      'm'       : (Nx, Ny) Laplacian multiplier.
      'objective': float  J_eps(pi*).
      'cost_term': float  <c, pi*>.
      'energy'  : float  eps * H(pi*).
      'p_resid' : float  marginal residual for x.
      'q_resid' : float  marginal residual for y.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    m = make_laplacian_multiplier(Nx, Ny)
    d = compute_rhs(p, q, c, m, eps)

    if method == 'direct':
        K_mat = build_K_matrix(Nx, Ny, m)
        z = dense_solve(K_mat, d)
    elif method == 'cg':
        K_op = make_K_linear_operator(Nx, Ny, m)
        z, info = cg(K_op, d, rtol=tol, maxiter=max_iter)
        if info != 0:
            import warnings
            warnings.warn(f"CG did not converge (info={info})")
    else:
        raise ValueError(f"Unknown method: {method}")

    phi_star = project_zero_mean(z[:Nx])
    psi_star = project_zero_mean(z[Nx:])

    # Primal recovery: pi* = (1/eps) G (A* z - c)
    lifted = lift(phi_star, psi_star, Nx, Ny)
    pi_star = apply_G(lifted - c, m) / eps

    # Diagnostics
    p_marg, q_marg = marginals(pi_star)
    p_resid = float(np.max(np.abs(p_marg - p)))
    q_resid = float(np.max(np.abs(q_marg - q)))

    cost_term = inner_product(c, pi_star)
    energy = eps * dirichlet_energy(pi_star, m)
    objective = cost_term + energy

    return {
        'pi_star' : pi_star,
        'phi_star': phi_star,
        'psi_star': psi_star,
        'z'       : z,
        'm'       : m,
        'objective': objective,
        'cost_term': cost_term,
        'energy'  : energy,
        'p_resid' : p_resid,
        'q_resid' : q_resid,
    }


# ---------------------------------------------------------------------------
# Standard synthetic problem setup
# ---------------------------------------------------------------------------

def make_synthetic_problem(
    Nx: int = 64,
    Ny: int = 64,
    marginal_type: str = 'default',
    cost_type: str = 'default',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a canonical synthetic torus problem.

    Parameters
    ----------
    Nx, Ny       : int  grid sizes.
    marginal_type: 'default' or 'exp3' or 'exp6'.
    cost_type    : 'default' or 'exp3' or 'exp6'.

    Returns
    -------
    (x, y, p, q, c)
    """
    x, y = make_torus_grids(Nx, Ny)

    if marginal_type == 'default':
        p = np.sin(2 * np.pi * x) + 0.4 * np.sin(4 * np.pi * x)
        q = np.cos(2 * np.pi * y) - 0.25 * np.cos(6 * np.pi * y)
    elif marginal_type == 'exp3':
        p = np.sin(2 * np.pi * x) + 0.2 * np.cos(6 * np.pi * x)
        q = np.cos(2 * np.pi * y) - 0.3 * np.sin(4 * np.pi * y)
    elif marginal_type == 'exp6':
        p = (np.sin(2 * np.pi * x) + 0.5 * np.sin(8 * np.pi * x)
             - 0.3 * np.cos(12 * np.pi * x))
        q = (np.cos(2 * np.pi * y) - 0.4 * np.sin(6 * np.pi * y)
             + 0.2 * np.cos(10 * np.pi * y))
    else:
        raise ValueError(f"Unknown marginal_type: {marginal_type}")

    # Enforce zero-mean
    p = project_zero_mean(p)
    q = project_zero_mean(q)

    X, Y = np.meshgrid(x, y, indexing='ij')

    if cost_type == 'default':
        c = 0.8 * np.cos(2 * np.pi * (X - Y)) + 0.5 * np.sin(2 * np.pi * X) * np.cos(4 * np.pi * Y)
    elif cost_type == 'exp3':
        c = np.cos(2 * np.pi * (X - Y)) + 0.2 * np.sin(6 * np.pi * X) * np.sin(2 * np.pi * Y)
    elif cost_type == 'exp6':
        c = (np.cos(2 * np.pi * (X - Y))
             + 0.5 * np.sin(10 * np.pi * X) * np.sin(8 * np.pi * Y)
             + 0.2 * np.cos(14 * np.pi * (X + Y)))
    else:
        raise ValueError(f"Unknown cost_type: {cost_type}")

    return x, y, p, q, c

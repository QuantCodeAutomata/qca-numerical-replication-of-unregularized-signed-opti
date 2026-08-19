"""
Shared torus discretization utilities for signed optimal transport experiments.

All experiments use a compact torus domain X = Y = T discretized on uniform grids.
The discrete Haar measure is approximated by uniform weights: each point has weight 1/N.
All functions use float64 / complex128 precision throughout.
"""

from typing import Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def make_torus_grids(Nx: int, Ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct uniform grids on T = [0, 1) for X and Y.

    Parameters
    ----------
    Nx : int  Number of grid points on X.
    Ny : int  Number of grid points on Y.

    Returns
    -------
    x : (Nx,) grid  x_i = i / Nx
    y : (Ny,) grid  y_j = j / Ny
    """
    x = np.arange(Nx, dtype=np.float64) / Nx
    y = np.arange(Ny, dtype=np.float64) / Ny
    return x, y


def make_freq_grids(Nx: int, Ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return integer FFT frequency arrays (not normalized by N).

    Uses numpy.fft.fftfreq * N to produce integer wavenumbers k, l.

    Parameters
    ----------
    Nx, Ny : int   Grid sizes.

    Returns
    -------
    kx : (Nx,) integer frequencies for X
    ky : (Ny,) integer frequencies for Y
    """
    kx = np.fft.fftfreq(Nx) * Nx   # integers: 0, 1, ..., Nx//2-1, -Nx//2, ..., -1
    ky = np.fft.fftfreq(Ny) * Ny
    return kx.astype(np.float64), ky.astype(np.float64)


def make_laplacian_multiplier(Nx: int, Ny: int) -> np.ndarray:
    """
    Build the joint Laplacian Fourier multiplier m(k, l) = |k|^2 + |l|^2 on T^2.

    The torus Laplacian on T has multipliers m_X(k) = |k|^2 in the FFT ordering.

    Parameters
    ----------
    Nx, Ny : int  Grid sizes.

    Returns
    -------
    m : (Nx, Ny) float64 array  m[i, j] = kx[i]^2 + ky[j]^2
    """
    kx, ky = make_freq_grids(Nx, Ny)
    m = kx[:, None] ** 2 + ky[None, :] ** 2     # shape (Nx, Ny)
    return m


# ---------------------------------------------------------------------------
# Green operator (inverse Laplacian on zero-mean subspace)
# ---------------------------------------------------------------------------

def apply_G(f: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Apply the Green operator G = L^{-1} to f in physical space.

    G is defined in Fourier space by G_hat(k, l) = 1 / m(k, l) for (k,l) != (0,0)
    and G_hat(0, 0) = 0 (projects to zero-mean subspace).

    Parameters
    ----------
    f : (Nx, Ny) float64 array in physical space.
    m : (Nx, Ny) float64  Laplacian multiplier from make_laplacian_multiplier().

    Returns
    -------
    Gf : (Nx, Ny) float64 array.
    """
    f_hat = np.fft.fft2(f)
    with np.errstate(divide='ignore', invalid='ignore'):
        G_hat = np.where(m == 0, 0.0, 1.0 / m)
    Gf_hat = G_hat * f_hat
    return np.real(np.fft.ifft2(Gf_hat))


def apply_G_hat(f_hat: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Apply the Green operator G to data already in Fourier space.

    Parameters
    ----------
    f_hat : (Nx, Ny) complex128  Fourier transform of f.
    m : (Nx, Ny) float64  Laplacian multiplier.

    Returns
    -------
    Gf_hat : (Nx, Ny) complex128.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        G_hat = np.where(m == 0, 0.0, 1.0 / m)
    return G_hat * f_hat


def apply_L(f: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Apply the Laplacian operator L to f: L_hat(k, l) = m(k, l) * f_hat(k, l).

    Parameters
    ----------
    f : (Nx, Ny) float64 array in physical space.
    m : (Nx, Ny) float64  Laplacian multiplier.

    Returns
    -------
    Lf : (Nx, Ny) float64 array.
    """
    f_hat = np.fft.fft2(f)
    Lf_hat = m * f_hat
    return np.real(np.fft.ifft2(Lf_hat))


# ---------------------------------------------------------------------------
# Marginal operators (A and A*)
# ---------------------------------------------------------------------------

def marginals(pi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute x- and y-marginals of a transport plan under normalized Haar measure.

    P_X pi = mean over axis 1 (average over y for each x)
    P_Y pi = mean over axis 0 (average over x for each y)

    Parameters
    ----------
    pi : (Nx, Ny) array.

    Returns
    -------
    p_marg : (Nx,)  x-marginal
    q_marg : (Ny,)  y-marginal
    """
    p_marg = np.mean(pi, axis=1)   # average over y
    q_marg = np.mean(pi, axis=0)   # average over x
    return p_marg, q_marg


def lift(phi: np.ndarray, psi: np.ndarray, Nx: int, Ny: int) -> np.ndarray:
    """
    Lift dual potentials (phi, psi) to the product grid: A*(phi, psi) = phi(x) + psi(y).

    Parameters
    ----------
    phi : (Nx,)  potential on X.
    psi : (Ny,)  potential on Y.
    Nx, Ny : int  Grid sizes.

    Returns
    -------
    lifted : (Nx, Ny) float64  phi[i] + psi[j] for all i, j.
    """
    return phi[:, None] + psi[None, :]


# ---------------------------------------------------------------------------
# Zero-mean projection
# ---------------------------------------------------------------------------

def project_zero_mean(f: np.ndarray) -> np.ndarray:
    """Subtract the sample mean to enforce zero-mean constraint."""
    return f - np.mean(f)


# ---------------------------------------------------------------------------
# Feasibility checker
# ---------------------------------------------------------------------------

def check_feasibility(
    pi: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    tol: float = 1e-10,
) -> dict:
    """
    Check feasibility of a signed coupling pi with marginals p and q.

    Returns a dict with:
      - 'p_resid': max-norm residual of x-marginal vs p
      - 'q_resid': max-norm residual of y-marginal vs q
      - 'mass'  : total mass of pi (should be 0 for signed OT with zero-mean marginals)
      - 'feasible': bool  all residuals < tol
    """
    p_marg, q_marg = marginals(pi)
    p_resid = np.max(np.abs(p_marg - p))
    q_resid = np.max(np.abs(q_marg - q))
    mass = np.mean(pi)      # total mass under normalized measure
    feasible = (p_resid < tol) and (q_resid < tol)
    return {
        'p_resid': float(p_resid),
        'q_resid': float(q_resid),
        'mass':    float(mass),
        'feasible': feasible,
    }


# ---------------------------------------------------------------------------
# Inner products
# ---------------------------------------------------------------------------

def inner_product(f: np.ndarray, g: np.ndarray) -> float:
    """
    Discrete L2 inner product under normalized Haar measure: <f, g> = mean(f * g).

    Parameters
    ----------
    f, g : (Nx, Ny) or (N,) real arrays of the same shape.

    Returns
    -------
    float  inner product value.
    """
    return float(np.mean(f * g))


def parseval_inner(f_hat: np.ndarray, g_hat: np.ndarray, N_total: int) -> float:
    """
    Parseval inner product in Fourier domain:  <f, g> = (1/N) sum f_hat * conj(g_hat).

    Note: for normalized Haar measure (each point weight 1/N), the Parseval relation is
    <f, g> = (1/N^2) sum f_hat conj(g_hat) where f_hat = fft2(f) (unnormalized FFT).

    Parameters
    ----------
    f_hat, g_hat : (Nx, Ny) complex128  Unnormalized FFT outputs.
    N_total : int  Nx * Ny.

    Returns
    -------
    float
    """
    return float(np.real(np.sum(f_hat * np.conj(g_hat))) / N_total ** 2)


# ---------------------------------------------------------------------------
# Dirichlet energy
# ---------------------------------------------------------------------------

def dirichlet_energy(pi: np.ndarray, m: np.ndarray) -> float:
    """
    Compute the Dirichlet energy H(pi) = 0.5 * <pi, L pi> under normalized Haar measure.

    Fourier domain formula:
      H(pi) = 0.5 * (1/N^2) * sum_kl m(k,l) |pi_hat(k,l)|^2

    Parameters
    ----------
    pi  : (Nx, Ny) float64 in physical space.
    m   : (Nx, Ny) float64 Laplacian multiplier.

    Returns
    -------
    float
    """
    Nx, Ny = pi.shape
    pi_hat = np.fft.fft2(pi)
    N2 = (Nx * Ny) ** 2
    return 0.5 * float(np.sum(m * np.abs(pi_hat) ** 2) / N2)


def regularized_objective(
    pi: np.ndarray, c: np.ndarray, m: np.ndarray, eps: float
) -> Tuple[float, float, float]:
    """
    Compute J_eps(pi) = <c, pi> + eps * H(pi).

    Returns
    -------
    (J_eps, cost_term, energy_term)
    """
    cost_term = inner_product(c, pi)
    energy_term = eps * dirichlet_energy(pi, m)
    return cost_term + energy_term, cost_term, energy_term

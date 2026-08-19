"""
Euler-Lagrange conditions and strong duality verification (Experiment 3).

Validates the variational characterization of the regularized signed OT minimizer.
The optimal pi* satisfies: c + eps * L pi* = phi*(x) + psi*(y)  (up to a constant),
which is the first-order stationarity condition of the regularized Lagrangian.
"""

from typing import List

import numpy as np

from src.torus_utils import (
    apply_L,
    dirichlet_energy,
    inner_product,
    lift,
    make_laplacian_multiplier,
    make_torus_grids,
    marginals,
    project_zero_mean,
)
from src.regularized_ot import make_synthetic_problem, solve_regularized_ot


# ---------------------------------------------------------------------------
# Laplacian applied to pi
# ---------------------------------------------------------------------------

def apply_L_pi(pi: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Apply Laplacian L to pi in physical space using FFT.

    Parameters
    ----------
    pi : (Nx, Ny) float64 transport plan in physical space.
    m  : (Nx, Ny) float64 Laplacian multiplier from make_laplacian_multiplier().

    Returns
    -------
    Lpi : (Nx, Ny) float64  L pi in physical space.
    """
    pi_hat = np.fft.fft2(pi)
    return np.real(np.fft.ifft2(m * pi_hat))


# ---------------------------------------------------------------------------
# Local weak Euler-Lagrange residual
# ---------------------------------------------------------------------------

def local_el_residual(
    pi_star: np.ndarray,
    c: np.ndarray,
    m: np.ndarray,
    eps: float,
    test_functions: list,
) -> list:
    """
    Local weak EL test: for each h in ker(A) (zero-marginal functions),
    compute R_local(h) = <c, h> + eps * <L pi*, h>.

    The energy inner product uses the Fourier-domain Parseval formula:
      eps * <L pi*, h> = eps * Re[ sum_kl m(k,l) pi*_hat(k,l) conj(h_hat(k,l)) ] / (Nx*Ny)^2

    At the optimum pi* this residual is zero for all h in ker(A).

    Parameters
    ----------
    pi_star        : (Nx, Ny) optimal transport plan.
    c              : (Nx, Ny) cost kernel.
    m              : (Nx, Ny) Laplacian multiplier.
    eps            : float regularization strength.
    test_functions : list of (Nx, Ny) arrays with zero marginals.

    Returns
    -------
    list of dicts, each with keys 'h_norm', 'residual', 'rel_residual'.
    """
    Nx, Ny = pi_star.shape
    N2 = float(Nx * Ny) ** 2
    pi_hat = np.fft.fft2(pi_star)

    # Precompute the EL field for normalization: c + eps * L pi*
    el_field = c + eps * apply_L_pi(pi_star, m)
    el_norm = float(np.sqrt(np.mean(el_field ** 2)))

    results = []
    for h in test_functions:
        h_hat = np.fft.fft2(h)
        cost_term = inner_product(c, h)
        energy_term = eps * float(np.real(np.sum(m * pi_hat * np.conj(h_hat))) / N2)
        residual = cost_term + energy_term
        h_norm = float(np.sqrt(np.mean(h ** 2)))
        denom = el_norm * h_norm
        rel_residual = abs(residual) / denom if denom > 0 else 0.0
        results.append({
            'h_norm': h_norm,
            'residual': float(residual),
            'rel_residual': float(rel_residual),
        })
    return results


# ---------------------------------------------------------------------------
# Global weak Euler-Lagrange residual
# ---------------------------------------------------------------------------

def global_el_residual(
    pi_star: np.ndarray,
    phi_star: np.ndarray,
    psi_star: np.ndarray,
    c: np.ndarray,
    m: np.ndarray,
    eps: float,
    test_functions: list,
) -> list:
    """
    Global weak EL test: for each zero-mean h, compute
      R_global(h) = <c + eps * L pi* - phi*(x) - psi*(y), h>.

    The EL condition states c + eps * L pi* = phi*(x) + psi*(y) up to a constant,
    so this residual is zero for any zero-mean h.

    Parameters
    ----------
    pi_star        : (Nx, Ny) optimal transport plan.
    phi_star       : (Nx,) dual potential on X (zero-mean).
    psi_star       : (Ny,) dual potential on Y (zero-mean).
    c              : (Nx, Ny) cost kernel.
    m              : (Nx, Ny) Laplacian multiplier.
    eps            : float regularization strength.
    test_functions : list of (Nx, Ny) zero-mean test arrays.

    Returns
    -------
    list of dicts, each with keys 'h_norm', 'residual', 'rel_residual'.
    """
    Nx, Ny = pi_star.shape
    Lpi = apply_L_pi(pi_star, m)
    lifted = lift(phi_star, psi_star, Nx, Ny)
    # EL field: S = c + eps * L pi* - phi*(x) - psi*(y), should be ~constant (≈ 0)
    S = c + eps * Lpi - lifted
    # Normalize by the cost scale so rel_residual measures the fraction of the
    # cost magnitude that leaks into the global EL condition.
    rms_c = float(np.sqrt(np.mean(c ** 2)))

    results = []
    for h in test_functions:
        residual = inner_product(S, h)
        h_norm = float(np.sqrt(np.mean(h ** 2)))
        denom = rms_c * h_norm
        rel_residual = abs(residual) / denom if denom > 0 else 0.0
        results.append({
            'h_norm': h_norm,
            'residual': float(residual),
            'rel_residual': float(rel_residual),
        })
    return results


# ---------------------------------------------------------------------------
# Strong Euler-Lagrange residual
# ---------------------------------------------------------------------------

def strong_el_residual(
    pi_star: np.ndarray,
    phi_star: np.ndarray,
    psi_star: np.ndarray,
    c: np.ndarray,
    m: np.ndarray,
    eps: float,
    Nx: int,
    Ny: int,
) -> dict:
    """
    Strong EL test: S(x,y) = c(x,y) + eps * L pi*(x,y) - phi*(x) - psi*(y)
    should be constant (approximately zero after zero-mean projection).

    Returns
    -------
    dict with keys:
      'S'           : (Nx, Ny) EL field.
      'C'           : float sample mean of S (the constant value).
      'residual'    : float  ||S - C||_2 under normalized Haar measure.
      'rel_residual': float  residual / rms(c).
    """
    Lpi = apply_L_pi(pi_star, m)
    lifted = lift(phi_star, psi_star, Nx, Ny)
    S = c + eps * Lpi - lifted
    C = float(np.mean(S))
    residual = float(np.sqrt(np.mean((S - C) ** 2)))
    rms_c = float(np.sqrt(np.mean(c ** 2)))
    rel_residual = residual / rms_c if rms_c > 0 else float(residual)
    return {
        'S': S,
        'C': C,
        'residual': residual,
        'rel_residual': rel_residual,
    }


# ---------------------------------------------------------------------------
# Lagrangian
# ---------------------------------------------------------------------------

def lagrangian(
    pi: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    c: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    m: np.ndarray,
    eps: float,
    Nx: int,
    Ny: int,
) -> float:
    """
    Compute L(pi, phi, psi) = <c, pi> + eps*H(pi) + <phi, p - P_X pi> + <psi, q - P_Y pi>

    where <phi, f>_X = mean_X(phi * f) under normalized Haar measure on X, and
    H(pi) = 0.5 * <pi, L pi> is the Dirichlet energy.

    Parameters
    ----------
    pi   : (Nx, Ny) transport plan.
    phi  : (Nx,) dual potential on X.
    psi  : (Ny,) dual potential on Y.
    c    : (Nx, Ny) cost kernel.
    p    : (Nx,) x-marginal target.
    q    : (Ny,) y-marginal target.
    m    : (Nx, Ny) Laplacian multiplier.
    eps  : float regularization strength.
    Nx, Ny : int grid sizes.

    Returns
    -------
    float Lagrangian value.
    """
    cost = inner_product(c, pi)
    energy = eps * dirichlet_energy(pi, m)
    p_marg, q_marg = marginals(pi)
    phi_term = float(np.mean(phi * (p - p_marg)))
    psi_term = float(np.mean(psi * (q - q_marg)))
    return cost + energy + phi_term + psi_term


# ---------------------------------------------------------------------------
# Saddle-point check
# ---------------------------------------------------------------------------

def saddle_point_check(
    pi_star: np.ndarray,
    phi_star: np.ndarray,
    psi_star: np.ndarray,
    c: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    m: np.ndarray,
    eps: float,
    Nx: int,
    Ny: int,
    n_random: int = 20,
    delta: float = 0.1,
    rng: np.random.Generator = None,
) -> dict:
    """
    Saddle-point verification at (pi*, phi*, psi*).

    Dual perturbation: L is linear in (phi, psi), so
      L(pi*, phi* + du, psi* + dv) - L(pi*, phi*, psi*)
        = <du, p - P_X pi*> + <dv, q - P_Y pi*>
    which is near zero when pi* is feasible.

    Primal perturbation: pi* unconstrained-minimizes L(·, phi*, psi*), so
      L(pi* + delta*w, phi*, psi*) >= L(pi*, phi*, psi*)
    for any perturbation w.

    Parameters
    ----------
    pi_star, phi_star, psi_star : optimal primal and dual variables.
    c, p, q, m                  : problem data.
    eps                         : regularization strength.
    Nx, Ny                      : grid sizes.
    n_random                    : number of random perturbations to test.
    delta                       : perturbation scale.
    rng                         : random number generator (default: seed 0).

    Returns
    -------
    dict with keys:
      'saddle_value'       : float  L(pi*, phi*, psi*).
      'dual_perturbations' : list of float  L values at perturbed duals.
      'primal_perturbations': list of float  L values at perturbed primals.
      'duality_gap'        : float  |<phi*, p - P_X pi*>| + |<psi*, q - P_Y pi*>|.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    saddle_value = lagrangian(pi_star, phi_star, psi_star, c, p, q, m, eps, Nx, Ny)

    # Dual perturbations: linearity means L stays near saddle_value when pi* is feasible
    dual_L_values = []
    for _ in range(n_random):
        u = project_zero_mean(rng.standard_normal(Nx).astype(np.float64))
        v = project_zero_mean(rng.standard_normal(Ny).astype(np.float64))
        u_norm = float(np.sqrt(np.mean(u ** 2)))
        v_norm = float(np.sqrt(np.mean(v ** 2)))
        if u_norm > 0:
            u /= u_norm
        if v_norm > 0:
            v /= v_norm
        phi_pert = phi_star + delta * u
        psi_pert = psi_star + delta * v
        L_val = lagrangian(pi_star, phi_pert, psi_pert, c, p, q, m, eps, Nx, Ny)
        dual_L_values.append(float(L_val))

    # Primal perturbations: L(pi* + delta*w, phi*, psi*) >= L(pi*, phi*, psi*)
    primal_L_values = []
    for _ in range(n_random):
        w = rng.standard_normal((Nx, Ny)).astype(np.float64)
        w -= np.mean(w)
        w_norm = float(np.sqrt(np.mean(w ** 2)))
        if w_norm > 0:
            w /= w_norm
        pi_pert = pi_star + delta * w
        L_val = lagrangian(pi_pert, phi_star, psi_star, c, p, q, m, eps, Nx, Ny)
        primal_L_values.append(float(L_val))

    # Duality gap: feasibility residual of pi* under normalized Haar measure
    p_marg, q_marg = marginals(pi_star)
    phi_gap = abs(float(np.mean(phi_star * (p - p_marg))))
    psi_gap = abs(float(np.mean(psi_star * (q - q_marg))))
    duality_gap = phi_gap + psi_gap

    return {
        'saddle_value': float(saddle_value),
        'dual_perturbations': dual_L_values,
        'primal_perturbations': primal_L_values,
        'duality_gap': float(duality_gap),
    }


# ---------------------------------------------------------------------------
# Full verification pipeline
# ---------------------------------------------------------------------------

def run_el_verification(
    Nx: int = 64,
    Ny: int = 64,
    eps: float = 0.1,
    n_test: int = 10,
    rng_seed: int = 42,
) -> dict:
    """
    Full Euler-Lagrange verification pipeline (Experiment 3).

    Steps
    -----
    1. Build problem with marginal_type='exp3', cost_type='exp3'.
    2. Solve regularized OT to get (pi*, phi*, psi*).
    3. Run local weak EL test (ker(A) test functions).
    4. Run global weak EL test (arbitrary zero-mean test functions).
    5. Compute strong EL residual.
    6. Run saddle-point check.
    7. Report all residuals and pass/fail flags.

    Parameters
    ----------
    Nx, Ny   : int  grid sizes.
    eps      : float  regularization strength.
    n_test   : int  number of test functions for weak EL tests.
    rng_seed : int  random seed for reproducibility.

    Returns
    -------
    Comprehensive dict with keys:
      'local_el'    : dict with 'residuals' (list of dicts).
      'global_el'   : dict with 'residuals' (list of dicts).
      'strong_el'   : dict with 'S', 'C', 'residual', 'rel_residual'.
      'saddle'      : dict from saddle_point_check.
      'duality_gap' : float feasibility-based duality gap.
      'pass'        : bool  all rel_residuals below 1e-6.
      'solver'      : dict with solver diagnostics.
    """
    rng = np.random.default_rng(rng_seed)
    x, y = make_torus_grids(Nx, Ny)

    # ------------------------------------------------------------------
    # 1. Build problem and solve
    # ------------------------------------------------------------------
    _, _, p, q, c = make_synthetic_problem(Nx, Ny, marginal_type='exp3', cost_type='exp3')
    m = make_laplacian_multiplier(Nx, Ny)
    sol = solve_regularized_ot(p, q, c, eps, Nx, Ny)
    pi_star = sol['pi_star']
    phi_star = sol['phi_star']
    psi_star = sol['psi_star']

    # ------------------------------------------------------------------
    # 2. Generate test functions
    # ------------------------------------------------------------------
    # ker(A): outer products a(x) * b(y) with mean(a) = mean(b) = 0
    ker_fns: List[np.ndarray] = []
    for _ in range(n_test):
        a = project_zero_mean(rng.standard_normal(Nx).astype(np.float64))
        b = project_zero_mean(rng.standard_normal(Ny).astype(np.float64))
        ker_fns.append(a[:, None] * b[None, :])

    # Arbitrary zero-mean functions on T^2
    zm_fns: List[np.ndarray] = []
    for _ in range(n_test):
        h = rng.standard_normal((Nx, Ny)).astype(np.float64)
        zm_fns.append(project_zero_mean(h))

    # ------------------------------------------------------------------
    # 3. Local weak EL test
    # ------------------------------------------------------------------
    local_residuals = local_el_residual(pi_star, c, m, eps, ker_fns)
    local_pass = all(r['rel_residual'] < 1e-6 for r in local_residuals)

    # ------------------------------------------------------------------
    # 4. Global weak EL test
    # ------------------------------------------------------------------
    global_residuals = global_el_residual(pi_star, phi_star, psi_star, c, m, eps, zm_fns)
    global_pass = all(r['rel_residual'] < 1e-6 for r in global_residuals)

    # ------------------------------------------------------------------
    # 5. Strong EL residual
    # ------------------------------------------------------------------
    strong = strong_el_residual(pi_star, phi_star, psi_star, c, m, eps, Nx, Ny)
    strong_pass = strong['rel_residual'] < 1e-6

    # ------------------------------------------------------------------
    # 6. Saddle-point check
    # ------------------------------------------------------------------
    saddle = saddle_point_check(
        pi_star, phi_star, psi_star, c, p, q, m, eps, Nx, Ny,
        n_random=20, delta=0.1, rng=rng,
    )

    # ------------------------------------------------------------------
    # 7. Assemble results
    # ------------------------------------------------------------------
    duality_gap = saddle['duality_gap']
    overall_pass = local_pass and global_pass and strong_pass and (duality_gap < 1e-6)

    return {
        'local_el': {
            'residuals': local_residuals,
            'pass': local_pass,
            'max_rel_residual': max(r['rel_residual'] for r in local_residuals),
        },
        'global_el': {
            'residuals': global_residuals,
            'pass': global_pass,
            'max_rel_residual': max(r['rel_residual'] for r in global_residuals),
        },
        'strong_el': strong,
        'saddle': saddle,
        'duality_gap': float(duality_gap),
        'pass': overall_pass,
        'solver': {
            'objective': sol['objective'],
            'p_resid': sol['p_resid'],
            'q_resid': sol['q_resid'],
            'Nx': Nx,
            'Ny': Ny,
            'eps': eps,
        },
    }

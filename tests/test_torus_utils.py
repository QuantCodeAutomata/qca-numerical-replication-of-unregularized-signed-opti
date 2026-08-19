import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.torus_utils import (
    make_torus_grids,
    make_freq_grids,
    make_laplacian_multiplier,
    apply_G,
    apply_L,
    project_zero_mean,
    marginals,
    lift,
    inner_product,
    dirichlet_energy,
    check_feasibility,
)


def test_make_torus_grids_shape():
    """Grids have correct shape and range [0,1)."""
    x, y = make_torus_grids(64, 32)
    assert x.shape == (64,)
    assert y.shape == (32,)
    assert x[0] == 0.0
    assert x[-1] < 1.0


def test_make_freq_grids_zero_at_origin():
    """Frequency 0 is at index 0."""
    kx, ky = make_freq_grids(64, 64)
    assert kx[0] == 0.0
    assert ky[0] == 0.0


def test_laplacian_multiplier_zero_at_origin():
    """m[0,0] = 0 (spectral Markov property)."""
    m = make_laplacian_multiplier(64, 64)
    assert m[0, 0] == 0.0
    assert np.all(m >= 0)


def test_laplacian_multiplier_spectral_gap():
    """Smallest positive m value should be 1 (for integer frequencies)."""
    m = make_laplacian_multiplier(64, 64)
    pos_vals = m[m > 0]
    assert np.min(pos_vals) == 1.0


def test_apply_G_zero_mean_output():
    """G maps to zero-mean subspace."""
    x, y = make_torus_grids(32, 32)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    m = make_laplacian_multiplier(32, 32)
    Gf = apply_G(f, m)
    assert abs(np.mean(Gf)) < 1e-12


def test_apply_G_inverse_L():
    """G L f = f on the zero-mean subspace."""
    x, y = make_torus_grids(32, 32)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = np.sin(2*np.pi*X) + 0.5*np.cos(4*np.pi*Y)
    f = project_zero_mean(f)
    m = make_laplacian_multiplier(32, 32)
    GLf = apply_G(apply_L(f, m), m)
    assert np.max(np.abs(GLf - f)) < 1e-10


def test_marginals_shape():
    pi = np.random.randn(32, 48)
    p, q = marginals(pi)
    assert p.shape == (32,)
    assert q.shape == (48,)


def test_marginals_mean():
    """marginals compute the mean over the correct axis."""
    pi = np.ones((10, 20))
    p, q = marginals(pi)
    assert np.allclose(p, 1.0)
    assert np.allclose(q, 1.0)


def test_lift_shape():
    phi = np.ones(32)
    psi = np.ones(48)
    lifted = lift(phi, psi, 32, 48)
    assert lifted.shape == (32, 48)


def test_lift_marginals():
    """Marginals of lift(phi, psi) = phi + mean(psi), psi + mean(phi)."""
    phi = np.array([1.0, 2.0, 3.0])
    psi = np.array([0.5, 1.5])
    lifted = lift(phi, psi, 3, 2)
    p, q = marginals(lifted)
    # P_X lifted = mean over y of (phi[i] + psi[j]) = phi[i] + mean(psi)
    assert np.allclose(p, phi + np.mean(psi))
    assert np.allclose(q, np.mean(phi) + psi)


def test_project_zero_mean():
    f = np.array([1.0, 2.0, 3.0])
    pf = project_zero_mean(f)
    assert abs(np.mean(pf)) < 1e-15


def test_inner_product_symmetry():
    f = np.random.randn(32, 32)
    g = np.random.randn(32, 32)
    assert abs(inner_product(f, g) - inner_product(g, f)) < 1e-14


def test_inner_product_positive():
    f = np.random.randn(32, 32)
    assert inner_product(f, f) >= 0


def test_dirichlet_energy_positive():
    pi = np.random.randn(32, 32)
    m = make_laplacian_multiplier(32, 32)
    assert dirichlet_energy(pi, m) >= 0


def test_dirichlet_energy_zero_for_constant():
    """Constant function has zero Dirichlet energy."""
    pi = np.ones((32, 32))
    m = make_laplacian_multiplier(32, 32)
    assert abs(dirichlet_energy(pi, m)) < 1e-12


def test_check_feasibility_baseline_coupling():
    """Baseline coupling pi0=p+q passes feasibility."""
    x, y = make_torus_grids(32, 32)
    p = project_zero_mean(np.sin(2*np.pi*x))
    q = project_zero_mean(np.cos(2*np.pi*y))
    pi0 = p[:, None] + q[None, :]
    result = check_feasibility(pi0, p, q, tol=1e-10)
    assert result['feasible']
    assert result['p_resid'] < 1e-10
    assert result['q_resid'] < 1e-10

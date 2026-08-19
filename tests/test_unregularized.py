import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.torus_utils import make_torus_grids, project_zero_mean, marginals
from src.unregularized_ot import (
    construct_baseline_coupling,
    construct_zero_marginal_perturbation,
    verify_zero_marginals,
    affine_family,
    run_case_A,
    run_case_B,
)


def test_baseline_coupling_marginals():
    """pi0 = p + q has correct marginals."""
    x, y = make_torus_grids(32, 32)
    p = project_zero_mean(np.sin(2*np.pi*x))
    q = project_zero_mean(np.cos(2*np.pi*y))
    pi0 = construct_baseline_coupling(p, q)
    p_marg, q_marg = marginals(pi0)
    assert np.allclose(p_marg, p, atol=1e-12)
    assert np.allclose(q_marg, q, atol=1e-12)


def test_baseline_coupling_zero_mass():
    """Total mass of pi0 is zero."""
    x, y = make_torus_grids(32, 32)
    p = project_zero_mean(np.sin(2*np.pi*x))
    q = project_zero_mean(np.cos(2*np.pi*y))
    pi0 = construct_baseline_coupling(p, q)
    assert abs(np.mean(pi0)) < 1e-12


def test_perturbation_zero_marginals():
    """r(x,y) = sin(2pi*x)*sin(2pi*y) has zero marginals."""
    x, y = make_torus_grids(64, 64)
    r = construct_zero_marginal_perturbation(x, y, 1, 1)
    result = verify_zero_marginals(r, tol=1e-10)
    assert result['passes']


def test_perturbation_nonzero():
    """r must be nonzero."""
    x, y = make_torus_grids(64, 64)
    r = construct_zero_marginal_perturbation(x, y, 1, 1)
    assert np.max(np.abs(r)) > 0.1


def test_affine_family_feasibility():
    """All pi_t = pi0 + t*r have the same marginals as pi0."""
    x, y = make_torus_grids(32, 32)
    p = project_zero_mean(np.sin(2*np.pi*x))
    q = project_zero_mean(np.cos(2*np.pi*y))
    pi0 = construct_baseline_coupling(p, q)
    r = construct_zero_marginal_perturbation(x, y, 1, 1)
    t_values = np.array([-50.0, 0.0, 50.0])
    for pi_t in affine_family(pi0, r, t_values):
        p_marg, q_marg = marginals(pi_t)
        assert np.allclose(p_marg, p, atol=1e-10)
        assert np.allclose(q_marg, q, atol=1e-10)


def test_case_A_linear_objective():
    """Objective is linear in t with nonzero slope (Case A).

    r = sin(2pi*x)*sin(2pi*y); we add 0.5*sin(2pi*X)*sin(2pi*Y) to c so that
    <c, r> = 0.5 * mean(sin^2(2pi*x)) * mean(sin^2(2pi*y)) = 0.5*0.5*0.5 = 0.125 > 0.
    """
    x, y = make_torus_grids(64, 64)
    X, Y = np.meshgrid(x, y, indexing='ij')
    p = project_zero_mean(np.sin(2*np.pi*x))
    q = project_zero_mean(np.cos(2*np.pi*y))
    # Add sin(2pi*X)*sin(2pi*Y) term so <c, r> != 0
    c = (np.sin(2*np.pi*X)*np.cos(2*np.pi*Y)
         + 0.3*np.cos(4*np.pi*(X-Y))
         + 0.5*np.sin(2*np.pi*X)*np.sin(2*np.pi*Y))
    r = construct_zero_marginal_perturbation(x, y, 1, 1)
    t_values = np.linspace(-100, 100, 201)
    result = run_case_A(p, q, c, r, t_values)
    assert result['passes_A']  # slope != 0
    assert abs(result['slope']) > 1e-6


def test_case_B_constant_objective():
    """Objective is constant in t after orthogonal projection (Case B)."""
    x, y = make_torus_grids(64, 64)
    X, Y = np.meshgrid(x, y, indexing='ij')
    p = project_zero_mean(np.sin(2*np.pi*x))
    q = project_zero_mean(np.cos(2*np.pi*y))
    c = np.sin(2*np.pi*X)*np.cos(2*np.pi*Y) + 0.3*np.cos(4*np.pi*(X-Y))
    r = construct_zero_marginal_perturbation(x, y, 1, 1)
    t_values = np.linspace(-100, 100, 201)
    result = run_case_B(p, q, c, r, t_values)
    assert result['passes_B']  # obj_variation < 1e-10
    assert result['obj_variation'] < 1e-8
    assert abs(result['slope_B']) < 1e-12

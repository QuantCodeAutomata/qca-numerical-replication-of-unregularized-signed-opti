import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.torus_utils import make_laplacian_multiplier
from src.regularized_ot import make_synthetic_problem, solve_regularized_ot, build_K_matrix


def test_solve_produces_feasible_plan():
    """Solver produces pi* with correct marginals."""
    x, y, p, q, c = make_synthetic_problem(32, 32)
    result = solve_regularized_ot(p, q, c, 0.1, 32, 32)
    assert result['p_resid'] < 1e-8
    assert result['q_resid'] < 1e-8


def test_solve_finite_objective():
    x, y, p, q, c = make_synthetic_problem(32, 32)
    result = solve_regularized_ot(p, q, c, 0.1, 32, 32)
    assert np.isfinite(result['objective'])
    assert np.isfinite(result['energy'])


def test_solve_positive_energy():
    x, y, p, q, c = make_synthetic_problem(32, 32)
    result = solve_regularized_ot(p, q, c, 0.1, 32, 32)
    assert result['energy'] >= 0


def test_solve_multiple_eps_feasible():
    """Solution is feasible for several eps values."""
    x, y, p, q, c = make_synthetic_problem(32, 32)
    for eps in [1e-2, 1e-1, 1.0]:
        result = solve_regularized_ot(p, q, c, eps, 32, 32)
        assert result['p_resid'] < 1e-7, f"Failed for eps={eps}"
        assert result['q_resid'] < 1e-7, f"Failed for eps={eps}"


def test_K_matrix_symmetric():
    m = make_laplacian_multiplier(16, 16)
    K = build_K_matrix(16, 16, m)
    assert np.max(np.abs(K - K.T)) < 1e-10


def test_K_matrix_psd():
    m = make_laplacian_multiplier(16, 16)
    K = build_K_matrix(16, 16, m)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-10)  # PSD


def test_invalid_eps_raises():
    x, y, p, q, c = make_synthetic_problem(16, 16)
    with pytest.raises(ValueError):
        solve_regularized_ot(p, q, c, -0.1, 16, 16)

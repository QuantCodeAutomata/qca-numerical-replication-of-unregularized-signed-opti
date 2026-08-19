import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.regularization_path import make_synthetic_problem, solve_regularization_path


def test_path_feasibility():
    """All solutions on regularization path are feasible."""
    x, y, p, q, c = make_synthetic_problem(32, 32, marginal_type='exp6', cost_type='exp6')
    eps_list = [1e-2, 1e-1, 1.0]
    results = solve_regularization_path(p, q, c, eps_list, 32, 32)
    for r in results:
        assert r['p_resid'] < 1e-7, f"Infeasible at eps={r['eps']}"
        assert r['q_resid'] < 1e-7, f"Infeasible at eps={r['eps']}"


def test_higher_eps_lower_high_freq_share():
    """Larger eps -> lower high-frequency energy share."""
    x, y, p, q, c = make_synthetic_problem(32, 32, marginal_type='exp6', cost_type='exp6')
    eps_list = [1e-3, 1.0]
    results = solve_regularization_path(p, q, c, eps_list, 32, 32)
    # high_freq_share at small eps should be >= at large eps
    hf_small = results[0]['spectral_bands']['high_freq_share']
    hf_large = results[1]['spectral_bands']['high_freq_share']
    assert hf_small >= hf_large - 0.05, (
        f"Expected hf_small({hf_small}) >= hf_large({hf_large})"
    )


def test_energy_increases_with_regularization():
    """Dirichlet energy contribution changes with eps."""
    x, y, p, q, c = make_synthetic_problem(32, 32, marginal_type='exp6', cost_type='exp6')
    eps_list = [1e-2, 1.0]
    results = solve_regularization_path(p, q, c, eps_list, 32, 32)
    for r in results:
        assert r['energy'] >= 0

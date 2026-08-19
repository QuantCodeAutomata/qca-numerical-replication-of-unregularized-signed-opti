import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.torus_utils import check_feasibility
from src.panel_analysis import generate_synthetic_panel, solve_panel_ot, run_panel_analysis


def test_synthetic_panel_zero_mean():
    """Generated marginals satisfy zero-mean constraint."""
    panel = generate_synthetic_panel(T=20, N=16, rng_seed=0)
    for p_t in panel['p_list']:
        assert abs(np.mean(p_t)) < 1e-12
    for q_t in panel['q_list']:
        assert abs(np.mean(q_t)) < 1e-12


def test_panel_ot_feasibility():
    """Each period's transport plan is feasible."""
    panel = generate_synthetic_panel(T=10, N=16, rng_seed=0)
    result = solve_panel_ot(panel, eps=0.1, Nx=16, Ny=16)
    for i, (pi_t, p_t, q_t) in enumerate(
        zip(result['pi_t_list'], panel['p_list'], panel['q_list'])
    ):
        feas = check_feasibility(pi_t, p_t, q_t, tol=1e-6)
        assert feas['feasible'], f"Period {i} infeasible: {feas}"


def test_regime_detection_finds_change():
    """Spectral diagnostics detect the known regime shift."""
    result = run_panel_analysis(T=100, N=32, eps=0.1, regime_change_t=50, rng_seed=42)
    # The detected regime change should be within 20 periods of the true change
    assert result['detected_change_period'] is not None
    true_change = 50
    detected = result['detected_change_period']
    assert abs(detected - true_change) <= 20, (
        f"Detected change at {detected}, expected near {true_change}"
    )

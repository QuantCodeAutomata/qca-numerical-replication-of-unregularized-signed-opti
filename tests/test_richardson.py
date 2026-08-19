import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.richardson import run_richardson_analysis


def test_richardson_converges_optimal_alpha():
    result = run_richardson_analysis(Nx=16, Ny=16, eps=0.1, max_iter=5000)
    alpha_opt_result = result['alpha_opt']
    assert alpha_opt_result['converged'], "Richardson did not converge at alpha_opt"


def test_richardson_alpha_opt_vs_direct():
    """Richardson solution matches direct solve."""
    result = run_richardson_analysis(Nx=16, Ny=16, eps=0.1, max_iter=5000)
    assert result['alpha_opt']['final_error'] < 1e-6


def test_richardson_diverges_outside_range():
    """Step size 2.1/lambda_max should diverge or not converge."""
    result = run_richardson_analysis(Nx=16, Ny=16, eps=0.1, max_iter=1000)
    # Divergent alpha should have increasing or stagnant residuals
    assert not result['alpha_diverge']['converged']


def test_K_is_spd():
    result = run_richardson_analysis(Nx=16, Ny=16, eps=0.1, max_iter=100)
    assert result['lambda_min'] > 0
    assert result['lambda_max'] > result['lambda_min']

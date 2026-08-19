import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.euler_lagrange import run_el_verification


def test_local_el_residuals_small():
    """Local EL residuals near zero for optimal solution."""
    result = run_el_verification(Nx=32, Ny=32, eps=0.1, n_test=5)
    for r in result['local_el']['residuals']:
        assert r['rel_residual'] < 1e-6, f"Local EL failed: {r}"


def test_global_el_residuals_small():
    result = run_el_verification(Nx=32, Ny=32, eps=0.1, n_test=5)
    for r in result['global_el']['residuals']:
        assert r['rel_residual'] < 1e-6, f"Global EL failed: {r}"


def test_strong_el_residual_small():
    result = run_el_verification(Nx=32, Ny=32, eps=0.1, n_test=5)
    assert result['strong_el']['rel_residual'] < 1e-6


def test_duality_gap_small():
    result = run_el_verification(Nx=32, Ny=32, eps=0.1, n_test=5)
    assert abs(result['duality_gap']) < 1e-6

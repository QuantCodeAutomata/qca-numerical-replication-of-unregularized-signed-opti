import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.gauss_seidel import run_block_gauss_seidel_analysis


def test_block_structure_symmetric():
    result = run_block_gauss_seidel_analysis(Nx=16, Ny=16, eps=0.1, max_iter=2000)
    assert result['kxy_kyx_sym_residual'] < 1e-10


def test_blocks_invertible():
    result = run_block_gauss_seidel_analysis(Nx=16, Ny=16, eps=0.1, max_iter=2000)
    assert result['min_eig_KXX'] > 0
    assert result['min_eig_KYY'] > 0


def test_gauss_seidel_convergence():
    result = run_block_gauss_seidel_analysis(Nx=16, Ny=16, eps=0.1, max_iter=2000)
    assert result['converged']


def test_spectral_radius_lt_1():
    result = run_block_gauss_seidel_analysis(Nx=16, Ny=16, eps=0.1, max_iter=2000)
    assert result['rho_M'] < 1.0


def test_lifting_norm_identity():
    result = run_block_gauss_seidel_analysis(Nx=16, Ny=16, eps=0.1, max_iter=2000)
    assert result['lifting_norm_diff_phi'] < 1e-10
    assert result['lifting_norm_diff_psi'] < 1e-10

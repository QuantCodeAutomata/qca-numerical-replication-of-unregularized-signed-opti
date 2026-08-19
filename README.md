# Numerical Replication of Unregularized Signed Optimal Transport Ill-Posedness

This repository implements numerical experiments replicating the main claims from a
paper on signed optimal transport on compact locally compact abelian groups.

## Experiments

| ID    | Title                                              |
|-------|----------------------------------------------------|
| Exp 1 | Unregularized Signed OT Ill-Posedness              |
| Exp 2 | Regularized Signed OT Existence and Uniqueness     |
| Exp 3 | Euler-Lagrange Conditions and Strong Duality       |
| Exp 4 | Richardson Iteration Convergence Analysis          |
| Exp 5 | Block Gauss-Seidel Spectral Sinkhorn Iteration     |
| Exp 6 | Regularization Path and Spectral Flatness          |
| Exp 7 | Prototype Fixed-Basis Panel Analysis               |

## Structure

```
repo/
├── data/               # (No empirical data needed — all experiments are synthetic)
├── src/                # Core implementations
│   ├── torus_utils.py          # Torus grid, FFT, marginals, inner products
│   ├── unregularized_ot.py     # Exp 1: ill-posedness demonstration
│   ├── regularized_ot.py       # Exp 2: regularized solver
│   ├── euler_lagrange.py       # Exp 3: variational conditions
│   ├── richardson.py           # Exp 4: Richardson iteration
│   ├── gauss_seidel.py         # Exp 5: block GS iteration
│   ├── regularization_path.py  # Exp 6: regularization path
│   └── panel_analysis.py       # Exp 7: panel analysis
├── exp/                # Experiment runner scripts
├── tests/              # Pytest tests
├── results/            # Generated figures and RESULTS.md
├── README.md
└── .gitignore
```

## Setup

```bash
pip install numpy scipy matplotlib pytest
```

## Running Experiments

```bash
cd /workspace/project
python exp/exp_1_ill_posedness.py
python exp/exp_2_regularized_ot.py
python exp/exp_3_euler_lagrange.py
python exp/exp_4_richardson.py
python exp/exp_5_gauss_seidel.py
python exp/exp_6_regularization_path.py
python exp/exp_7_panel.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Results

See `results/RESULTS.md` for a summary of all experiment outcomes.

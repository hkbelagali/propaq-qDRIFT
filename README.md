# propaq-qDRIFT

Simple wrappers to run qDRIFT circuits with [propaq](https://github.com/hkbelagali/propaq).

qDRIFT (Campbell, 2019) is a random method to copy time evolution
`exp(-i H t)` for a Hamiltonian `H = sum_i coeff_i * P_i`. At each step, it
picks one term at random — a term with a bigger `|coeff_i|` has a bigger
chance to be picked — and applies a small rotation for that term. One
random circuit is a noisy guess. Many random circuits, averaged, give a
good estimate.

This package wraps `propaq`'s Heisenberg-picture Pauli propagator to do
this, so you do not need to write the sampling code by hand.

## Install

```bash
pip install -e .
```

This also installs `propaq` as a dependency.

## Use

```python
from propaq_qdrift import Hamiltonian, qdrift_expectation_value

# Build H = Z0 Z1 + 0.5 X0 + 0.5 X1
h = Hamiltonian.from_labels([
    (1.0, "ZZ"),
    (0.5, "XI"),
    (0.5, "IX"),
])

# Measure the Hamiltonian itself (the energy)
observable = h.to_pauli_term_sum()

result = qdrift_expectation_value(
    h, observable,
    total_time=1.0,
    n_steps=100,     # qDRIFT segments per circuit
    n_samples=20,    # random circuits to average
    seed=0,          # optional, for a repeatable run
)

print(result.mean, "+/-", result.stderr)
```

Pauli labels: the first letter is qubit 0, the next letter is qubit 1, and
so on. Use `I` for identity. For example, `"ZIX"` means Z on qubit 0,
identity on qubit 1, X on qubit 2.

See `examples/tfim_qdrift.py` for a full example (a transverse-field Ising
model).

## Main pieces

- `Hamiltonian`, `PauliTerm`, `pauli_string` — build a Hamiltonian from Pauli labels.
- `qdrift_circuit`, `sample_qdrift_rotations` — build one random qDRIFT circuit.
- `qdrift_expectation_value`, `QDriftResult` — run many circuits and average.

## Test

```bash
pip install -e ".[test]"
pytest tests/
```

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

The Hamiltonian and the per-step circuit are plain Qiskit objects
(`qiskit.quantum_info.SparsePauliOp` and `qiskit.circuit.QuantumCircuit`).
This package only adds the qDRIFT random-sampling step; it then hands the
result to propaq through propaq's own Qiskit converters
(`PauliCircuit.from_qiskit`, `PauliTermSum.from_sparse_pauli_op`), instead
of building propaq's native `PauliRotation`/`PauliString` objects by hand.

## Install

```bash
pip install -e .
```

This also installs `propaq` and `qiskit` as dependencies.

## Use

```python
from qiskit.quantum_info import SparsePauliOp
from propaq_qdrift import qdrift_expectation_value

# Build H = Z0 Z1 + 0.5 X0 + 0.5 X1
h = SparsePauliOp.from_list([
    ("ZZ", 1.0),
    ("XI", 0.5),
    ("IX", 0.5),
])

# Measure the Hamiltonian itself (the energy)
observable = h

result = qdrift_expectation_value(
    h, observable,
    total_time=1.0,
    n_steps=100,     # qDRIFT segments per circuit
    n_samples=20,    # random circuits to average
    seed=0,          # optional, for a repeatable run
)

print(result.mean, "+/-", result.stderr)
```

Pauli labels follow Qiskit's own convention: the *last* letter is qubit 0.
For example, `"XIZ"` means Z on qubit 0, identity on qubit 1, X on qubit
2. For a Hamiltonian with many terms, `SparsePauliOp.from_sparse_list`
is usually easier than writing out full-length labels by hand — see
`examples/tfim_qdrift.py`.

The observable can also be passed as a propaq `PauliTermSum` directly, if
you already have one.

## Main pieces

- `sample_qdrift_circuit` — draw one random qDRIFT circuit, as a Qiskit `QuantumCircuit`.
- `qdrift_circuit` — the same, already converted to a propaq `PauliCircuit`.
- `qdrift_expectation_value`, `QDriftResult` — run many circuits and average.
- `qdrift_lambda` — the qDRIFT normalization constant (`sum |coeff_i|`) for a Hamiltonian.

## Test

```bash
pip install -e ".[test]"
pytest tests/
```

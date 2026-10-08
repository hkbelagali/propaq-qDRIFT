"""
Check propaq against a plain statevector simulation.

propaq computes `<observable>` with the Heisenberg picture: it moves the
observable backward through the circuit, and never builds a state vector.
A statevector simulator does the opposite: it moves the state forward
through the circuit, then reads off `<state|observable|state>` at the
end. The two methods are different code paths to the same physics. If
they do not agree, something is wrong.

This test builds several circuits (qDRIFT circuits, plain Trotter
circuits, and circuits with generic Qiskit gates), runs each one both
ways, and checks the two expectation values match to near machine
precision. We use small qubit counts only, so the statevector method
stays cheap.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from propaq.circuits import PauliCircuit
from propaq.datatypes import PauliTermSum
from propaq.propagators.pauli import PauliPropagator
from propaq_qdrift.qdrift import sample_qdrift_circuit

ATOL = 1e-9


def check_matches_statevector(
    qc: QuantumCircuit,
    observable: SparsePauliOp,
    initial_state: int,
) -> tuple[float, float]:
    """
    Run `qc` both through propaq (Heisenberg picture) and through a plain
    statevector simulation (Schrodinger picture), and return both results.
    """
    n_qubits = qc.num_qubits

    # Heisenberg picture, via propaq.
    propaq_circuit = PauliCircuit.from_qiskit(qc)
    propaq_observable = PauliTermSum.from_sparse_pauli_op(observable)
    prop = PauliPropagator()
    heisenberg_value = prop.expectation_value(
        propaq_observable, propaq_circuit, initial_state=initial_state
    ).expectation_value

    # Schrodinger picture, via a plain statevector.
    state = Statevector.from_int(initial_state, dims=2**n_qubits)
    state = state.evolve(qc)
    statevector_value = state.expectation_value(observable).real

    return heisenberg_value, statevector_value


def assert_matches_statevector(qc, observable, initial_state):
    heisenberg_value, statevector_value = check_matches_statevector(qc, observable, initial_state)
    assert heisenberg_value == pytest.approx(statevector_value, abs=ATOL), (
        f"propaq={heisenberg_value}, statevector={statevector_value}"
    )


# ---------------------------------------------------------------------------
# qDRIFT circuits


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_qdrift_circuit_matches_statevector(seed):
    n_qubits = 5
    hamiltonian = SparsePauliOp.from_sparse_list(
        [("ZZ", [i, i + 1], 1.0) for i in range(n_qubits - 1)]
        + [("X", [i], 0.5) for i in range(n_qubits)],
        num_qubits=n_qubits,
    )
    rng = random.Random(seed)
    qc = sample_qdrift_circuit(hamiltonian, total_time=0.8, n_steps=12, rng=rng)

    observable = hamiltonian  # measure the energy itself
    for initial_state in (0, 1, (1 << n_qubits) - 1):
        assert_matches_statevector(qc, observable, initial_state)


def test_qdrift_circuit_with_multi_term_observable():
    n_qubits = 4
    hamiltonian = SparsePauliOp.from_sparse_list(
        [("ZZ", [i, i + 1], 1.0) for i in range(n_qubits - 1)]
        + [("X", [i], 0.3) for i in range(n_qubits)],
        num_qubits=n_qubits,
    )
    rng = random.Random(7)
    qc = sample_qdrift_circuit(hamiltonian, total_time=1.5, n_steps=20, rng=rng)

    observable = SparsePauliOp.from_sparse_list(
        [("Z", [0], 1.0), ("Z", [1], 0.5), ("X", [2], 0.25)], num_qubits=n_qubits
    )
    assert_matches_statevector(qc, observable, initial_state=0)


# ---------------------------------------------------------------------------
# Plain Trotter circuits, built from native Qiskit rotation gates


def test_tfim_trotter_circuit_matches_statevector():
    n_qubits = 5
    dt = 0.15
    n_trotter_steps = 6
    j_coupling = 1.0
    h_field = 0.6

    qc = QuantumCircuit(n_qubits)
    for _ in range(n_trotter_steps):
        for i in range(n_qubits - 1):
            qc.rzz(2 * j_coupling * dt, i, i + 1)
        for i in range(n_qubits):
            qc.rx(2 * h_field * dt, i)

    observable = SparsePauliOp.from_sparse_list(
        [("ZZ", [i, i + 1], 1.0) for i in range(n_qubits - 1)], num_qubits=n_qubits
    )
    for initial_state in (0, 0b10101):
        assert_matches_statevector(qc, observable, initial_state)


# ---------------------------------------------------------------------------
# Generic circuits with gates outside the pure-rotation basis (h, cx, swap),
# to exercise propaq's Qiskit-transpiler decomposition path, not just its
# directly-supported native gates.


def test_generic_random_circuit_matches_statevector():
    n_qubits = 4
    rng = np.random.default_rng(123)
    qc = QuantumCircuit(n_qubits)
    gate_choices = ["h", "cx", "rz", "rx", "ry", "swap"]
    for _ in range(25):
        gate = rng.choice(gate_choices)
        if gate == "h":
            qc.h(int(rng.integers(n_qubits)))
        elif gate == "cx":
            a, b = rng.choice(n_qubits, size=2, replace=False)
            qc.cx(int(a), int(b))
        elif gate == "swap":
            a, b = rng.choice(n_qubits, size=2, replace=False)
            qc.swap(int(a), int(b))
        else:
            angle = float(rng.uniform(-math.pi, math.pi))
            getattr(qc, gate)(angle, int(rng.integers(n_qubits)))

    observable = SparsePauliOp.from_sparse_list(
        [("Z", [0], 1.0), ("X", [1], 1.0), ("Y", [2], 1.0)], num_qubits=n_qubits
    )
    for initial_state in (0, 0b0101):
        assert_matches_statevector(qc, observable, initial_state)


def test_empty_circuit_matches_statevector():
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)
    observable = SparsePauliOp.from_list([("ZZI", 1.0), ("IIX", 0.5)])
    assert_matches_statevector(qc, observable, initial_state=0b011)

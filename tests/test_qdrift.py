"""Tests for propaq_qdrift.qdrift. The Hamiltonian is a Qiskit SparsePauliOp."""

import math
import random

import pytest
from qiskit.quantum_info import SparsePauliOp

from propaq_qdrift.qdrift import (
    qdrift_circuit,
    qdrift_expectation_value,
    qdrift_lambda,
    sample_qdrift_circuit,
)

OBSERVABLE_Z = SparsePauliOp.from_list([("Z", 1.0)])  # Z on the single qubit


def test_single_term_hamiltonian_matches_exact_rotation():
    """
    With one Hamiltonian term, qDRIFT always picks that term. The circuit
    is then just n_steps copies of the same rotation, which add up to one
    exact rotation. So the qDRIFT estimate should exactly match the known
    closed-form answer, for any number of steps.
    """
    coeff = 0.7
    hamiltonian = SparsePauliOp.from_list([("X", coeff)])

    total_time = 1.3
    expected = math.cos(2.0 * coeff * total_time)

    for n_steps in (1, 2, 10, 50):
        result = qdrift_expectation_value(
            hamiltonian, OBSERVABLE_Z, total_time, n_steps, n_samples=1, seed=0
        )
        assert result.mean == pytest.approx(expected, abs=1e-9)


def test_single_term_negative_coeff():
    coeff = -0.4
    hamiltonian = SparsePauliOp.from_list([("X", coeff)])

    total_time = 2.0
    expected = math.cos(2.0 * coeff * total_time)

    result = qdrift_expectation_value(
        hamiltonian, OBSERVABLE_Z, total_time, n_steps=5, n_samples=1, seed=1
    )
    assert result.mean == pytest.approx(expected, abs=1e-9)


def test_zero_time_is_identity():
    hamiltonian = SparsePauliOp.from_list([("X", 1.0), ("Z", 0.5)])

    result = qdrift_expectation_value(
        hamiltonian, OBSERVABLE_Z, total_time=0.0, n_steps=10, n_samples=1, seed=0
    )
    assert result.mean == pytest.approx(1.0, abs=1e-9)


def test_sample_qdrift_circuit_gate_count():
    hamiltonian = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5)])
    rng = random.Random(42)
    qc = sample_qdrift_circuit(hamiltonian, total_time=1.0, n_steps=7, rng=rng)
    assert len(qc.data) == 7


def test_qdrift_circuit_runs():
    # Each qDRIFT step is one PauliEvolutionGate, but propaq may decompose a
    # multi-qubit Pauli (like "ZZ") into more than one native rotation, so
    # the final rotation count can be bigger than n_steps -- it should never
    # be smaller, and the circuit should not be empty.
    hamiltonian = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5), ("IX", 0.5)])
    circuit = qdrift_circuit(hamiltonian, total_time=0.5, n_steps=20, rng=random.Random(0))
    assert len(circuit.rotations) >= 20


def test_n_steps_must_be_positive():
    hamiltonian = SparsePauliOp.from_list([("X", 1.0)])
    with pytest.raises(ValueError):
        qdrift_circuit(hamiltonian, total_time=1.0, n_steps=0)


def test_n_samples_must_be_positive():
    hamiltonian = SparsePauliOp.from_list([("X", 1.0)])
    with pytest.raises(ValueError):
        qdrift_expectation_value(hamiltonian, OBSERVABLE_Z, 1.0, 5, n_samples=0)


def test_rejects_complex_coefficient():
    hamiltonian = SparsePauliOp.from_list([("X", 1.0j)])
    with pytest.raises(ValueError):
        qdrift_circuit(hamiltonian, total_time=1.0, n_steps=5)


def test_stderr_is_nan_for_single_sample():
    hamiltonian = SparsePauliOp.from_list([("X", 1.0)])
    result = qdrift_expectation_value(hamiltonian, OBSERVABLE_Z, 1.0, 5, n_samples=1, seed=0)
    assert math.isnan(result.stderr)


def test_stderr_is_finite_for_many_samples():
    # A two-term Hamiltonian so different random circuits can differ.
    hamiltonian = SparsePauliOp.from_list([("X", 1.0), ("Z", 1.0)])
    result = qdrift_expectation_value(
        hamiltonian, OBSERVABLE_Z, total_time=1.0, n_steps=4, n_samples=30, seed=0
    )
    assert not math.isnan(result.stderr)
    assert result.stderr >= 0.0
    assert len(result.values) == 30


def test_seed_gives_reproducible_result():
    hamiltonian = SparsePauliOp.from_list([("X", 1.0), ("Z", 1.0)])
    r1 = qdrift_expectation_value(hamiltonian, OBSERVABLE_Z, 1.0, 6, n_samples=10, seed=123)
    r2 = qdrift_expectation_value(hamiltonian, OBSERVABLE_Z, 1.0, 6, n_samples=10, seed=123)
    assert r1.values == r2.values


def test_qdrift_lambda():
    hamiltonian = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5), ("IX", -0.5)])
    assert qdrift_lambda(hamiltonian) == pytest.approx(2.0)


def test_accepts_propaq_pauli_term_sum_observable():
    """The observable can also be a propaq PauliTermSum directly, not just a SparsePauliOp."""
    from propaq.datatypes import PauliString, PauliTermSum

    hamiltonian = SparsePauliOp.from_list([("X", 0.7)])
    observable = PauliTermSum({PauliString(0, 1, 1): 1.0})  # Z on the single qubit

    total_time = 1.3
    expected = math.cos(2.0 * 0.7 * total_time)
    result = qdrift_expectation_value(
        hamiltonian, observable, total_time, n_steps=10, n_samples=1, seed=0
    )
    assert result.mean == pytest.approx(expected, abs=1e-9)

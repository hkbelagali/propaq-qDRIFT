"""Tests for propaq_qdrift.qdrift."""

import math
import random

import pytest

from propaq_qdrift.hamiltonian import Hamiltonian
from propaq_qdrift.qdrift import (
    qdrift_circuit,
    qdrift_expectation_value,
    sample_qdrift_rotations,
)


def test_single_term_hamiltonian_matches_exact_rotation():
    """
    With one Hamiltonian term, qDRIFT always picks that term. The circuit
    is then just n_steps copies of the same rotation, which add up to one
    exact rotation. So the qDRIFT estimate should exactly match the known
    closed-form answer, for any number of steps.
    """
    coeff = 0.7
    hamiltonian = Hamiltonian.from_labels([(coeff, "X")])
    observable = hamiltonian_observable_z()

    total_time = 1.3
    expected = math.cos(2.0 * coeff * total_time)

    for n_steps in (1, 2, 10, 50):
        result = qdrift_expectation_value(
            hamiltonian, observable, total_time, n_steps, n_samples=1, seed=0
        )
        assert result.mean == pytest.approx(expected, abs=1e-9)


def hamiltonian_observable_z():
    from propaq.datatypes import PauliString, PauliTermSum

    obs = PauliTermSum()
    obs.add(PauliString(0, 1, 1), 1.0)  # Z on the single qubit
    return obs


def test_single_term_negative_coeff():
    coeff = -0.4
    hamiltonian = Hamiltonian.from_labels([(coeff, "X")])
    observable = hamiltonian_observable_z()

    total_time = 2.0
    expected = math.cos(2.0 * coeff * total_time)

    result = qdrift_expectation_value(
        hamiltonian, observable, total_time, n_steps=5, n_samples=1, seed=1
    )
    assert result.mean == pytest.approx(expected, abs=1e-9)


def test_zero_time_is_identity():
    hamiltonian = Hamiltonian.from_labels([(1.0, "X"), (0.5, "Z")])
    observable = hamiltonian_observable_z()

    result = qdrift_expectation_value(
        hamiltonian, observable, total_time=0.0, n_steps=10, n_samples=1, seed=0
    )
    assert result.mean == pytest.approx(1.0, abs=1e-9)


def test_sample_qdrift_rotations_count_and_type():
    hamiltonian = Hamiltonian.from_labels([(1.0, "ZZ"), (0.5, "XI")])
    rng = random.Random(42)
    rotations = sample_qdrift_rotations(hamiltonian, total_time=1.0, n_steps=7, rng=rng)
    assert len(rotations) == 7


def test_qdrift_circuit_runs():
    hamiltonian = Hamiltonian.from_labels([(1.0, "ZZ"), (0.5, "XI"), (0.5, "IX")])
    circuit = qdrift_circuit(hamiltonian, total_time=0.5, n_steps=20, rng=random.Random(0))
    assert len(circuit.rotations) == 20


def test_n_steps_must_be_positive():
    hamiltonian = Hamiltonian.from_labels([(1.0, "X")])
    with pytest.raises(ValueError):
        qdrift_circuit(hamiltonian, total_time=1.0, n_steps=0)


def test_n_samples_must_be_positive():
    hamiltonian = Hamiltonian.from_labels([(1.0, "X")])
    observable = hamiltonian_observable_z()
    with pytest.raises(ValueError):
        qdrift_expectation_value(hamiltonian, observable, 1.0, 5, n_samples=0)


def test_stderr_is_nan_for_single_sample():
    hamiltonian = Hamiltonian.from_labels([(1.0, "X")])
    observable = hamiltonian_observable_z()
    result = qdrift_expectation_value(hamiltonian, observable, 1.0, 5, n_samples=1, seed=0)
    assert math.isnan(result.stderr)


def test_stderr_is_finite_for_many_samples():
    # A two-term Hamiltonian so different random circuits can differ.
    hamiltonian = Hamiltonian.from_labels([(1.0, "X"), (1.0, "Z")])
    observable = hamiltonian_observable_z()
    result = qdrift_expectation_value(
        hamiltonian, observable, total_time=1.0, n_steps=4, n_samples=30, seed=0
    )
    assert not math.isnan(result.stderr)
    assert result.stderr >= 0.0
    assert len(result.values) == 30


def test_seed_gives_reproducible_result():
    hamiltonian = Hamiltonian.from_labels([(1.0, "X"), (1.0, "Z")])
    observable = hamiltonian_observable_z()
    r1 = qdrift_expectation_value(hamiltonian, observable, 1.0, 6, n_samples=10, seed=123)
    r2 = qdrift_expectation_value(hamiltonian, observable, 1.0, 6, n_samples=10, seed=123)
    assert r1.values == r2.values

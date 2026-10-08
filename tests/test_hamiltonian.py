"""Tests for propaq_qdrift.hamiltonian."""

import pytest

from propaq_qdrift.hamiltonian import Hamiltonian, PauliTerm, pauli_string


def test_pauli_string_identity():
    p = pauli_string("III")
    assert p.x == 0
    assert p.z == 0
    assert p.n_qubits == 3


def test_pauli_string_single_letters():
    assert pauli_string("X").x == 0b1
    assert pauli_string("X").z == 0b0
    assert pauli_string("Z").x == 0b0
    assert pauli_string("Z").z == 0b1
    assert pauli_string("Y").x == 0b1
    assert pauli_string("Y").z == 0b1


def test_pauli_string_qubit_order():
    # First letter is qubit 0 (bit 0). "XI" means X on qubit 0, I on qubit 1.
    p = pauli_string("XI")
    assert p.x == 0b01
    assert p.z == 0b00

    # "IX" means I on qubit 0, X on qubit 1.
    p2 = pauli_string("IX")
    assert p2.x == 0b10
    assert p2.z == 0b00


def test_pauli_string_mixed():
    # "ZXY" -> qubit0=Z, qubit1=X, qubit2=Y
    p = pauli_string("ZXY")
    assert p.n_qubits == 3
    # Z: x=0,z=1 at bit0 -> z bit0 set
    # X: x=1,z=0 at bit1 -> x bit1 set
    # Y: x=1,z=1 at bit2 -> x bit2 and z bit2 set
    assert p.x == 0b100 | 0b010
    assert p.z == 0b100 | 0b001


def test_pauli_string_lowercase_allowed():
    p = pauli_string("xz")
    assert p.x == 0b01
    assert p.z == 0b10


def test_pauli_string_rejects_bad_letter():
    with pytest.raises(ValueError):
        pauli_string("XQ")


def test_pauli_string_rejects_empty():
    with pytest.raises(ValueError):
        pauli_string("")


def test_hamiltonian_from_labels():
    h = Hamiltonian.from_labels([(1.0, "ZZ"), (0.5, "XI"), (0.5, "IX")])
    assert len(h) == 3
    assert h.n_qubits == 2
    assert h.lambda_ == pytest.approx(2.0)


def test_hamiltonian_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        Hamiltonian.from_labels([(1.0, "Z"), (1.0, "ZZ")])


def test_hamiltonian_rejects_empty():
    with pytest.raises(ValueError):
        Hamiltonian([])


def test_hamiltonian_lambda_with_negative_coeffs():
    h = Hamiltonian.from_labels([(-2.0, "Z"), (1.0, "X")])
    assert h.lambda_ == pytest.approx(3.0)


def test_hamiltonian_to_pauli_term_sum():
    h = Hamiltonian.from_labels([(1.0, "Z"), (0.5, "X")])
    pts = h.to_pauli_term_sum()
    assert len(pts) == 2


def test_pauli_term_from_label():
    term = PauliTerm.from_label(2.0, "XZ")
    assert term.coeff == 2.0
    assert term.pauli.n_qubits == 2

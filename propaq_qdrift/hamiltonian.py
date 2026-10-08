"""
Simple tools to build a Hamiltonian from Pauli strings.

Use these tools to write a Hamiltonian as a sum of terms. Each term has a
coefficient and a Pauli string, like "ZZII" or "XIYI". Then use the
Hamiltonian with the qDRIFT tools in `propaq_qdrift.qdrift`.
"""

from __future__ import annotations

from dataclasses import dataclass

from propaq.datatypes import PauliString, PauliTermSum

_LETTER_TO_XZ = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}


def pauli_string(label: str) -> PauliString:
    """
    Build a PauliString from a label, like "XZIY".

    Rule: the first letter is qubit 0. The next letter is qubit 1. And so
    on. Use "I" for identity (no gate) on a qubit.

    Arguments:
        label: A string of letters I, X, Y, Z. One letter per qubit.

    Returns:
        A PauliString that matches the label.
    """
    if not label:
        raise ValueError("The label must have at least one letter.")

    x_mask = 0
    z_mask = 0
    for qubit_index, letter in enumerate(label):
        letter = letter.upper()
        if letter not in _LETTER_TO_XZ:
            raise ValueError(
                f"Bad letter {letter!r} in label {label!r}. Use only I, X, Y, or Z."
            )
        x_bit, z_bit = _LETTER_TO_XZ[letter]
        x_mask |= x_bit << qubit_index
        z_mask |= z_bit << qubit_index

    return PauliString(x_mask, z_mask, len(label))


@dataclass(frozen=True)
class PauliTerm:
    """One term in a Hamiltonian: `coeff * pauli`."""

    coeff: float
    pauli: PauliString

    @classmethod
    def from_label(cls, coeff: float, label: str) -> "PauliTerm":
        """Build a term from a coefficient and a Pauli label, like `(1.5, "ZZII")`."""
        return cls(coeff, pauli_string(label))


class Hamiltonian:
    """
    A Hamiltonian, written as a sum of Pauli terms: `H = sum_i coeff_i * P_i`.

    All terms must act on the same number of qubits.
    """

    def __init__(self, terms: list[PauliTerm]):
        if not terms:
            raise ValueError("A Hamiltonian needs at least one term.")

        qubit_counts = {term.pauli.n_qubits for term in terms}
        if len(qubit_counts) != 1:
            raise ValueError(
                f"All terms must act on the same number of qubits. Got sizes: {sorted(qubit_counts)}."
            )

        self.terms: list[PauliTerm] = list(terms)
        self.n_qubits: int = terms[0].pauli.n_qubits

    @classmethod
    def from_labels(cls, items: list[tuple[float, str]]) -> "Hamiltonian":
        """
        Build a Hamiltonian from a list of (coeff, label) pairs.

        Example:
            `Hamiltonian.from_labels([(1.0, "ZZII"), (0.5, "XIII"), (0.5, "IXII")])`
        """
        return cls([PauliTerm.from_label(coeff, label) for coeff, label in items])

    @property
    def lambda_(self) -> float:
        """
        The qDRIFT normalization constant: the sum of `|coeff_i|` over all terms.

        This number sets the qDRIFT segment time. A bigger lambda_ means
        qDRIFT needs more steps for the same accuracy.
        """
        return sum(abs(term.coeff) for term in self.terms)

    def to_pauli_term_sum(self) -> PauliTermSum:
        """
        Turn this Hamiltonian into a propaq `PauliTermSum`.

        Use this to measure the Hamiltonian itself as an observable (for
        example, to track energy over time).
        """
        term_sum = PauliTermSum()
        for term in self.terms:
            # PauliTermSum.add() wants a plain real number here, not a
            # Python complex, even though real Hamiltonian terms have no
            # imaginary part.
            term_sum.add(term.pauli, term.coeff)
        return term_sum

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        return f"Hamiltonian(n_qubits={self.n_qubits}, n_terms={len(self.terms)}, lambda_={self.lambda_:.6g})"

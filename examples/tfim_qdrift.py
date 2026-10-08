#!/usr/bin/env python3
"""
Example: qDRIFT on a small transverse-field Ising model (TFIM).

H = J * sum_i Z_i Z_{i+1}  +  h * sum_i X_i   (open chain)

This script builds the Hamiltonian, runs qDRIFT to copy time evolution,
and prints the estimated energy at the end.

Usage:
    python3 examples/tfim_qdrift.py
"""

from __future__ import annotations

from propaq_qdrift import Hamiltonian, qdrift_expectation_value


def build_tfim(n_qubits: int, j_coupling: float, h_field: float) -> Hamiltonian:
    """Build an open-chain TFIM Hamiltonian with n_qubits sites."""
    terms: list[tuple[float, str]] = []

    # ZZ coupling terms, one for each neighbor pair.
    for site in range(n_qubits - 1):
        label = ["I"] * n_qubits
        label[site] = "Z"
        label[site + 1] = "Z"
        terms.append((j_coupling, "".join(label)))

    # Transverse field terms, one per site.
    for site in range(n_qubits):
        label = ["I"] * n_qubits
        label[site] = "X"
        terms.append((h_field, "".join(label)))

    return Hamiltonian.from_labels(terms)


def main() -> None:
    n_qubits = 6
    hamiltonian = build_tfim(n_qubits, j_coupling=1.0, h_field=0.5)
    print(f"Hamiltonian: {hamiltonian}")

    # Measure total energy (the Hamiltonian itself, as an observable).
    observable = hamiltonian.to_pauli_term_sum()

    total_time = 1.0
    n_steps = 200
    n_samples = 20

    result = qdrift_expectation_value(
        hamiltonian,
        observable,
        total_time,
        n_steps,
        initial_state=0,  # all qubits start in |0>
        n_samples=n_samples,
        seed=0,
    )

    print(f"qDRIFT estimate of <H> at t={total_time}:")
    print(f"  mean   = {result.mean:.6f}")
    print(f"  stderr = {result.stderr:.6f}")
    print(f"  (from {result.n_samples} circuits, {result.n_steps} steps each)")


if __name__ == "__main__":
    main()

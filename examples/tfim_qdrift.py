#!/usr/bin/env python3
"""
Example: qDRIFT on a small transverse-field Ising model (TFIM).

H = J * sum_i Z_i Z_{i+1}  +  h * sum_i X_i   (open chain)

The Hamiltonian is a plain Qiskit SparsePauliOp. This script runs qDRIFT
to copy time evolution under it, and prints the estimated energy at the
end.

Usage:
    python3 examples/tfim_qdrift.py
"""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp

from propaq_qdrift import qdrift_expectation_value, qdrift_lambda


def build_tfim(n_qubits: int, j_coupling: float, h_field: float) -> SparsePauliOp:
    """Build an open-chain TFIM Hamiltonian with n_qubits sites, as a SparsePauliOp."""
    sparse_terms: list[tuple[str, list[int], float]] = []

    # ZZ coupling terms, one for each neighbor pair.
    for site in range(n_qubits - 1):
        sparse_terms.append(("ZZ", [site, site + 1], j_coupling))

    # Transverse field terms, one per site.
    for site in range(n_qubits):
        sparse_terms.append(("X", [site], h_field))

    return SparsePauliOp.from_sparse_list(sparse_terms, num_qubits=n_qubits)


def main() -> None:
    n_qubits = 6
    hamiltonian = build_tfim(n_qubits, j_coupling=1.0, h_field=0.5)
    print(f"Hamiltonian: {len(hamiltonian)} terms on {hamiltonian.num_qubits} qubits")
    print(f"qDRIFT lambda = {qdrift_lambda(hamiltonian):.3f}")

    # Measure total energy (the Hamiltonian itself, as the observable).
    observable = hamiltonian

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

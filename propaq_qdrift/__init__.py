"""
propaq-qDRIFT: simple wrappers to run qDRIFT circuits with propaq.

The Hamiltonian and the per-step circuit are plain Qiskit objects
(`qiskit.quantum_info.SparsePauliOp` and `qiskit.circuit.QuantumCircuit`).
This package only adds the qDRIFT sampling step, then hands the result to
propaq through propaq's own Qiskit converters (`PauliCircuit.from_qiskit`,
`PauliTermSum.from_sparse_pauli_op`).

Main pieces:
    sample_qdrift_circuit  -- draw one random qDRIFT circuit (a Qiskit QuantumCircuit)
    qdrift_circuit         -- the same, already converted to a propaq PauliCircuit
    qdrift_expectation_value, QDriftResult  -- run many circuits and average
    qdrift_lambda          -- the qDRIFT normalization constant for a Hamiltonian
"""

from .qdrift import QDriftResult as QDriftResult
from .qdrift import qdrift_circuit as qdrift_circuit
from .qdrift import qdrift_expectation_value as qdrift_expectation_value
from .qdrift import qdrift_lambda as qdrift_lambda
from .qdrift import sample_qdrift_circuit as sample_qdrift_circuit

__version__ = "0.1.0"

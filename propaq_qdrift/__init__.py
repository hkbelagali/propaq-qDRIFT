"""
propaq-qDRIFT: simple wrappers to run qDRIFT circuits with propaq.

Main pieces:
    Hamiltonian, PauliTerm, pauli_string  -- build a Hamiltonian from Pauli labels
    qdrift_circuit, sample_qdrift_rotations  -- build one random qDRIFT circuit
    qdrift_expectation_value, QDriftResult  -- run many circuits and average
"""

from .hamiltonian import Hamiltonian as Hamiltonian
from .hamiltonian import PauliTerm as PauliTerm
from .hamiltonian import pauli_string as pauli_string
from .qdrift import QDriftResult as QDriftResult
from .qdrift import qdrift_circuit as qdrift_circuit
from .qdrift import qdrift_expectation_value as qdrift_expectation_value
from .qdrift import sample_qdrift_rotations as sample_qdrift_rotations

__version__ = "0.1.0"

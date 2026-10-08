"""
qDRIFT: a random method to build short Trotter-like circuits.

qDRIFT (Campbell, 2019) builds a random circuit to copy time evolution
`exp(-i H t)` for a Hamiltonian `H = sum_i coeff_i * P_i`. At each step, it
picks one term at random. A term with a bigger `|coeff_i|` has a bigger
chance to be picked. Then it applies a small rotation for that term.

One random circuit is a noisy guess at the true evolution. Run many random
circuits and average the results, to get a good estimate.

This module builds the random circuit as a plain Qiskit `QuantumCircuit`,
using `qiskit.circuit.library.PauliEvolutionGate` for each step. It then
hands that circuit to propaq's own Qiskit converter
(`PauliCircuit.from_qiskit`), instead of building propaq's native
`PauliRotation`/`PauliString` objects by hand. This keeps the Hamiltonian
and the circuit in standard Qiskit types (`SparsePauliOp`,
`QuantumCircuit`) everywhere outside of propaq itself.

This module wraps `propaq`'s Pauli propagator to run the circuit. It uses
the Heisenberg picture: it does not build a state, it propagates an
observable backward through the circuit and reads off the expectation
value.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from propaq.circuits import PauliCircuit
from propaq.datatypes import PauliTermSum
from propaq.propagators.pauli import PauliPropagator


def _coeffs_as_real(hamiltonian: SparsePauliOp) -> np.ndarray:
    """Check that every coefficient is real, and return them as plain floats."""
    coeffs = hamiltonian.coeffs
    if not np.allclose(coeffs.imag, 0.0):
        raise ValueError(
            "The Hamiltonian has a term with a non-zero imaginary coefficient. "
            "qDRIFT needs a Hermitian Hamiltonian, so every coefficient must be real."
        )
    return coeffs.real.astype(float)


def qdrift_lambda(hamiltonian: SparsePauliOp) -> float:
    """
    The qDRIFT normalization constant: the sum of `|coeff_i|` over all terms.

    This number sets the qDRIFT segment time. A bigger lambda means qDRIFT
    needs more steps for the same accuracy.
    """
    return float(np.sum(np.abs(_coeffs_as_real(hamiltonian))))


def sample_qdrift_circuit(
    hamiltonian: SparsePauliOp,
    total_time: float,
    n_steps: int,
    rng: random.Random,
) -> QuantumCircuit:
    """
    Draw one random qDRIFT circuit, as a plain Qiskit QuantumCircuit.

    Each of the `n_steps` gates picks one Hamiltonian term at random, with
    a bigger `|coeff_i|` giving a bigger chance of being picked. Each pick
    becomes one `PauliEvolutionGate` on the full qubit register.

    Arguments:
        hamiltonian: The Hamiltonian to evolve under, as a Qiskit SparsePauliOp.
        total_time: The total evolution time `t`.
        n_steps: The number of random segments. More steps gives a more
            accurate circuit, but a longer one.
        rng: A `random.Random` instance, so runs can be made reproducible.

    Returns:
        A Qiskit QuantumCircuit with `n_steps` PauliEvolutionGate instructions.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    if len(hamiltonian) < 1:
        raise ValueError("The Hamiltonian has no terms.")

    coeffs = _coeffs_as_real(hamiltonian)
    lam = float(np.sum(np.abs(coeffs)))
    if lam == 0.0:
        raise ValueError("All Hamiltonian coefficients are zero; there is nothing to evolve.")

    tau = lam * total_time / n_steps  # time each sampled gate stands in for
    term_indices = list(range(len(hamiltonian)))
    weights = list(np.abs(coeffs))

    qc = QuantumCircuit(hamiltonian.num_qubits)
    picks = rng.choices(term_indices, weights=weights, k=n_steps)
    for i in picks:
        pauli_label = hamiltonian.paulis[i]
        sign = 1.0 if coeffs[i] >= 0 else -1.0
        # A PauliEvolutionGate on a Pauli with coefficient 1, for time
        # tau * sign, gives exp(-i * tau * sign * P) -- exactly the step
        # qDRIFT needs for a term with coefficient coeffs[i].
        step_op = SparsePauliOp(pauli_label, coeffs=[1.0])
        qc.append(PauliEvolutionGate(step_op, time=tau * sign), range(hamiltonian.num_qubits))

    return qc


def qdrift_circuit(
    hamiltonian: SparsePauliOp,
    total_time: float,
    n_steps: int,
    rng: random.Random | None = None,
) -> PauliCircuit:
    """
    Build one random qDRIFT circuit that approximates `exp(-i * H * total_time)`.

    Arguments:
        hamiltonian: The Hamiltonian to evolve under, as a Qiskit SparsePauliOp.
        total_time: The total evolution time `t`.
        n_steps: The number of random segments.
        rng: Optional `random.Random` instance for reproducible runs. A new
            one is made if not given.

    Returns:
        A propaq PauliCircuit, converted from the Qiskit circuit via
        `PauliCircuit.from_qiskit`.
    """
    rng = rng if rng is not None else random.Random()
    qc = sample_qdrift_circuit(hamiltonian, total_time, n_steps, rng)
    return PauliCircuit.from_qiskit(qc)


@dataclass
class QDriftResult:
    """The result of a qDRIFT run: an average over many random circuits."""

    mean: float
    """The average expectation value, over all sampled circuits."""

    stderr: float
    """The standard error of the mean. `nan` if only one sample was run."""

    n_samples: int
    """The number of random circuits that were run and averaged."""

    n_steps: int
    """The number of qDRIFT segments in each circuit."""

    values: list[float] = field(repr=False)
    """The raw expectation value from each sampled circuit, in run order."""


def qdrift_expectation_value(
    hamiltonian: SparsePauliOp,
    observable: SparsePauliOp | PauliTermSum,
    total_time: float,
    n_steps: int,
    *,
    initial_state: int = 0,
    n_samples: int = 1,
    seed: int | None = None,
    propagator: PauliPropagator | None = None,
) -> QDriftResult:
    """
    Estimate `<observable>` after time evolution `exp(-i * H * total_time)`, using qDRIFT.

    This runs `n_samples` independent random qDRIFT circuits and averages
    the results. One circuit alone is a noisy guess; averaging over many
    gives a much better estimate of the true qDRIFT channel average, and
    lets you read off the spread (the standard error) too.

    Arguments:
        hamiltonian: The Hamiltonian `H = sum_i coeff_i * P_i`, as a Qiskit
            SparsePauliOp.
        observable: The observable to measure. Pass a Qiskit SparsePauliOp
            (it is converted with propaq's own `PauliTermSum.from_sparse_pauli_op`),
            or a propaq PauliTermSum directly.
        total_time: The total evolution time `t`.
        n_steps: The number of qDRIFT segments per circuit. More steps
            means less error from the random approximation, at the cost of
            a longer circuit.
        initial_state: The starting computational basis state (as an int).
        n_samples: The number of independent random circuits to average.
        seed: Optional seed, so the run can be repeated exactly.
        propagator: Optional, a ready-made PauliPropagator (for example,
            one set up with truncation or noise). A plain one is made if
            this is not given.

    Returns:
        A QDriftResult with the mean, the standard error, and all raw values.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    if isinstance(observable, SparsePauliOp):
        observable = PauliTermSum.from_sparse_pauli_op(observable)

    rng = random.Random(seed)
    prop = propagator if propagator is not None else PauliPropagator()

    values: list[float] = []
    for _ in range(n_samples):
        circuit = qdrift_circuit(hamiltonian, total_time, n_steps, rng)
        result = prop.expectation_value(observable, circuit, initial_state=initial_state)
        values.append(result.expectation_value)

    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        stderr = math.sqrt(variance / len(values))
    else:
        stderr = float("nan")

    return QDriftResult(
        mean=mean,
        stderr=stderr,
        n_samples=n_samples,
        n_steps=n_steps,
        values=values,
    )

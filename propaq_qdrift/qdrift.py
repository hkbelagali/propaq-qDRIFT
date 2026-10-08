"""
qDRIFT: a random method to build short Trotter-like circuits.

qDRIFT (Campbell, 2019) builds a random circuit to copy time evolution
`exp(-i H t)` for a Hamiltonian `H = sum_i coeff_i * P_i`. At each step, it
picks one term at random. A term with a bigger `|coeff_i|` has a bigger
chance to be picked. Then it applies a small rotation for that term.

One random circuit is a noisy guess at the true evolution. Run many random
circuits and average the results, to get a good estimate.

This module wraps `propaq`'s Pauli propagator to do this. It uses the
Heisenberg picture: it does not build a state, it propagates an observable
backward through the circuit and reads off the expectation value.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from propaq.circuits import PauliCircuit
from propaq.circuits.pauli.rotation import PauliRotation
from propaq.datatypes import PauliTermSum
from propaq.propagators.pauli import PauliPropagator

from .hamiltonian import Hamiltonian


def sample_qdrift_rotations(
    hamiltonian: Hamiltonian,
    total_time: float,
    n_steps: int,
    rng: random.Random,
) -> list[PauliRotation]:
    """
    Draw one random list of qDRIFT rotations for `hamiltonian` and `total_time`.

    Each of the `n_steps` rotations picks one Hamiltonian term at random,
    with a bigger `|coeff_i|` giving a bigger chance of being picked. Run
    all the rotations in order to get one qDRIFT circuit.

    Arguments:
        hamiltonian: The Hamiltonian to evolve under.
        total_time: The total evolution time `t`.
        n_steps: The number of random segments. More steps gives a more
            accurate circuit, but a longer one.
        rng: A `random.Random` instance, so runs can be made reproducible.

    Returns:
        A list of `n_steps` PauliRotation gates, in the order to apply them.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")

    lam = hamiltonian.lambda_
    if lam == 0.0:
        raise ValueError("All Hamiltonian coefficients are zero; there is nothing to evolve.")

    weights = [abs(term.coeff) for term in hamiltonian.terms]
    tau = lam * total_time / n_steps  # time each sampled rotation stands in for

    picked_terms = rng.choices(hamiltonian.terms, weights=weights, k=n_steps)

    rotations: list[PauliRotation] = []
    for term in picked_terms:
        sign = 1.0 if term.coeff >= 0 else -1.0
        # PauliRotation(P, angle) applies exp(-i * angle * P / 2); to copy
        # exp(-i * tau * sign * P), pass angle = 2 * tau * sign.
        angle = 2.0 * tau * sign
        rotations.append(PauliRotation(term.pauli, angle))

    return rotations


def qdrift_circuit(
    hamiltonian: Hamiltonian,
    total_time: float,
    n_steps: int,
    rng: random.Random | None = None,
) -> PauliCircuit:
    """
    Build one random qDRIFT circuit that approximates `exp(-i * H * total_time)`.

    Arguments:
        hamiltonian: The Hamiltonian to evolve under.
        total_time: The total evolution time `t`.
        n_steps: The number of random segments.
        rng: Optional `random.Random` instance for reproducible runs. A new
            one is made if not given.

    Returns:
        A propaq PauliCircuit with `n_steps` rotations.
    """
    rng = rng if rng is not None else random.Random()
    rotations = sample_qdrift_rotations(hamiltonian, total_time, n_steps, rng)
    return PauliCircuit(rotations)


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
    hamiltonian: Hamiltonian,
    observable: PauliTermSum,
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
        hamiltonian: The Hamiltonian `H = sum_i coeff_i * P_i`.
        observable: The observable to measure, as a propaq PauliTermSum.
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

"""Structured synthesis of the diagonal phases.

A generic ``DiagonalGate`` on ``n_q`` qubits costs O(2^n) gates, which makes any
resource estimate built on it an upper bound rather than a representative count.
Every diagonal needed here is at most *quadratic in the register index*, and a
quadratic diagonal factorises into a global phase, ``n_q`` single-qubit phases
and ``n_q (n_q - 1) / 2`` controlled phases -- O(n^2) instead of O(2^n).

The mechanism is that register bits are idempotent, ``b^2 = b``, so for
``j = sum_q 2^q b_q``::

    j^2 = sum_q 4^q b_q + 2 sum_{q < q'} 2^(q + q') b_q b_q'

with no higher-order terms.  Each expansion below is validated against the exact
diagonal in ``tests/test_phase_synthesis.py``.

Quadratic position phase (harmonic oscillator)
----------------------------------------------
With ``x_j = x_min + dx j`` and phase ``exp(-i V(x_j) tau / hbar)``,
``V = m w^2 x^2 / 2``, write ``A = m w^2 tau / (2 hbar)``::

    constant     = -A x_min^2
    linear[q]    = -A (2 x_min dx 2^q + dx^2 4^q)
    pairwise[q,q'] = -2 A dx^2 2^(q + q')

Signed momentum-square phase (periodic kinetic)
-----------------------------------------------
FFT bin ordering is exactly two's complement: bin ``k`` carries
``k~ = k - 2^n [b_(n-1)]``, so ``k~ = sum_q s_q b_q`` with ``s_q = 2^q`` for
``q < n-1`` and ``s_(n-1) = -2^(n-1)``.  Then ``k~^2`` expands as above with
``s_q`` in place of ``2^q``.  The sign of the top bit is what makes the highest
bin carry ``-N/2`` rather than ``+N/2``.

Folded sine-mode phase (Dirichlet kinetic, extended register)
-------------------------------------------------------------
On the ``n+2``-qubit QST register the mode index is the triangle wave
``mu(m) = r if r <= N else 2N - r`` with ``r = m mod 2N``.  That looks
non-polynomial, but with ``c_q`` the bits of ``m`` and ``N = 2^n``::

    mu^2 = r^2 + c_n (4N^2 - 4N r)

and the ``c_n`` single-bit contribution cancels identically because
``4N^2 - 4N 2^n = 0``.  Collecting terms leaves

    single[q]        = 4^q                       for q = 0..n
    pair[q, q']      = 2 * 2^(q + q')            for q < q' < n
    pair[q, n]       = -2^(q + n + 1)            for q < n

so the fold costs nothing beyond a sign flip on the pairs involving the fold
bit, and bit ``n+1`` does not enter at all.  The phase therefore remains O(n^2).

Linear tilt phase
-----------------
``V(x) = F (x - L/2)`` is linear in the index, so it needs only a global phase
and ``n_q`` single-qubit rotations -- no two-qubit gates at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit

from ..grids import validate_power_of_two

__all__ = [
    "BitExpansion",
    "evaluate_bit_expansion",
    "folded_sine_kinetic_expansion",
    "harmonic_position_expansion",
    "linear_tilt_expansion",
    "signed_momentum_expansion",
    "structured_phase_circuit",
]


@dataclass
class BitExpansion:
    """A diagonal phase written as constant + linear + pairwise bit terms."""

    n_qubits: int
    constant: float
    linear: np.ndarray
    pairwise: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.linear = np.asarray(self.linear, dtype=float)
        if self.linear.shape != (self.n_qubits,):
            raise ValueError(f"linear must have shape ({self.n_qubits},).")
        if self.pairwise is None:
            self.pairwise = np.zeros((self.n_qubits, self.n_qubits))
        self.pairwise = np.asarray(self.pairwise, dtype=float)
        if self.pairwise.shape != (self.n_qubits, self.n_qubits):
            raise ValueError(f"pairwise must have shape ({self.n_qubits}, {self.n_qubits}).")

    @property
    def n_two_qubit_terms(self) -> int:
        """Number of controlled-phase gates this expansion needs."""
        return int(np.count_nonzero(np.triu(self.pairwise, 1)))

    @property
    def n_single_qubit_terms(self) -> int:
        return int(np.count_nonzero(self.linear))


def evaluate_bit_expansion(expansion: BitExpansion) -> np.ndarray:
    """Return the diagonal ``exp(i phi(m))`` implied by an expansion."""
    size = 2**expansion.n_qubits
    indices = np.arange(size)
    bits = ((indices[:, None] >> np.arange(expansion.n_qubits)[None, :]) & 1).astype(float)

    phase = expansion.constant + bits @ expansion.linear
    upper = np.triu(expansion.pairwise, 1)
    phase = phase + np.einsum("mq,qp,mp->m", bits, upper, bits)
    return np.exp(1j * phase)


def structured_phase_circuit(expansion: BitExpansion, name: str = "phase") -> QuantumCircuit:
    """Emit the expansion as a global phase, ``p`` gates and ``cp`` gates."""
    circuit = QuantumCircuit(expansion.n_qubits, name=name)
    circuit.global_phase += expansion.constant

    for qubit, angle in enumerate(expansion.linear):
        if angle != 0.0:
            circuit.p(angle, qubit)

    for control in range(expansion.n_qubits):
        for target in range(control + 1, expansion.n_qubits):
            angle = expansion.pairwise[control, target]
            if angle != 0.0:
                circuit.cp(angle, control, target)
    return circuit


# ------------------------------------------------------------- builders ----


def harmonic_position_expansion(
    n_grid: int,
    x_left: float,
    spacing: float,
    mass: float,
    omega: float,
    hbar: float,
    duration: float,
) -> BitExpansion:
    """Return ``exp(-i m w^2 x_j^2 tau / (2 hbar))`` as a bit expansion."""
    validate_power_of_two(n_grid)
    n_qubits = int(np.log2(n_grid))
    amplitude = mass * omega**2 * duration / (2.0 * hbar)
    weights = 2.0 ** np.arange(n_qubits)

    linear = -amplitude * (2.0 * x_left * spacing * weights + spacing**2 * weights**2)
    pairwise = -2.0 * amplitude * spacing**2 * np.outer(weights, weights)
    return BitExpansion(n_qubits, -amplitude * x_left**2, linear, np.triu(pairwise, 1))


def signed_momentum_expansion(
    n_grid: int,
    spacing: float,
    mass: float,
    hbar: float,
    duration: float,
) -> BitExpansion:
    """Return ``exp(-i p_k^2 tau / (2 m hbar))`` in FFT bin order, as a bit expansion."""
    validate_power_of_two(n_grid)
    n_qubits = int(np.log2(n_grid))
    length = n_grid * spacing
    amplitude = (2.0 * np.pi * hbar / length) ** 2 * duration / (2.0 * mass * hbar)

    # Two's complement: the top bit carries a negative weight.
    weights = 2.0 ** np.arange(n_qubits)
    weights[-1] *= -1.0

    linear = -amplitude * weights**2
    pairwise = -2.0 * amplitude * np.outer(weights, weights)
    return BitExpansion(n_qubits, 0.0, linear, np.triu(pairwise, 1))


def folded_sine_kinetic_expansion(
    n_grid: int,
    length: float,
    mass: float,
    hbar: float,
    duration: float,
) -> BitExpansion:
    """Return the QST kinetic phase on the ``n+2``-qubit extended register.

    Implements the triangle-wave mode map exactly, in closed form, without
    enumerating the ``4N`` diagonal entries.
    """
    validate_power_of_two(n_grid)
    n_data = int(np.log2(n_grid))
    n_total = n_data + 2
    amplitude = (hbar * np.pi**2 / (2.0 * mass * length**2)) * duration

    linear = np.zeros(n_total)
    pairwise = np.zeros((n_total, n_total))

    # r occupies bits 0..n_data; bit n_data is the fold bit; bit n_data+1 is
    # unused because mu is periodic with period 2N.
    for q in range(n_data + 1):
        linear[q] = -amplitude * 4.0**q
    for q in range(n_data + 1):
        for p in range(q + 1, n_data + 1):
            if p == n_data:
                pairwise[q, p] = amplitude * 2.0 ** (q + n_data + 1)
            else:
                pairwise[q, p] = -amplitude * 2.0 * 2.0 ** (q + p)
    return BitExpansion(n_total, 0.0, linear, pairwise)


def linear_tilt_expansion(
    n_grid: int,
    length: float,
    tilt_force: float,
    hbar: float,
    duration: float,
) -> BitExpansion:
    """Return ``exp(-i F (x_j - L/2) tau / hbar)`` on the Dirichlet midpoint grid.

    Linear in the index, so no controlled-phase gates are needed at all.
    """
    validate_power_of_two(n_grid)
    n_qubits = int(np.log2(n_grid))
    spacing = length / n_grid
    amplitude = tilt_force * duration / hbar

    weights = 2.0 ** np.arange(n_qubits)
    linear = -amplitude * spacing * weights
    constant = -amplitude * (0.5 * spacing - 0.5 * length)
    return BitExpansion(n_qubits, constant, linear, np.zeros((n_qubits, n_qubits)))

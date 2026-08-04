"""Quantum sine transform: an exact circuit realisation of the DST-II.

This module implements the Dirichlet kinetic propagator as a circuit, in a form
whose action is checked against the analytical DST-II matrix rather than against
a relabelled QFT.

Construction
------------
The orthonormal DST-II of size ``N`` is obtained from a DFT of size ``M = 4N``
by an odd extension on the half-integer grid.  For data ``x[j]``, ``j = 0..N-1``,
define the length-``M`` sequence

    y[2j + 1]      = +x[j]
    y[4N - 2j - 1] = -x[j]
    y[m]           =  0        otherwise (all even m).

Then, with the forward convention ``F[nu] = M^(-1/2) sum_m y[m] exp(-2 pi i nu m / M)``,

    sum_m y[m] exp(-2 pi i nu m / M) = -2i sum_j x[j] sin(2 pi nu (2j+1) / M)
                                     = -2i sum_j x[j] sin(pi nu (j + 1/2) / N),

which is exactly the DST-II kernel.  The odd extension supplies the antisymmetry
that encodes the Dirichlet walls, and the interleaving onto odd positions
supplies the half-cell shift of the midpoint grid.

Qubit layout (little-endian, ``n_q + 2`` qubits, register index ``m``)
---------------------------------------------------------------------
    qubit 0          parity ancilla, holds bit 0 of ``m`` (the interleave)
    qubits 1..n_q    data register, holds ``j``
    qubit n_q + 1    reflection ancilla, holds bit n_q+1 of ``m``

The data basis state ``|j>`` with both ancillas in ``|0>`` therefore sits at
register index ``m = 2j``.

The extension circuit is

    X(q_0);  X(q_{n+1});  H(q_{n+1});  CX(q_{n+1}, q_i) for i = 1..n_q

taking ``|0> |j> |0>`` to ``(|0> |j> |1> - |1> |~j> |1>) / sqrt(2)``, since
``4N - 2j - 1`` has bit 0 set, top bit set, and middle bits ``N - 1 - j``, which
is the bitwise complement of ``j`` on ``n_q`` bits.  Being a product of X, H and
CX gates it is manifestly unitary, so its inverse is its adjoint and the
ancillas are returned to ``|0>`` exactly, with no measurement or reset.

Applying the kinetic phase
--------------------------
The transform does not leave the DST-II coefficients in a clean ``n_q``-qubit
register: mode ``nu`` appears at the four register indices
``{nu, 2N - nu, 2N + nu, 4N - nu}`` (only ``{N, 3N}`` for ``nu = N``), with
amplitudes fixed relative to one another.  Extracting the coefficients would
require an extra scalar per mode: the amplitude at index ``nu`` equals the
DST-II coefficient divided by ``2i`` for ``nu < N`` and by ``i sqrt(2)`` for
``nu = N``.

That extraction is unnecessary here.  A diagonal that assigns the *same* phase
to all indices carrying a given mode preserves the extended subspace, so the
kinetic propagator is realised directly by

    U_kin = E^dagger . F^dagger . D_mu . F . E

where ``D_mu[m] = exp(-i E_{mu(m)} dt / hbar)`` and ``mu`` is the triangle-wave
index map ``mu(m) = r if r <= N else 2N - r`` with ``r = m mod 2N``.  Indices
``m = 0`` and ``m = 2N`` carry zero amplitude and are assigned zero phase.

On the data subspace this equals ``S^T diag(exp(-i E_nu dt / hbar)) S`` exactly,
which is what ``tests/test_qst_circuit.py`` verifies against the analytical
DST-II matrix.

The diagonal is emitted here as a generic ``DiagonalGate``.  That is correct but
its synthesis cost is not representative; structured phase synthesis and the
resulting resource counts are handled in
:mod:`boundary_aware_dynamics.circuits.phases`.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, QFTGate

from ..grids import sine_mode_energies, validate_power_of_two
from ..transforms import analytical_dst2_matrix

__all__ = [
    "extended_register_size",
    "mode_index_map",
    "odd_extension_circuit",
    "qst_kinetic_propagator_circuit",
    "reference_dirichlet_kinetic_operator",
    "sine_mode_phase_diagonal",
]


def extended_register_size(n_grid: int) -> tuple[int, int]:
    """Return ``(n_data_qubits, n_total_qubits)`` for the QST register.

    The QST needs two ancillas beyond the data register: one for the half-cell
    interleave and one for the odd reflection.
    """
    validate_power_of_two(n_grid)
    n_data_qubits = int(np.log2(n_grid))
    return n_data_qubits, n_data_qubits + 2


def mode_index_map(n_grid: int) -> np.ndarray:
    """Return ``mu(m)`` for ``m = 0..4N-1``: which sine mode each index carries.

    The map is a triangle wave. Indices ``0`` and ``2N`` carry no amplitude in
    the extended subspace and are reported as mode ``0``.
    """
    validate_power_of_two(n_grid)
    m = np.arange(4 * n_grid)
    r = m % (2 * n_grid)
    return np.where(r <= n_grid, r, 2 * n_grid - r)


def odd_extension_circuit(n_grid: int) -> QuantumCircuit:
    """Return the unitary that odd-extends and interleaves the data register."""
    n_data_qubits, n_total = extended_register_size(n_grid)
    circuit = QuantumCircuit(n_total, name="QST_extend")

    circuit.x(0)                       # move data onto odd register indices
    circuit.x(n_total - 1)             # prepare the reflection ancilla in |1>
    circuit.h(n_total - 1)             # -> (|0> - |1>)/sqrt(2), supplying the minus sign
    for data_qubit in range(1, n_data_qubits + 1):
        circuit.cx(n_total - 1, data_qubit)   # j -> ~j on the reflected branch
    return circuit


def sine_mode_phase_diagonal(
    n_grid: int,
    length: float,
    mass: float,
    hbar: float,
    time_step: float,
) -> np.ndarray:
    """Return ``exp(-i E_mu(m) dt / hbar)`` over the ``4N`` extended indices."""
    energies = sine_mode_energies(n_grid, length, mass, hbar)
    # Prepend a zero for the unused mode-0 slots at m = 0 and m = 2N.
    energies_by_mode = np.concatenate(([0.0], energies))
    return np.exp(-1j * energies_by_mode[mode_index_map(n_grid)] * time_step / hbar)


def qst_kinetic_propagator_circuit(
    n_grid: int,
    length: float,
    mass: float,
    hbar: float,
    time_step: float,
) -> QuantumCircuit:
    """Return the exact Dirichlet kinetic propagator ``E^dag F^dag D F E``.

    On the data subspace (both ancillas in ``|0>``) this equals the DST-II
    diagonalised propagator ``S^T exp(-i E dt / hbar) S``.
    """
    _, n_total = extended_register_size(n_grid)
    diagonal = sine_mode_phase_diagonal(n_grid, length, mass, hbar, time_step)

    circuit = QuantumCircuit(n_total, name="QST_kinetic_step")
    circuit.compose(odd_extension_circuit(n_grid), inplace=True)
    # Forward DFT: Qiskit's QFTGate carries the opposite exponential sign.
    circuit.append(QFTGate(n_total).inverse(), range(n_total))
    circuit.append(DiagonalGate(list(diagonal)), range(n_total))
    circuit.append(QFTGate(n_total), range(n_total))
    circuit.compose(odd_extension_circuit(n_grid).inverse(), inplace=True)
    return circuit


def reference_dirichlet_kinetic_operator(
    n_grid: int,
    length: float,
    mass: float,
    hbar: float,
    time_step: float,
) -> np.ndarray:
    """Return the analytical target ``S^T exp(-i E dt / hbar) S`` (``N x N``).

    Built from :func:`analytical_dst2_matrix`, so validating the circuit against
    it does not go through SciPy.
    """
    sine_matrix = analytical_dst2_matrix(n_grid)
    energies = sine_mode_energies(n_grid, length, mass, hbar)
    return sine_matrix.T @ (np.exp(-1j * energies * time_step / hbar)[:, None] * sine_matrix)

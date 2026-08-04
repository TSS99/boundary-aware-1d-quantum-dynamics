"""QFT convention and the periodic (ring) kinetic propagator.

Qiskit's ``QFTGate`` implements

    QFT |j> = N^(-1/2) sum_k exp(+2 pi i j k / N) |k>

which is the *inverse* of the NumPy forward transform used in
:mod:`boundary_aware_dynamics.transforms`.  The forward transform is therefore
``QFTGate(n).inverse()``.  ``QFTGate`` includes the terminating swap network, so
input and output share the same little-endian qubit ordering and no manual bit
reversal is required.

The kinetic phase must be applied in ``numpy.fft.fftfreq`` bin order, i.e. bin
``k`` carries signed index ``k`` for ``k < N/2`` and ``k - N`` otherwise.  Bin
``N/2`` carries ``-N/2``.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, QFTGate

from ..grids import fft_kinetic_energies, validate_power_of_two
from ..transforms import dft_matrix

__all__ = [
    "periodic_kinetic_phase_diagonal",
    "qft_kinetic_propagator_circuit",
    "qft_matrix",
    "reference_periodic_kinetic_operator",
]


def qft_matrix(n_qubits: int) -> np.ndarray:
    """Return the matrix Qiskit's ``QFTGate`` implements: the inverse orthonormal DFT."""
    return dft_matrix(2**n_qubits).conj().T


def periodic_kinetic_phase_diagonal(
    n_grid: int,
    spacing: float,
    mass: float,
    hbar: float,
    time_step: float,
) -> np.ndarray:
    """Return ``exp(-i p_k^2 dt / (2 m hbar))`` in FFT bin order."""
    validate_power_of_two(n_grid)
    energies = fft_kinetic_energies(n_grid, spacing, mass, hbar)
    return np.exp(-1j * energies * time_step / hbar)


def qft_kinetic_propagator_circuit(
    n_grid: int,
    spacing: float,
    mass: float,
    hbar: float,
    time_step: float,
) -> QuantumCircuit:
    """Return the periodic kinetic propagator ``F^dag D_T F`` on ``log2 N`` qubits.

    No ancillas are required: the periodic transform acts on the data register
    directly.  This is the structural difference from the Dirichlet case, where
    the odd extension costs two ancillas.
    """
    validate_power_of_two(n_grid)
    n_qubits = int(np.log2(n_grid))
    diagonal = periodic_kinetic_phase_diagonal(n_grid, spacing, mass, hbar, time_step)

    circuit = QuantumCircuit(n_qubits, name="QFT_kinetic_step")
    circuit.append(QFTGate(n_qubits).inverse(), range(n_qubits))
    circuit.append(DiagonalGate(list(diagonal)), range(n_qubits))
    circuit.append(QFTGate(n_qubits), range(n_qubits))
    return circuit


def reference_periodic_kinetic_operator(
    n_grid: int,
    spacing: float,
    mass: float,
    hbar: float,
    time_step: float,
) -> np.ndarray:
    """Return the analytical target ``F^dag diag(exp(-i T dt / hbar)) F``."""
    matrix = dft_matrix(n_grid)
    diagonal = periodic_kinetic_phase_diagonal(n_grid, spacing, mass, hbar, time_step)
    return matrix.conj().T @ (diagonal[:, None] * matrix)

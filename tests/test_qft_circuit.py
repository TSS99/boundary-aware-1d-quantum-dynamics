"""Qiskit QFT convention and the periodic kinetic propagator."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator

from boundary_aware_dynamics.circuits.qft import (
    periodic_kinetic_phase_diagonal,
    qft_kinetic_propagator_circuit,
    qft_matrix,
    reference_periodic_kinetic_operator,
)
from boundary_aware_dynamics.transforms import dft_matrix, fft_forward

QUBIT_COUNTS = [1, 2, 3, 4]
TOLERANCE = 1e-10

SPACING, MASS, HBAR = 0.25, 1.0, 1.0


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
def test_qiskit_qft_is_the_inverse_numpy_transform(n_qubits):
    # Fixes the exponential sign. If this is inverted, the kinetic phase is
    # applied to -p instead of p and the dynamics run backwards in momentum.
    observed = Operator(QFTGate(n_qubits)).data
    assert np.allclose(observed, qft_matrix(n_qubits), atol=1e-12)
    assert np.allclose(Operator(QFTGate(n_qubits).inverse()).data, dft_matrix(2**n_qubits), atol=1e-12)


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
def test_qft_gate_needs_no_manual_bit_reversal(n_qubits):
    # QFTGate includes the terminating swaps, so a basis state maps to the
    # little-endian spectrum directly. A missing swap network would show up as
    # a bit-reversal permutation of the columns.
    n_grid = 2**n_qubits
    operator = Operator(QFTGate(n_qubits).inverse()).data
    for j in range(n_grid):
        basis = np.zeros(n_grid, dtype=complex)
        basis[j] = 1.0
        assert np.allclose(operator @ basis, fft_forward(basis), atol=1e-12)


@pytest.mark.parametrize("n_qubits", [2, 3])
def test_qft_transforms_seeded_random_states_correctly(n_qubits):
    n_grid = 2**n_qubits
    operator = Operator(QFTGate(n_qubits).inverse()).data
    rng = np.random.default_rng(7)
    for _ in range(5):
        vector = rng.normal(size=n_grid) + 1j * rng.normal(size=n_grid)
        vector /= np.linalg.norm(vector)
        assert np.allclose(operator @ vector, fft_forward(vector), atol=1e-12)


def test_kinetic_phase_diagonal_follows_fftfreq_ordering():
    n_grid = 8
    diagonal = periodic_kinetic_phase_diagonal(n_grid, SPACING, MASS, HBAR, 0.3)
    # Bins k and N-k carry the same energy, hence identical phases.
    assert np.allclose(diagonal[1:4], diagonal[5:][::-1])
    assert diagonal[0] == pytest.approx(1.0)


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
@pytest.mark.parametrize("time_step", [0.0, 0.05, 0.3])
def test_periodic_propagator_matches_the_analytical_operator(n_qubits, time_step):
    n_grid = 2**n_qubits
    circuit = qft_kinetic_propagator_circuit(n_grid, SPACING, MASS, HBAR, time_step)
    observed = Operator(circuit).data
    expected = reference_periodic_kinetic_operator(n_grid, SPACING, MASS, HBAR, time_step)
    assert np.linalg.norm(observed - expected) < TOLERANCE


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
def test_periodic_propagator_uses_no_ancillas(n_qubits):
    # The structural contrast with the QST: a ring needs no odd extension.
    circuit = qft_kinetic_propagator_circuit(2**n_qubits, SPACING, MASS, HBAR, 0.1)
    assert circuit.num_qubits == n_qubits


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
def test_periodic_propagator_is_unitary_and_time_reversible(n_qubits):
    n_grid = 2**n_qubits
    forward = Operator(qft_kinetic_propagator_circuit(n_grid, SPACING, MASS, HBAR, 0.19)).data
    backward = Operator(qft_kinetic_propagator_circuit(n_grid, SPACING, MASS, HBAR, -0.19)).data
    assert np.allclose(forward.conj().T @ forward, np.eye(n_grid), atol=TOLERANCE)
    assert np.allclose(forward @ backward, np.eye(n_grid), atol=TOLERANCE)


@pytest.mark.parametrize("n_qubits", [2, 3])
def test_periodic_propagator_gives_each_plane_wave_its_own_eigenphase(n_qubits):
    from boundary_aware_dynamics.grids import fft_kinetic_energies

    n_grid = 2**n_qubits
    time_step = 0.23
    operator = Operator(qft_kinetic_propagator_circuit(n_grid, SPACING, MASS, HBAR, time_step)).data
    energies = fft_kinetic_energies(n_grid, SPACING, MASS, HBAR)
    j = np.arange(n_grid)

    for k in range(n_grid):
        plane_wave = np.exp(2j * np.pi * k * j / n_grid) / np.sqrt(n_grid)
        expected = np.exp(-1j * energies[k] * time_step / HBAR) * plane_wave
        assert np.allclose(operator @ plane_wave, expected, atol=TOLERANCE)


@pytest.mark.parametrize("n_qubits", [2, 3])
def test_periodic_and_dirichlet_propagators_differ(n_qubits):
    # Same physical box, same time step, different boundary topology: the two
    # circuits must not implement the same operator.
    from boundary_aware_dynamics.circuits.qst import (
        qst_kinetic_propagator_circuit,
        reference_dirichlet_kinetic_operator,
    )

    n_grid, length, time_step = 2**n_qubits, 10.0, 0.4
    spacing = length / n_grid
    periodic = reference_periodic_kinetic_operator(n_grid, spacing, MASS, HBAR, time_step)
    dirichlet = reference_dirichlet_kinetic_operator(n_grid, length, MASS, HBAR, time_step)
    assert np.linalg.norm(periodic - dirichlet) > 0.1

    # And each circuit must match its own reference, not the other's.
    observed = Operator(qst_kinetic_propagator_circuit(n_grid, length, MASS, HBAR, time_step)).data
    indices = 2 * np.arange(n_grid)
    assert np.linalg.norm(observed[np.ix_(indices, indices)] - dirichlet) < TOLERANCE

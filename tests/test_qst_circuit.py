"""The QST circuit is validated against the analytical DST-II, not against a QFT.

The previous implementation in this repository labelled a bare ``QFTGate`` as
"QST"; ``test_a_bare_qft_is_not_a_sine_transform`` is kept as an explicit guard
so that substitution cannot silently return.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator

from boundary_aware_dynamics.circuits.qst import (
    extended_register_size,
    mode_index_map,
    odd_extension_circuit,
    qst_kinetic_propagator_circuit,
    reference_dirichlet_kinetic_operator,
)
from boundary_aware_dynamics.transforms import analytical_dst2_matrix, dst2_matrix

QUBIT_COUNTS = [1, 2, 3]
TOLERANCE = 1e-10

LENGTH, MASS, HBAR = 10.0, 1.0, 1.0


def data_subspace_indices(n_grid: int) -> np.ndarray:
    """Register indices holding ``|j>`` with both ancillas in ``|0>``: ``m = 2j``."""
    return 2 * np.arange(n_grid)


def data_block(operator: np.ndarray, n_grid: int) -> np.ndarray:
    indices = data_subspace_indices(n_grid)
    return operator[np.ix_(indices, indices)]


# ------------------------------------------------------------ register size --


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
def test_qst_costs_exactly_two_ancillas(n_qubits):
    n_grid = 2**n_qubits
    n_data, n_total = extended_register_size(n_grid)
    assert n_data == n_qubits
    assert n_total == n_qubits + 2
    assert qst_kinetic_propagator_circuit(n_grid, LENGTH, MASS, HBAR, 0.1).num_qubits == n_qubits + 2


# --------------------------------------------------------------- index map --


def test_mode_index_map_is_a_triangle_wave_hitting_every_mode():
    n_grid = 8
    modes = mode_index_map(n_grid)
    assert modes.size == 4 * n_grid
    # Mode nu sits at four indices, except nu = N which sits at two.
    for nu in range(1, n_grid):
        assert sorted(np.flatnonzero(modes == nu)) == [nu, 2 * n_grid - nu, 2 * n_grid + nu, 4 * n_grid - nu]
    assert sorted(np.flatnonzero(modes == n_grid)) == [n_grid, 3 * n_grid]
    # Only the two zero-amplitude indices are left unassigned.
    assert sorted(np.flatnonzero(modes == 0)) == [0, 2 * n_grid]


# --------------------------------------------------------------- extension --


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
def test_odd_extension_produces_the_expected_sequence(n_qubits):
    n_grid = 2**n_qubits
    operator = Operator(odd_extension_circuit(n_grid)).data

    for j in range(n_grid):
        extended = operator[:, 2 * j]          # image of |j> with clean ancillas
        expected = np.zeros(4 * n_grid, dtype=complex)
        expected[2 * j + 1] = 1.0 / np.sqrt(2.0)
        expected[4 * n_grid - 2 * j - 1] = -1.0 / np.sqrt(2.0)
        assert np.allclose(extended, expected, atol=1e-12)


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
def test_odd_extension_is_unitary_so_ancillas_uncompute_exactly(n_qubits):
    n_grid = 2**n_qubits
    circuit = odd_extension_circuit(n_grid)
    operator = Operator(circuit).data
    identity = np.eye(operator.shape[0])
    assert np.allclose(operator.conj().T @ operator, identity, atol=1e-12)

    round_trip = circuit.compose(circuit.inverse())
    assert np.allclose(Operator(round_trip).data, identity, atol=1e-12)


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
def test_extended_sequence_reproduces_the_dst2_kernel(n_qubits):
    # The defining property: the DFT of the odd extension carries the DST-II
    # kernel, with the known per-mode factor (2i below the Nyquist mode,
    # i sqrt(2) at it).
    n_grid = 2**n_qubits
    operator = Operator(odd_extension_circuit(n_grid)).data
    sine_matrix = analytical_dst2_matrix(n_grid)

    rng = np.random.default_rng(11)
    x = rng.normal(size=n_grid) + 1j * rng.normal(size=n_grid)
    x /= np.linalg.norm(x)

    extended = operator[:, 2 * np.arange(n_grid)] @ x
    spectrum = np.fft.fft(extended, norm="ortho")
    coefficients = sine_matrix @ x

    for nu in range(1, n_grid):
        assert spectrum[nu] * 2j == pytest.approx(coefficients[nu - 1], abs=1e-12)
    assert spectrum[n_grid] * 1j * np.sqrt(2.0) == pytest.approx(coefficients[n_grid - 1], abs=1e-12)


# ------------------------------------------------------- kinetic propagator --


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
@pytest.mark.parametrize("time_step", [0.0, 0.03, 0.25])
def test_qst_propagator_matches_the_analytical_dst2_propagator(n_qubits, time_step):
    # The headline claim: on the data subspace the circuit equals
    # S^T exp(-i E dt / hbar) S built from the closed-form DST-II matrix.
    n_grid = 2**n_qubits
    circuit = qst_kinetic_propagator_circuit(n_grid, LENGTH, MASS, HBAR, time_step)
    observed = data_block(Operator(circuit).data, n_grid)
    expected = reference_dirichlet_kinetic_operator(n_grid, LENGTH, MASS, HBAR, time_step)
    assert np.linalg.norm(observed - expected) < TOLERANCE


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
def test_qst_propagator_leaves_no_amplitude_in_the_ancillas(n_qubits):
    # If the ancillas were not uncomputed, the data block would be sub-unitary
    # and probability would leak into ancilla-excited states.
    n_grid = 2**n_qubits
    circuit = qst_kinetic_propagator_circuit(n_grid, LENGTH, MASS, HBAR, 0.17)
    operator = Operator(circuit).data
    columns = operator[:, data_subspace_indices(n_grid)]

    leaked = np.delete(columns, data_subspace_indices(n_grid), axis=0)
    assert np.max(np.abs(leaked)) < TOLERANCE

    block = data_block(operator, n_grid)
    assert np.allclose(block.conj().T @ block, np.eye(n_grid), atol=TOLERANCE)


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
def test_zero_time_step_gives_the_identity(n_qubits):
    n_grid = 2**n_qubits
    circuit = qst_kinetic_propagator_circuit(n_grid, LENGTH, MASS, HBAR, 0.0)
    assert np.allclose(data_block(Operator(circuit).data, n_grid), np.eye(n_grid), atol=TOLERANCE)


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
def test_qst_propagator_is_time_reversible(n_qubits):
    n_grid = 2**n_qubits
    forward = qst_kinetic_propagator_circuit(n_grid, LENGTH, MASS, HBAR, 0.21)
    backward = qst_kinetic_propagator_circuit(n_grid, LENGTH, MASS, HBAR, -0.21)
    product = data_block(Operator(forward).data, n_grid) @ data_block(Operator(backward).data, n_grid)
    assert np.allclose(product, np.eye(n_grid), atol=TOLERANCE)


@pytest.mark.parametrize("n_qubits", [2, 3])
def test_qst_propagator_gives_each_sine_mode_its_own_eigenphase(n_qubits):
    from boundary_aware_dynamics.grids import sine_mode_energies

    n_grid = 2**n_qubits
    time_step = 0.13
    block = data_block(
        Operator(qst_kinetic_propagator_circuit(n_grid, LENGTH, MASS, HBAR, time_step)).data, n_grid
    )
    sine_matrix = analytical_dst2_matrix(n_grid)
    energies = sine_mode_energies(n_grid, LENGTH, MASS, HBAR)

    for mode in range(n_grid):
        vector = sine_matrix[mode]
        expected = np.exp(-1j * energies[mode] * time_step / HBAR) * vector
        assert np.allclose(block @ vector, expected, atol=TOLERANCE)


@pytest.mark.parametrize("n_qubits", QUBIT_COUNTS)
def test_circuit_agrees_with_scipy_dst_as_a_cross_check(n_qubits):
    n_grid = 2**n_qubits
    from boundary_aware_dynamics.grids import sine_mode_energies

    time_step = 0.09
    sine_matrix = dst2_matrix(n_grid)
    energies = sine_mode_energies(n_grid, LENGTH, MASS, HBAR)
    scipy_target = sine_matrix.T @ (np.exp(-1j * energies * time_step / HBAR)[:, None] * sine_matrix)
    observed = data_block(
        Operator(qst_kinetic_propagator_circuit(n_grid, LENGTH, MASS, HBAR, time_step)).data, n_grid
    )
    assert np.linalg.norm(observed - scipy_target) < TOLERANCE


# ------------------------------------------------------------ the guard rail --


@pytest.mark.parametrize("n_qubits", [2, 3])
def test_a_bare_qft_is_not_a_sine_transform(n_qubits):
    # Regression guard.  An earlier version of this repository built
    # QFTGate(n + 2).inverse(), labelled it "QST", and reported resource counts
    # for it.  The disagreement with the true DST-II is of the same magnitude as
    # the transform itself, so the substitution is not a small approximation.
    n_grid = 2**n_qubits
    circuit = QuantumCircuit(n_qubits + 2)
    circuit.append(QFTGate(n_qubits + 2).inverse(), range(n_qubits + 2))
    proxy = Operator(circuit).data[:n_grid, :n_grid]

    sine_matrix = analytical_dst2_matrix(n_grid)
    error = np.linalg.norm(proxy - sine_matrix)
    assert error > 0.5 * np.linalg.norm(sine_matrix)

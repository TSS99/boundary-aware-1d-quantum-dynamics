"""Structured phase expansions, checked against the exact diagonals."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from boundary_aware_dynamics.circuits.phases import (
    BitExpansion,
    evaluate_bit_expansion,
    folded_sine_kinetic_expansion,
    harmonic_position_expansion,
    linear_tilt_expansion,
    signed_momentum_expansion,
    structured_phase_circuit,
)
from boundary_aware_dynamics.circuits.qft import periodic_kinetic_phase_diagonal
from boundary_aware_dynamics.circuits.qst import sine_mode_phase_diagonal
from boundary_aware_dynamics.grids import dirichlet_midpoint_grid, periodic_grid
from boundary_aware_dynamics.propagators import harmonic_potential, tilted_potential

QUBITS = [1, 2, 3, 4]
LENGTH, MASS, HBAR, OMEGA = 10.0, 1.0, 1.0, 1.0
DURATION = 0.137
TOLERANCE = 1e-12


def phases_match(observed: np.ndarray, expected: np.ndarray) -> bool:
    return bool(np.max(np.abs(observed - expected)) < TOLERANCE)


# ------------------------------------------------------ exactness checks ----


@pytest.mark.parametrize("n_qubits", QUBITS)
def test_quadratic_position_phase_is_exact(n_qubits):
    n_grid = 2**n_qubits
    grid = periodic_grid(-8.0, 8.0, n_grid)
    expansion = harmonic_position_expansion(
        n_grid, -8.0, grid.spacing, MASS, OMEGA, HBAR, DURATION
    )
    expected = np.exp(-1j * harmonic_potential(grid.positions, MASS, OMEGA) * DURATION / HBAR)
    assert phases_match(evaluate_bit_expansion(expansion), expected)


@pytest.mark.parametrize("n_qubits", QUBITS)
def test_signed_momentum_phase_matches_fft_bin_ordering(n_qubits):
    # Two's complement is what makes bin N/2 carry -N/2. If the top bit were
    # given a positive weight this test would fail on the upper half of the
    # spectrum only.
    n_grid = 2**n_qubits
    grid = periodic_grid(-8.0, 8.0, n_grid)
    expansion = signed_momentum_expansion(n_grid, grid.spacing, MASS, HBAR, DURATION)
    expected = periodic_kinetic_phase_diagonal(n_grid, grid.spacing, MASS, HBAR, DURATION)
    assert phases_match(evaluate_bit_expansion(expansion), expected)


@pytest.mark.parametrize("n_qubits", QUBITS)
def test_folded_sine_phase_reproduces_the_triangle_wave_map(n_qubits):
    # The mode map is a triangle wave, yet mu^2 still closes at quadratic order
    # in the register bits. This is the identity that keeps the QST kinetic
    # phase at O(n^2) instead of O(4N).
    n_grid = 2**n_qubits
    expansion = folded_sine_kinetic_expansion(n_grid, LENGTH, MASS, HBAR, DURATION)
    expected = sine_mode_phase_diagonal(n_grid, LENGTH, MASS, HBAR, DURATION)
    assert phases_match(evaluate_bit_expansion(expansion), expected)


@pytest.mark.parametrize("n_qubits", QUBITS)
def test_linear_tilt_phase_is_exact(n_qubits):
    n_grid = 2**n_qubits
    grid = dirichlet_midpoint_grid(LENGTH, n_grid)
    expansion = linear_tilt_expansion(n_grid, LENGTH, 5.0, HBAR, DURATION)
    expected = np.exp(-1j * tilted_potential(grid.positions, LENGTH, 5.0) * DURATION / HBAR)
    assert phases_match(evaluate_bit_expansion(expansion), expected)


# ----------------------------------------------------------- circuits ------


@pytest.mark.parametrize("n_qubits", QUBITS)
def test_circuit_reproduces_its_expansion_including_global_phase(n_qubits):
    n_grid = 2**n_qubits
    grid = periodic_grid(-8.0, 8.0, n_grid)
    expansion = harmonic_position_expansion(
        n_grid, -8.0, grid.spacing, MASS, OMEGA, HBAR, DURATION
    )
    operator = Operator(structured_phase_circuit(expansion)).data
    assert np.allclose(operator, np.diag(evaluate_bit_expansion(expansion)), atol=1e-12)


@pytest.mark.parametrize("n_qubits", QUBITS)
def test_folded_sine_circuit_matches_the_qst_diagonal(n_qubits):
    n_grid = 2**n_qubits
    expansion = folded_sine_kinetic_expansion(n_grid, LENGTH, MASS, HBAR, DURATION)
    operator = Operator(structured_phase_circuit(expansion)).data
    expected = sine_mode_phase_diagonal(n_grid, LENGTH, MASS, HBAR, DURATION)
    assert np.allclose(np.diag(operator), expected, atol=1e-12)


# -------------------------------------------------------------- scaling ----


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
def test_quadratic_phases_need_only_quadratically_many_gates(n_qubits):
    n_grid = 2**n_qubits
    grid = periodic_grid(-8.0, 8.0, n_grid)
    expansion = harmonic_position_expansion(
        n_grid, -8.0, grid.spacing, MASS, OMEGA, HBAR, DURATION
    )
    assert expansion.n_single_qubit_terms <= n_qubits
    assert expansion.n_two_qubit_terms == n_qubits * (n_qubits - 1) // 2
    # The point of structured synthesis: far fewer than the 2^n diagonal entries.
    assert expansion.n_two_qubit_terms < n_grid


def test_linear_tilt_needs_no_two_qubit_gates_at_all():
    expansion = linear_tilt_expansion(64, LENGTH, 5.0, HBAR, DURATION)
    assert expansion.n_two_qubit_terms == 0
    circuit = structured_phase_circuit(expansion)
    assert all(instruction.operation.num_qubits == 1 for instruction in circuit.data)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
def test_folded_sine_phase_ignores_the_top_extended_qubit(n_qubits):
    # mu has period 2N, so bit n+1 cannot affect the phase. If it did, the
    # reflected copies would pick up different phases and the ancillas would
    # not uncompute.
    expansion = folded_sine_kinetic_expansion(2**n_qubits, LENGTH, MASS, HBAR, DURATION)
    top = expansion.n_qubits - 1
    assert expansion.linear[top] == 0.0
    assert np.all(expansion.pairwise[:, top] == 0.0)
    assert np.all(expansion.pairwise[top, :] == 0.0)


# ------------------------------------------------------------ validation ---


def test_expansion_shapes_are_validated():
    with pytest.raises(ValueError, match="linear must have shape"):
        BitExpansion(3, 0.0, np.zeros(2))
    with pytest.raises(ValueError, match="pairwise must have shape"):
        BitExpansion(3, 0.0, np.zeros(3), np.zeros((2, 2)))


def test_zero_duration_gives_the_identity_phase():
    expansion = harmonic_position_expansion(16, -8.0, 1.0, MASS, OMEGA, HBAR, 0.0)
    assert np.allclose(evaluate_bit_expansion(expansion), 1.0)

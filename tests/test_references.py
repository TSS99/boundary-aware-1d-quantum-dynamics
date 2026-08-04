"""Reference solutions, including the closed-form tilt matrix elements."""

from __future__ import annotations

import numpy as np
import pytest

from boundary_aware_dynamics.grids import dirichlet_midpoint_grid, periodic_grid, sine_mode_energies
from boundary_aware_dynamics.references import (
    finite_difference_reference,
    harmonic_reference,
    hermite_basis,
    sine_basis,
    sine_galerkin_hamiltonian,
    sine_galerkin_reference,
    tilt_matrix_elements,
)
from boundary_aware_dynamics.states import gaussian_wavepacket, sine_windowed_gaussian

LENGTH, MASS, HBAR = 10.0, 1.0, 1.0


# ------------------------------------------------------- tilt potential -----


def test_tilt_matrix_elements_match_numerical_quadrature():
    # The closed form is derived by hand in references.py; this checks the
    # derivation rather than trusting it.
    n_modes, tilt = 8, 1.3
    dense = np.linspace(0.0, LENGTH, 200_001)
    basis = sine_basis(dense, LENGTH, n_modes)
    potential = tilt * (dense - 0.5 * LENGTH)

    expected = np.array(
        [[np.trapezoid(basis[a] * potential * basis[b], dense) for b in range(n_modes)] for a in range(n_modes)]
    )
    assert np.allclose(tilt_matrix_elements(n_modes, LENGTH, tilt), expected, atol=1e-6)


def test_tilt_matrix_is_symmetric_with_a_vanishing_diagonal():
    elements = tilt_matrix_elements(12, LENGTH, 1.0)
    assert np.allclose(elements, elements.T)
    # sin^2 is symmetric about L/2 while the tilt is antisymmetric there.
    assert np.allclose(np.diag(elements), 0.0)


def test_tilt_couples_only_opposite_parity_modes():
    elements = tilt_matrix_elements(10, LENGTH, 1.0)
    for a in range(10):
        for b in range(10):
            if (a + b) % 2 == 0:      # nu = a+1, nu' = b+1, so nu + nu' is even
                assert elements[a, b] == 0.0
            else:
                assert elements[a, b] != 0.0


def test_tilt_scales_linearly_with_the_force():
    base = tilt_matrix_elements(6, LENGTH, 1.0)
    assert np.allclose(tilt_matrix_elements(6, LENGTH, 2.5), 2.5 * base)


def test_zero_tilt_gives_the_free_well_hamiltonian():
    hamiltonian = sine_galerkin_hamiltonian(16, LENGTH, MASS, HBAR, tilt_force=0.0)
    assert np.allclose(hamiltonian, np.diag(sine_mode_energies(16, LENGTH, MASS, HBAR)))


# -------------------------------------------------------- sine-Galerkin ----


def well_packet(positions: np.ndarray) -> np.ndarray:
    """The Benchmark B/C initial state as a callable of position."""
    envelope = np.exp(-((positions - 5.0) ** 2) / (4.0 * 0.8**2))
    carrier = np.exp(2.0j * (positions - 5.0))
    return envelope * carrier * np.sin(np.pi * positions / LENGTH)


def test_sine_galerkin_reproduces_free_eigenmode_phases():
    grid = dirichlet_midpoint_grid(LENGTH, 64)
    times = np.array([0.0, 0.5, 1.7])

    def mode_three(positions: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / LENGTH) * np.sin(3.0 * np.pi * positions / LENGTH) + 0j

    solution = sine_galerkin_reference(mode_three, grid, times, MASS, HBAR, n_modes=64)
    mode = sine_basis(grid.positions, LENGTH, 3)[2]
    energy = sine_mode_energies(64, LENGTH, MASS, HBAR)[2]
    for index, time in enumerate(times):
        expected = mode * np.exp(-1j * energy * time / HBAR)
        overlap = grid.spacing * np.vdot(expected, solution.states[index])
        assert abs(overlap) == pytest.approx(1.0, abs=1e-8)
        assert np.angle(overlap) == pytest.approx(0.0, abs=1e-6)


def test_sine_galerkin_conserves_norm_and_captures_the_full_state():
    grid = dirichlet_midpoint_grid(LENGTH, 64)
    times = np.linspace(0.0, 3.0, 7)

    solution = sine_galerkin_reference(
        well_packet, grid, times, MASS, HBAR, n_modes=256, tilt_force=1.0
    )
    for state in solution.states:
        assert grid.spacing * np.vdot(state, state).real == pytest.approx(1.0, abs=1e-10)
    # Projecting on a dense grid keeps the coefficient norm at one; projecting
    # on the 64-point simulation grid would alias every mode above N and break
    # this badly.
    assert solution.diagnostics["raw_coefficient_norm"] == pytest.approx(1.0, abs=1e-6)
    assert solution.diagnostics["tail_weight"] < 1e-6


def test_sine_galerkin_reference_converges_in_basis_size():
    grid = dirichlet_midpoint_grid(LENGTH, 64)
    times = np.array([0.0, 2.0])

    finest = sine_galerkin_reference(
        well_packet, grid, times, MASS, HBAR, 512, tilt_force=1.0
    ).final_state
    errors = []
    for n_modes in (16, 24, 32):
        coarse = sine_galerkin_reference(well_packet, grid, times, MASS, HBAR, n_modes, tilt_force=1.0)
        errors.append(np.sqrt(grid.spacing) * np.linalg.norm(coarse.final_state - finest))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-2


# ---------------------------------------------------- finite difference -----


def test_finite_difference_agrees_with_the_sine_reference_for_the_free_well():
    # Independent basis and independent method: this is the non-circular check
    # that the zero-potential well benchmark needs.
    grid = dirichlet_midpoint_grid(LENGTH, 64)
    times = np.array([0.0, 1.0])

    sine = sine_galerkin_reference(well_packet, grid, times, MASS, HBAR, n_modes=256).final_state
    difference = finite_difference_reference(
        well_packet, grid, times, MASS, HBAR, refinement=16
    ).final_state
    assert fidelity_of(sine, difference, grid.spacing) > 0.999


def test_finite_difference_reference_converges_with_refinement():
    grid = dirichlet_midpoint_grid(LENGTH, 64)
    times = np.array([0.0, 1.0])
    sine = sine_galerkin_reference(well_packet, grid, times, MASS, HBAR, n_modes=256).final_state

    infidelities = []
    for refinement in (2, 4, 8):
        state = finite_difference_reference(
            well_packet, grid, times, MASS, HBAR, refinement=refinement
        ).final_state
        infidelities.append(1.0 - fidelity_of(sine, state, grid.spacing))
    # Second-order finite differences: each doubling should cut the error, and
    # the trend must be monotone rather than noise-dominated.
    assert infidelities[0] > infidelities[1] > infidelities[2]
    assert infidelities[-1] < 1e-4


def test_finite_difference_grid_contains_every_simulation_midpoint():
    # The fine grid is chosen so no interpolation is needed in either direction;
    # if that alignment breaks, the reference silently acquires interpolation
    # error and stops being a clean independent check.
    grid = dirichlet_midpoint_grid(LENGTH, 16)
    refinement = 4
    spacing = LENGTH / (2 * grid.n_grid * refinement)
    n_fine = 2 * grid.n_grid * refinement - 1
    positions = spacing * np.arange(1, n_fine + 1)
    nodes = (2 * np.arange(grid.n_grid) + 1) * refinement - 1
    assert np.allclose(positions[nodes], grid.positions)


def test_finite_difference_conserves_norm():
    grid = dirichlet_midpoint_grid(LENGTH, 32)
    times = np.linspace(0.0, 2.0, 5)
    solution = finite_difference_reference(well_packet, grid, times, MASS, HBAR, refinement=8)
    for state in solution.states:
        assert grid.spacing * np.vdot(state, state).real == pytest.approx(1.0, abs=1e-10)


# ------------------------------------------------------------- harmonic ----


def test_harmonic_reference_keeps_an_eigenstate_stationary_in_density():
    grid = periodic_grid(-8.0, 8.0, 128)
    times = np.linspace(0.0, 2.0 * np.pi, 5)
    solution = harmonic_reference(
        grid, times, centre=0.0, momentum=0.0, sigma=np.sqrt(0.5),
        mass=MASS, omega=1.0, hbar=HBAR,
        dense_grid_size=4096, basis_cap=64, tail_tolerance=1e-10,
    )
    # sigma = sqrt(hbar / 2 m omega) is the harmonic ground state, so |psi|^2
    # must not move at all.
    ground_density = np.abs(solution.states[0]) ** 2
    for state in solution.states[1:]:
        assert np.max(np.abs(np.abs(state) ** 2 - ground_density)) < 1e-8


def test_harmonic_reference_reports_its_truncation():
    grid = periodic_grid(-8.0, 8.0, 64)
    solution = harmonic_reference(
        grid, np.array([0.0, 1.0]), centre=2.0, momentum=0.0, sigma=1.0,
        mass=MASS, omega=1.0, hbar=HBAR,
        dense_grid_size=4096, basis_cap=128, tail_tolerance=1e-10,
    )
    assert 0 < solution.diagnostics["n_modes"] < 128
    assert solution.diagnostics["tail_weight"] < 1e-9
    assert solution.diagnostics["raw_coefficient_norm"] == pytest.approx(1.0, abs=1e-6)


def test_harmonic_basis_is_orthonormal():
    dense = np.linspace(-12.0, 12.0, 20_001)
    basis = hermite_basis(dense, 8, MASS, 1.0, HBAR)
    gram = np.array([[np.trapezoid(basis[a] * basis[b], dense) for b in range(8)] for a in range(8)])
    assert np.allclose(gram, np.eye(8), atol=1e-8)


def test_exhausted_basis_cap_is_reported_not_silently_truncated():
    grid = periodic_grid(-8.0, 8.0, 64)
    with pytest.raises(RuntimeError, match="basis cap"):
        harmonic_reference(
            grid, np.array([0.0]), centre=2.0, momentum=5.0, sigma=0.2,
            mass=MASS, omega=1.0, hbar=HBAR,
            dense_grid_size=4096, basis_cap=8, tail_tolerance=1e-12,
        )


def fidelity_of(a: np.ndarray, b: np.ndarray, spacing: float) -> float:
    return float(abs(spacing * np.vdot(a, b)) ** 2)


def test_gaussian_reference_input_is_normalised():
    grid = periodic_grid(-8.0, 8.0, 64)
    psi = gaussian_wavepacket(grid.positions, grid.spacing, 2.0, 0.0, 1.0)
    assert grid.spacing * np.vdot(psi, psi).real == pytest.approx(1.0)

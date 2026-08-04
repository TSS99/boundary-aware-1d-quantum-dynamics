"""Diagnostic measures and observables."""

from __future__ import annotations

import numpy as np
import pytest

from boundary_aware_dynamics.diagnostics import (
    boundary_diagnostics,
    density_errors,
    energy_expectation,
    fidelity,
    momentum_expectation,
    position_expectation,
    state_errors,
    wrap_around_probability,
)
from boundary_aware_dynamics.grids import dirichlet_midpoint_grid, periodic_grid, sine_mode_energies
from boundary_aware_dynamics.propagators import harmonic_potential
from boundary_aware_dynamics.references import sine_basis
from boundary_aware_dynamics.states import gaussian_wavepacket, sine_windowed_gaussian

LENGTH, MASS, HBAR = 10.0, 1.0, 1.0


def test_identical_states_give_zero_error_everywhere():
    grid = dirichlet_midpoint_grid(LENGTH, 64)
    psi = sine_windowed_gaussian(grid.positions, grid.spacing, LENGTH, 5.0, 2.0, 0.8)
    errors = state_errors(psi, psi, grid.spacing)

    assert errors.fidelity == pytest.approx(1.0)
    assert errors.infidelity == pytest.approx(0.0, abs=1e-14)
    assert errors.l2_state_error == pytest.approx(0.0, abs=1e-14)
    assert errors.density_l1_error == pytest.approx(0.0, abs=1e-14)
    assert errors.norm_error == pytest.approx(0.0, abs=1e-14)


def test_a_global_phase_changes_no_physical_measure():
    # Fidelity is phase-invariant by construction; the L2 state error is only
    # phase-invariant because state_errors aligns the phase first.
    grid = dirichlet_midpoint_grid(LENGTH, 64)
    psi = sine_windowed_gaussian(grid.positions, grid.spacing, LENGTH, 5.0, 2.0, 0.8)
    rotated = psi * np.exp(1.1j)

    errors = state_errors(psi, rotated, grid.spacing)
    assert errors.fidelity == pytest.approx(1.0)
    assert errors.l2_state_error < 1e-14
    # Without alignment the naive difference would be large.
    assert np.sqrt(grid.spacing) * np.linalg.norm(rotated - psi) > 1.0


def test_orthogonal_states_have_zero_fidelity():
    grid = dirichlet_midpoint_grid(LENGTH, 64)
    modes = sine_basis(grid.positions, LENGTH, 3)
    assert fidelity(modes[0], modes[2], grid.spacing) == pytest.approx(0.0, abs=1e-12)


def test_fidelity_never_exceeds_one():
    # The old plotting code drew fidelity axes up to 1.002; the quantity itself
    # is bounded and is clipped here so that can never be a real data point.
    grid = dirichlet_midpoint_grid(LENGTH, 32)
    psi = sine_windowed_gaussian(grid.positions, grid.spacing, LENGTH, 5.0, 2.0, 0.8)
    assert fidelity(psi, psi * 1.0000001, grid.spacing) <= 1.0


def test_position_expectation_recovers_the_packet_centre():
    grid = periodic_grid(-20.0, 20.0, 1024)
    psi = gaussian_wavepacket(grid.positions, grid.spacing, 3.0, 0.0, 1.5)
    mean, _, variance = position_expectation(psi, grid)
    assert mean == pytest.approx(3.0, abs=1e-8)
    assert np.sqrt(variance) == pytest.approx(1.5, rel=1e-6)


def test_momentum_expectation_on_a_periodic_grid():
    grid = periodic_grid(-20.0, 20.0, 1024)
    psi = gaussian_wavepacket(grid.positions, grid.spacing, 0.0, 1.5, 1.0)
    assert momentum_expectation(psi, grid, HBAR) == pytest.approx(1.5, abs=1e-6)


def test_momentum_expectation_vanishes_for_a_real_dirichlet_state():
    # A real standing wave carries no net momentum; the sine-basis derivative
    # must reproduce that rather than leaking a spurious value.
    grid = dirichlet_midpoint_grid(LENGTH, 64)
    mode = sine_basis(grid.positions, LENGTH, 3)[2].astype(complex)
    assert momentum_expectation(mode, grid, HBAR) == pytest.approx(0.0, abs=1e-10)


def test_energy_expectation_of_a_well_eigenmode_is_its_eigenvalue():
    grid = dirichlet_midpoint_grid(LENGTH, 64)
    mode = sine_basis(grid.positions, LENGTH, 5)[4].astype(complex)
    kinetic, potential, total = energy_expectation(mode, grid, np.zeros(64), MASS, HBAR)
    expected = sine_mode_energies(64, LENGTH, MASS, HBAR)[4]
    assert kinetic == pytest.approx(expected, rel=1e-10)
    assert potential == pytest.approx(0.0, abs=1e-12)
    assert total == pytest.approx(kinetic)


def test_energy_expectation_of_the_harmonic_ground_state():
    grid = periodic_grid(-12.0, 12.0, 512)
    omega = 1.0
    sigma = np.sqrt(HBAR / (2.0 * MASS * omega))
    psi = gaussian_wavepacket(grid.positions, grid.spacing, 0.0, 0.0, sigma)
    potential = harmonic_potential(grid.positions, MASS, omega)
    kinetic, potential_energy, total = energy_expectation(psi, grid, potential, MASS, HBAR)

    assert total == pytest.approx(0.5 * HBAR * omega, rel=1e-6)
    # Virial theorem: the two halves are equal for the harmonic ground state.
    assert kinetic == pytest.approx(potential_energy, rel=1e-6)


def test_wrap_around_probability_detects_amplitude_at_the_edges():
    grid = periodic_grid(0.0, LENGTH, 128)
    centred = gaussian_wavepacket(grid.positions, grid.spacing, 5.0, 0.0, 0.5)
    at_edge = gaussian_wavepacket(grid.positions, grid.spacing, 0.2, 0.0, 0.5)
    assert wrap_around_probability(centred, grid) < 1e-12
    assert wrap_around_probability(at_edge, grid) > 0.3


def test_wrap_around_fraction_is_validated():
    grid = periodic_grid(0.0, LENGTH, 32)
    psi = gaussian_wavepacket(grid.positions, grid.spacing, 5.0, 0.0, 1.0)
    for bad in (0.0, 0.5, 0.9):
        with pytest.raises(ValueError, match="fraction must lie"):
            wrap_around_probability(psi, grid, bad)


def test_wall_residual_is_small_for_a_state_that_respects_the_walls():
    grid = dirichlet_midpoint_grid(LENGTH, 128)
    respecting = sine_windowed_gaussian(grid.positions, grid.spacing, LENGTH, 5.0, 2.0, 0.8)
    violating = gaussian_wavepacket(grid.positions, grid.spacing, 0.3, 0.0, 0.5)

    assert boundary_diagnostics(respecting, grid, HBAR, MASS)["wall_residual"] < 1e-4
    assert boundary_diagnostics(violating, grid, HBAR, MASS)["wall_residual"] > 0.1


def test_near_wall_probability_is_reported_separately_from_wrap_around():
    # For a hard-wall problem, probability near the wall is physical and must
    # not be labelled leakage; only wrap-around indicates a wrong topology.
    grid = dirichlet_midpoint_grid(LENGTH, 64)
    psi = sine_basis(grid.positions, LENGTH, 1)[0].astype(complex)
    diagnostics = boundary_diagnostics(psi, grid, HBAR, MASS)
    assert set(diagnostics) == {
        "near_wall_probability", "wall_residual", "wrap_around_probability", "left_wall_current"
    }
    assert diagnostics["near_wall_probability"] > 0.0
    assert diagnostics["wall_residual"] < 1e-2


def test_density_errors_ignore_phase():
    grid = dirichlet_midpoint_grid(LENGTH, 64)
    psi = sine_windowed_gaussian(grid.positions, grid.spacing, LENGTH, 5.0, 2.0, 0.8)
    l1, linf = density_errors(psi, psi * np.exp(0.7j), grid.spacing)
    assert l1 == pytest.approx(0.0, abs=1e-14)
    assert linf == pytest.approx(0.0, abs=1e-14)

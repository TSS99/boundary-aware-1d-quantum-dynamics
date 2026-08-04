"""Split-operator propagator physics."""

from __future__ import annotations

import numpy as np
import pytest

from boundary_aware_dynamics.diagnostics import fidelity, state_errors
from boundary_aware_dynamics.grids import dirichlet_midpoint_grid, periodic_grid
from boundary_aware_dynamics.propagators import (
    harmonic_potential,
    one_step_operator,
    split_operator_evolution,
    tilted_potential,
)
from boundary_aware_dynamics.references import exact_discrete_propagation, sine_galerkin_reference
from boundary_aware_dynamics.states import gaussian_wavepacket, sine_windowed_gaussian

LENGTH, MASS, HBAR = 10.0, 1.0, 1.0
TILT = 5.0          # matches configs/paper.yaml


def well_grid(n_grid: int = 64):
    return dirichlet_midpoint_grid(LENGTH, n_grid)


def well_state(grid):
    return sine_windowed_gaussian(grid.positions, grid.spacing, LENGTH, 5.0, 2.0, 0.8)


def well_packet(positions: np.ndarray) -> np.ndarray:
    envelope = np.exp(-((positions - 5.0) ** 2) / (4.0 * 0.8**2))
    return envelope * np.exp(2.0j * (positions - 5.0)) * np.sin(np.pi * positions / LENGTH)


# ------------------------------------------------------------ invariants ----


@pytest.mark.parametrize("boundary", ["periodic", "dirichlet"])
def test_norm_is_conserved(boundary):
    if boundary == "periodic":
        grid = periodic_grid(-8.0, 8.0, 64)
        psi0 = gaussian_wavepacket(grid.positions, grid.spacing, 2.0, 0.0, 1.0)
        potential = harmonic_potential(grid.positions, MASS, 1.0)
    else:
        grid = well_grid()
        psi0, potential = well_state(grid), tilted_potential(grid.positions, LENGTH, TILT)

    run = split_operator_evolution(psi0, grid, potential, 2.0, 50, MASS, HBAR)
    assert run.max_norm_error < 1e-12


@pytest.mark.parametrize("boundary", ["periodic", "dirichlet"])
def test_time_reversal_returns_the_initial_state(boundary):
    if boundary == "periodic":
        grid = periodic_grid(-8.0, 8.0, 64)
        psi0 = gaussian_wavepacket(grid.positions, grid.spacing, 2.0, 0.0, 1.0)
        potential = harmonic_potential(grid.positions, MASS, 1.0)
    else:
        grid = well_grid()
        psi0, potential = well_state(grid), tilted_potential(grid.positions, LENGTH, TILT)

    forward = split_operator_evolution(psi0, grid, potential, 1.5, 40, MASS, HBAR)
    backward = split_operator_evolution(forward.final_state, grid, potential, -1.5, 40, MASS, HBAR)
    assert fidelity(psi0, backward.final_state, grid.spacing) == pytest.approx(1.0, abs=1e-12)


def test_zero_length_interval_leaves_the_state_untouched():
    grid = well_grid()
    psi0 = well_state(grid)
    run = split_operator_evolution(psi0, grid, None, 0.0, 10, MASS, HBAR)
    for state in run.states:
        assert np.allclose(state, psi0, atol=1e-14)


@pytest.mark.parametrize("boundary", ["periodic", "dirichlet"])
def test_loop_matches_the_explicit_one_step_matrix(boundary):
    if boundary == "periodic":
        grid = periodic_grid(-8.0, 8.0, 32)
        psi0 = gaussian_wavepacket(grid.positions, grid.spacing, 2.0, 0.0, 1.0)
        potential = harmonic_potential(grid.positions, MASS, 1.0)
    else:
        grid = well_grid(32)
        psi0, potential = well_state(grid), tilted_potential(grid.positions, LENGTH, TILT)

    run = split_operator_evolution(psi0, grid, potential, 0.4, 1, MASS, HBAR)
    matrix = one_step_operator(grid, potential, 0.4, MASS, HBAR)
    assert np.allclose(run.final_state, matrix @ psi0, atol=1e-12)


def test_potential_shape_is_validated():
    grid = well_grid(32)
    with pytest.raises(ValueError, match="potential must have shape"):
        split_operator_evolution(well_state(grid), grid, np.zeros(8), 1.0, 5, MASS, HBAR)


def test_non_positive_step_count_is_rejected():
    grid = well_grid(32)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        split_operator_evolution(well_state(grid), grid, None, 1.0, 0, MASS, HBAR)


# --------------------------------------------- exactness and Trotter error --


@pytest.mark.parametrize("n_steps", [5, 40, 200])
def test_zero_potential_dirichlet_propagation_is_exact_at_any_step_count(n_steps):
    # The sine transform diagonalises the Dirichlet Laplacian, so with V = 0
    # there is no splitting error at all and the step count is irrelevant.
    # This is precisely why Benchmark B is a control, not a Trotter test.
    grid = well_grid()
    times = np.linspace(0.0, 6.0, n_steps + 1)
    run = split_operator_evolution(well_state(grid), grid, None, 6.0, n_steps, MASS, HBAR)
    reference = sine_galerkin_reference(well_packet, grid, times, MASS, HBAR, n_modes=64)
    assert 1.0 - fidelity(reference.final_state, run.final_state, grid.spacing) < 1e-12


def test_step_count_changes_nothing_for_the_free_well():
    grid = well_grid()
    coarse = split_operator_evolution(well_state(grid), grid, None, 6.0, 10, MASS, HBAR)
    fine = split_operator_evolution(well_state(grid), grid, None, 6.0, 320, MASS, HBAR)
    assert fidelity(coarse.final_state, fine.final_state, grid.spacing) == pytest.approx(1.0, abs=1e-12)


def test_tilted_well_does_have_trotter_error():
    # Adding an interior potential makes [T, V] != 0, so now the step count
    # matters.  Without this the repository has no genuine Trotter benchmark.
    grid = well_grid()
    potential = tilted_potential(grid.positions, LENGTH, TILT)
    coarse = split_operator_evolution(well_state(grid), grid, potential, 2.0, 10, MASS, HBAR)
    fine = split_operator_evolution(well_state(grid), grid, potential, 2.0, 1280, MASS, HBAR)
    assert fidelity(coarse.final_state, fine.final_state, grid.spacing) < 0.99


def test_tilted_well_error_falls_at_second_order():
    # Compared against exact diagonalisation of the *same discrete* Hamiltonian,
    # so spatial resolution and pseudospectral aliasing cancel and only the
    # splitting error is measured. r = 40 is left out because it sits in the
    # pre-asymptotic transient; the configs record the same fit interval.
    grid = well_grid()
    potential = tilted_potential(grid.positions, LENGTH, TILT)
    reference = exact_discrete_propagation(
        well_state(grid), grid, np.array([0.0, 2.0]), MASS, HBAR, potential
    ).final_state

    step_counts = (80, 160, 320, 640)
    errors = []
    for n_steps in step_counts:
        run = split_operator_evolution(well_state(grid), grid, potential, 2.0, n_steps, MASS, HBAR)
        errors.append(state_errors(reference, run.final_state, grid.spacing).l2_state_error)

    ratios = [a / b for a, b in zip(errors, errors[1:])]
    assert all(3.7 < ratio < 4.3 for ratio in ratios), ratios

    slope = np.polyfit(np.log(2.0 / np.array(step_counts)), np.log(errors), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.05)


def test_tilted_well_infidelity_falls_at_fourth_order():
    # Infidelity is the square of the state error to leading order, so a
    # second-order method shows a fourth-order infidelity slope.
    grid = well_grid()
    potential = tilted_potential(grid.positions, LENGTH, TILT)
    reference = exact_discrete_propagation(
        well_state(grid), grid, np.array([0.0, 2.0]), MASS, HBAR, potential
    ).final_state

    step_counts = (80, 160, 320)
    infidelities = [
        state_errors(
            reference,
            split_operator_evolution(well_state(grid), grid, potential, 2.0, n, MASS, HBAR).final_state,
            grid.spacing,
        ).infidelity
        for n in step_counts
    ]
    slope = np.polyfit(np.log(2.0 / np.array(step_counts)), np.log(infidelities), 1)[0]
    assert slope == pytest.approx(4.0, abs=0.15)


def test_exact_discrete_reference_isolates_trotter_error():
    # It must agree with the split-operator result in the limit of many steps,
    # since both describe the same discrete Hamiltonian.
    grid = well_grid()
    potential = tilted_potential(grid.positions, LENGTH, TILT)
    reference = exact_discrete_propagation(
        well_state(grid), grid, np.array([0.0, 2.0]), MASS, HBAR, potential
    ).final_state
    # The floor here is round-off accumulated over 4096 sequential transform
    # pairs (~2e-12), not a modelling error.
    run = split_operator_evolution(well_state(grid), grid, potential, 2.0, 4096, MASS, HBAR)
    assert 1.0 - fidelity(reference, run.final_state, grid.spacing) < 1e-11


def test_exact_discrete_reference_matches_the_free_well_propagator_exactly():
    # With V = 0 the Dirichlet splitting is already exact, so the two must agree
    # to machine precision at any step count.
    grid = well_grid()
    reference = exact_discrete_propagation(
        well_state(grid), grid, np.array([0.0, 6.0]), MASS, HBAR, None
    ).final_state
    run = split_operator_evolution(well_state(grid), grid, None, 6.0, 7, MASS, HBAR)
    assert 1.0 - fidelity(reference, run.final_state, grid.spacing) < 1e-12


# ------------------------------------------------------ eigenstate checks ---


def test_harmonic_ground_state_density_is_stationary():
    grid = periodic_grid(-8.0, 8.0, 128)
    sigma = np.sqrt(HBAR / (2.0 * MASS * 1.0))
    psi0 = gaussian_wavepacket(grid.positions, grid.spacing, 0.0, 0.0, sigma)
    run = split_operator_evolution(
        psi0, grid, harmonic_potential(grid.positions, MASS, 1.0), 2.0 * np.pi, 200, MASS, HBAR
    )
    initial_density = np.abs(psi0) ** 2
    for state in run.states:
        # Stationary up to Trotter and grid error, not exactly; the peak
        # density is ~0.4, so this is a relative deviation below 3e-4.
        assert np.max(np.abs(np.abs(state) ** 2 - initial_density)) < 1e-4


def test_well_eigenmode_only_acquires_a_phase():
    grid = well_grid()
    from boundary_aware_dynamics.grids import sine_mode_energies
    from boundary_aware_dynamics.references import sine_basis

    mode = sine_basis(grid.positions, LENGTH, 4)[3].astype(complex)   # nu = 4
    run = split_operator_evolution(mode, grid, None, 2.0, 25, MASS, HBAR)
    energy = sine_mode_energies(64, LENGTH, MASS, HBAR)[3]
    expected = mode * np.exp(-1j * energy * 2.0 / HBAR)
    assert np.allclose(run.final_state, expected, atol=1e-12)


# ---------------------------------------------------- boundary behaviour ----


def test_periodic_and_dirichlet_propagation_of_the_same_well_state_diverge():
    # The central claim, at the propagator level.
    n_grid = 64
    dirichlet = well_grid(n_grid)
    ring = periodic_grid(0.0, LENGTH, n_grid)
    psi0 = well_state(dirichlet)

    by_dirichlet = split_operator_evolution(psi0, dirichlet, None, 6.0, 100, MASS, HBAR)
    by_periodic = split_operator_evolution(psi0, ring, None, 6.0, 100, MASS, HBAR)

    assert fidelity(by_dirichlet.states[0], by_periodic.states[0], dirichlet.spacing) == pytest.approx(1.0)
    assert fidelity(by_dirichlet.final_state, by_periodic.final_state, dirichlet.spacing) < 0.2


def test_propagation_records_which_transform_it_used():
    grid = well_grid(32)
    run = split_operator_evolution(well_state(grid), grid, None, 1.0, 10, MASS, HBAR)
    assert run.metadata["transform"] == "DST-II"
    assert run.metadata["exact_for_zero_potential"] is True

    ring = periodic_grid(0.0, LENGTH, 32)
    ring_run = split_operator_evolution(well_state(grid), ring, None, 1.0, 10, MASS, HBAR)
    assert ring_run.metadata["transform"] == "FFT"
    assert ring_run.metadata["exact_for_zero_potential"] is False

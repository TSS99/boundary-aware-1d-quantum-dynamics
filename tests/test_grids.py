"""Grid geometry and spectral index conventions."""

from __future__ import annotations

import numpy as np
import pytest

from boundary_aware_dynamics.grids import (
    dirichlet_midpoint_grid,
    fft_kinetic_energies,
    fft_momenta,
    periodic_grid,
    signed_frequency_indices,
    sine_mode_energies,
    sine_mode_indices,
    validate_power_of_two,
)


@pytest.mark.parametrize("n_grid", [0, 3, 6, 12, 100])
def test_non_power_of_two_grids_are_rejected(n_grid):
    with pytest.raises(ValueError, match="power of two"):
        validate_power_of_two(n_grid)


@pytest.mark.parametrize("n_grid", [1, 2, 4, 8, 16, 64, 256])
def test_power_of_two_grids_are_accepted(n_grid):
    validate_power_of_two(n_grid)


def test_periodic_grid_excludes_the_right_endpoint():
    grid = periodic_grid(-8.0, 8.0, 64)
    assert grid.positions[0] == pytest.approx(-8.0)
    assert grid.positions[-1] == pytest.approx(8.0 - grid.spacing)
    assert grid.spacing == pytest.approx(16.0 / 64)
    assert grid.n_data_qubits == 6
    assert grid.boundary == "periodic"


def test_periodic_grid_rejects_inverted_extent():
    with pytest.raises(ValueError, match="x_right must be larger"):
        periodic_grid(1.0, -1.0, 8)


def test_dirichlet_grid_samples_cell_midpoints_and_excludes_both_walls():
    length, n_grid = 10.0, 64
    grid = dirichlet_midpoint_grid(length, n_grid)
    spacing = length / n_grid
    assert grid.positions[0] == pytest.approx(0.5 * spacing)
    assert grid.positions[-1] == pytest.approx(length - 0.5 * spacing)
    assert np.all(grid.positions > 0.0)
    assert np.all(grid.positions < length)
    assert grid.length == pytest.approx(length)


def test_signed_frequency_indices_match_numpy_fftfreq_ordering():
    for n_grid in (8, 16, 64):
        expected = np.fft.fftfreq(n_grid) * n_grid
        assert np.allclose(signed_frequency_indices(n_grid), expected)


def test_signed_frequency_indices_place_nyquist_negative():
    # Bin N/2 must carry -N/2, not +N/2; getting this wrong flips the sign of
    # the highest momentum and silently corrupts the kinetic phase.
    indices = signed_frequency_indices(8)
    assert indices[0] == 0.0
    assert indices[3] == 3.0
    assert indices[4] == -4.0
    assert indices[-1] == -1.0


def test_fft_momenta_follow_the_2pi_hbar_over_L_rule():
    n_grid, spacing, hbar = 16, 0.25, 2.0
    length = n_grid * spacing
    expected = 2.0 * np.pi * hbar * signed_frequency_indices(n_grid) / length
    assert np.allclose(fft_momenta(n_grid, spacing, hbar), expected)


def test_fft_kinetic_energies_are_symmetric_in_momentum():
    energies = fft_kinetic_energies(16, 0.5, mass=1.0, hbar=1.0)
    # p and -p carry the same kinetic energy, so bins k and N-k must agree.
    assert np.allclose(energies[1:8], energies[9:][::-1])


def test_sine_mode_indices_start_at_one():
    # The constant mode is not a Dirichlet eigenfunction, so DST-II output bin k
    # carries mode nu = k + 1.  An off-by-one here misassigns every energy.
    indices = sine_mode_indices(8)
    assert indices[0] == 1.0
    assert indices[-1] == 8.0


def test_sine_mode_energies_match_the_closed_form():
    n_grid, length, mass, hbar = 32, 10.0, 1.5, 2.0
    nu = np.arange(1, n_grid + 1, dtype=float)
    expected = hbar**2 * np.pi**2 * nu**2 / (2.0 * mass * length**2)
    assert np.allclose(sine_mode_energies(n_grid, length, mass, hbar), expected)


def test_sine_mode_ground_energy_is_the_textbook_value():
    # E_1 = hbar^2 pi^2 / (2 m L^2) for the infinite square well.
    energies = sine_mode_energies(16, length=1.0, mass=1.0, hbar=1.0)
    assert energies[0] == pytest.approx(np.pi**2 / 2.0)


@pytest.mark.parametrize("bad_mass", [0.0, -1.0])
def test_non_positive_mass_is_rejected(bad_mass):
    with pytest.raises(ValueError, match="mass must be positive"):
        sine_mode_energies(8, length=1.0, mass=bad_mass, hbar=1.0)

"""Initial states, normalisation conventions and the Gaussian width convention."""

from __future__ import annotations

import numpy as np
import pytest

from boundary_aware_dynamics.grids import dirichlet_midpoint_grid, periodic_grid
from boundary_aware_dynamics.states import (
    align_global_phase,
    amplitudes_to_physical,
    euclidean_norm,
    gaussian_wavepacket,
    physical_to_amplitudes,
    quadrature_norm,
    sine_windowed_gaussian,
)


def test_physical_samples_carry_unit_quadrature_norm():
    grid = periodic_grid(-8.0, 8.0, 64)
    psi = gaussian_wavepacket(grid.positions, grid.spacing, centre=0.0, momentum=1.0, sigma=1.0)
    assert quadrature_norm(psi, grid.spacing) == pytest.approx(1.0)


def test_quantum_amplitudes_carry_unit_euclidean_norm():
    # The conversion sqrt(dx) psi is exactly what makes a quadrature-normalised
    # physical state into a legal statevector.  Handing Qiskit the unconverted
    # array is the mistake this test exists to catch.
    grid = periodic_grid(-8.0, 8.0, 64)
    psi = gaussian_wavepacket(grid.positions, grid.spacing, centre=0.0, momentum=1.0, sigma=1.0)
    amplitudes = physical_to_amplitudes(psi, grid.spacing)

    assert euclidean_norm(amplitudes) == pytest.approx(1.0)
    assert euclidean_norm(psi) != pytest.approx(1.0)   # the unconverted array is not a statevector


def test_amplitude_conversion_round_trips():
    grid = dirichlet_midpoint_grid(10.0, 32)
    psi = gaussian_wavepacket(grid.positions, grid.spacing, centre=5.0, momentum=2.0, sigma=0.8)
    recovered = amplitudes_to_physical(physical_to_amplitudes(psi, grid.spacing), grid.spacing)
    assert np.allclose(recovered, psi)


def test_sigma_is_the_standard_deviation_of_the_density():
    # The 4 sigma^2 convention: |psi|^2 ~ exp[-(x - x0)^2 / (2 sigma^2)], so the
    # measured standard deviation of the density must equal sigma itself.
    # Under a 2 sigma^2 convention this would come out at sigma / sqrt(2).
    grid = periodic_grid(-40.0, 40.0, 4096)
    for sigma in (0.5, 1.0, 2.0):
        psi = gaussian_wavepacket(grid.positions, grid.spacing, centre=3.0, momentum=0.0, sigma=sigma)
        density = np.abs(psi) ** 2
        mean = grid.spacing * np.sum(grid.positions * density)
        variance = grid.spacing * np.sum((grid.positions - mean) ** 2 * density)

        assert mean == pytest.approx(3.0, abs=1e-8)
        assert np.sqrt(variance) == pytest.approx(sigma, rel=1e-6)


def test_momentum_parameter_sets_the_mean_momentum():
    grid = periodic_grid(-40.0, 40.0, 4096)
    momentum = 2.0
    psi = gaussian_wavepacket(grid.positions, grid.spacing, centre=0.0, momentum=momentum, sigma=1.0)
    # Work in amplitudes so |spectrum|^2 is already a unit-sum probability
    # distribution over momentum bins; the physical samples would need a dx.
    amplitudes = physical_to_amplitudes(psi, grid.spacing)
    spectrum = np.fft.fft(amplitudes, norm="ortho")
    momenta = 2.0 * np.pi * np.fft.fftfreq(grid.n_grid, d=grid.spacing)
    mean_momentum = float(np.sum(momenta * np.abs(spectrum) ** 2))
    assert mean_momentum == pytest.approx(momentum, abs=1e-6)


def test_sine_window_suppresses_the_state_at_both_walls():
    length, n_grid = 10.0, 256
    grid = dirichlet_midpoint_grid(length, n_grid)
    psi = sine_windowed_gaussian(
        grid.positions, grid.spacing, length, centre=5.0, momentum=2.0, sigma=0.8
    )
    assert quadrature_norm(psi, grid.spacing) == pytest.approx(1.0)
    # Density in the first and last cell must be negligible for a hard wall.
    assert np.abs(psi[0]) ** 2 < 1e-12
    assert np.abs(psi[-1]) ** 2 < 1e-12


def test_sine_window_rejects_a_centre_outside_the_well():
    grid = dirichlet_midpoint_grid(10.0, 32)
    with pytest.raises(ValueError, match="inside the well"):
        sine_windowed_gaussian(grid.positions, grid.spacing, 10.0, centre=12.0, momentum=0.0, sigma=1.0)


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_non_positive_sigma_is_rejected(sigma):
    grid = periodic_grid(-1.0, 1.0, 8)
    with pytest.raises(ValueError, match="sigma must be positive"):
        gaussian_wavepacket(grid.positions, grid.spacing, centre=0.0, momentum=0.0, sigma=sigma)


def test_global_phase_alignment_removes_an_arbitrary_phase():
    rng = np.random.default_rng(3)
    reference = rng.normal(size=16) + 1j * rng.normal(size=16)
    reference /= np.linalg.norm(reference)
    rotated = reference * np.exp(1j * 0.9)

    # Without alignment the raw L2 difference is large even though the two
    # states are physically identical.
    assert np.linalg.norm(rotated - reference) > 0.5
    assert np.linalg.norm(align_global_phase(rotated, reference) - reference) < 1e-12


def test_global_phase_alignment_leaves_a_genuine_difference_intact():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=16) + 1j * rng.normal(size=16)
    reference /= np.linalg.norm(reference)
    other = rng.normal(size=16) + 1j * rng.normal(size=16)
    other /= np.linalg.norm(other)
    assert np.linalg.norm(align_global_phase(other, reference) - reference) > 0.1

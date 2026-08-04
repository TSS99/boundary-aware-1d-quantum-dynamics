"""Transform conventions, validated against closed forms rather than themselves.

The point of these tests is that SciPy and NumPy are checked against analytical
expressions, not against another call to the same library.  A DST validated only
by another DST call would pass even if the convention were wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from boundary_aware_dynamics.grids import dirichlet_midpoint_grid, sine_mode_energies
from boundary_aware_dynamics.transforms import (
    analytical_dst2_matrix,
    analytical_sine_mode,
    dft_matrix,
    dst2_forward,
    dst2_inverse,
    dst2_matrix,
    fft_forward,
    fft_inverse,
)

GRID_SIZES = [4, 8, 16, 32]


# ---------------------------------------------------------------- Fourier ---


@pytest.mark.parametrize("n_grid", GRID_SIZES)
def test_numpy_fft_matches_the_explicit_dft_matrix(n_grid):
    rng = np.random.default_rng(0)
    vector = rng.normal(size=n_grid) + 1j * rng.normal(size=n_grid)
    assert np.allclose(fft_forward(vector), dft_matrix(n_grid) @ vector)


@pytest.mark.parametrize("n_grid", GRID_SIZES)
def test_fft_is_unitary_and_round_trips(n_grid):
    matrix = dft_matrix(n_grid)
    assert np.allclose(matrix.conj().T @ matrix, np.eye(n_grid))
    rng = np.random.default_rng(1)
    vector = rng.normal(size=n_grid) + 1j * rng.normal(size=n_grid)
    assert np.allclose(fft_inverse(fft_forward(vector)), vector)


def test_fft_maps_a_plane_wave_to_a_single_bin():
    # exp(+2 pi i k0 j / N) must land in bin k0 under the forward convention.
    n_grid, k0 = 16, 3
    j = np.arange(n_grid)
    plane_wave = np.exp(2j * np.pi * k0 * j / n_grid)
    spectrum = np.abs(fft_forward(plane_wave))
    assert np.argmax(spectrum) == k0
    assert spectrum[k0] == pytest.approx(np.sqrt(n_grid))
    assert np.allclose(np.delete(spectrum, k0), 0.0, atol=1e-12)


# ------------------------------------------------------------------- DST ----


@pytest.mark.parametrize("n_grid", GRID_SIZES)
def test_scipy_dst2_matches_the_analytical_closed_form(n_grid):
    # This is the test that pins the convention: SciPy's orthonormal DST-II is
    # compared against sqrt(2/N) sin(pi nu (j + 1/2)/N) with the 1/sqrt(2)
    # correction on the Nyquist row.
    assert np.allclose(dst2_matrix(n_grid), analytical_dst2_matrix(n_grid), atol=1e-13)


@pytest.mark.parametrize("n_grid", GRID_SIZES)
def test_dst2_is_real_orthogonal(n_grid):
    matrix = dst2_matrix(n_grid)
    assert np.allclose(matrix.imag, 0.0)
    assert np.allclose(matrix @ matrix.T, np.eye(n_grid), atol=1e-13)


@pytest.mark.parametrize("n_grid", GRID_SIZES)
def test_dst2_round_trips(n_grid):
    rng = np.random.default_rng(2)
    vector = rng.normal(size=n_grid) + 1j * rng.normal(size=n_grid)
    assert np.allclose(dst2_inverse(dst2_forward(vector)), vector)


def test_dst2_highest_mode_is_the_alternating_nyquist_row():
    n_grid = 16
    row = dst2_matrix(n_grid)[-1]
    expected = ((-1.0) ** np.arange(n_grid)) / np.sqrt(n_grid)
    assert np.allclose(row, expected, atol=1e-13)


@pytest.mark.parametrize("mode_index", [1, 2, 5, 15])
def test_dst2_localises_a_sampled_sine_mode_in_the_right_bin(mode_index):
    # Mode nu must appear in output bin nu - 1.  This is the off-by-one that
    # would silently misassign every kinetic energy.
    n_grid, length = 16, 10.0
    grid = dirichlet_midpoint_grid(length, n_grid)
    samples = analytical_sine_mode(mode_index, n_grid, length) * np.sqrt(grid.spacing)
    spectrum = np.abs(dst2_forward(samples))
    assert np.argmax(spectrum) == mode_index - 1
    assert spectrum[mode_index - 1] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(np.delete(spectrum, mode_index - 1), 0.0, atol=1e-12)


def test_analytical_sine_modes_are_orthonormal_under_quadrature():
    n_grid, length = 32, 10.0
    spacing = length / n_grid
    modes = np.array([analytical_sine_mode(nu, n_grid, length) for nu in range(1, n_grid)])
    gram = spacing * (modes @ modes.T)
    assert np.allclose(gram, np.eye(n_grid - 1), atol=1e-13)


def test_analytical_sine_modes_extrapolate_to_zero_at_both_walls():
    # The midpoint grid never samples the walls, so check the continuum function
    # itself vanishes there by evaluating the closed form at x = 0 and x = L.
    length = 10.0
    for nu in (1, 3, 7):
        assert np.sin(nu * np.pi * 0.0 / length) == pytest.approx(0.0)
        assert np.sin(nu * np.pi * length / length) == pytest.approx(0.0, abs=1e-13)


@pytest.mark.parametrize("mode_index", [1, 4, 9])
def test_dst_propagator_gives_a_sine_mode_exactly_its_eigenphase(mode_index):
    # A Dirichlet eigenmode must evolve as exp(-i E_nu t / hbar) with no change
    # in shape.  This is what makes the zero-potential well propagator exact --
    # and therefore what makes it a control, not a Trotter benchmark.
    n_grid, length, mass, hbar, time = 32, 10.0, 1.0, 1.0, 0.7
    grid = dirichlet_midpoint_grid(length, n_grid)
    energies = sine_mode_energies(n_grid, length, mass, hbar)
    samples = analytical_sine_mode(mode_index, n_grid, length) * np.sqrt(grid.spacing)

    evolved = dst2_inverse(dst2_forward(samples) * np.exp(-1j * energies * time / hbar))
    expected = samples * np.exp(-1j * energies[mode_index - 1] * time / hbar)
    assert np.allclose(evolved, expected, atol=1e-13)


def test_dst_and_fft_propagation_disagree_for_a_hard_wall_state():
    # The central physical claim of the project, as a regression guard: the two
    # transforms represent different boundary topologies, so they must give
    # materially different dynamics once the packet reaches the wall.
    n_grid, length, mass, hbar = 64, 10.0, 1.0, 1.0
    grid = dirichlet_midpoint_grid(length, n_grid)
    positions, spacing = grid.positions, grid.spacing

    packet = np.exp(-((positions - 5.0) ** 2) / (4.0 * 0.8**2)) * np.exp(2j * (positions - 5.0))
    packet *= np.sin(np.pi * positions / length)
    packet /= np.sqrt(spacing * np.vdot(packet, packet).real)

    energies = sine_mode_energies(n_grid, length, mass, hbar)
    momenta = 2.0 * np.pi * hbar * np.fft.fftfreq(n_grid, d=spacing)

    def fidelity(time: float) -> float:
        by_dst = dst2_inverse(dst2_forward(packet) * np.exp(-1j * energies * time / hbar))
        by_fft = fft_inverse(fft_forward(packet) * np.exp(-1j * (momenta**2 / (2 * mass)) * time / hbar))
        return float(abs(spacing * np.vdot(by_dst, by_fft)) ** 2)

    assert fidelity(0.0) == pytest.approx(1.0, abs=1e-12)
    assert fidelity(2.0) < 0.7   # packet is reaching the wall
    assert fidelity(6.0) < 0.2   # after wall interaction the two models diverge

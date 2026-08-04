"""Spatial grids and spectral index conventions.

Two grid families are used, one per boundary condition:

Periodic grid
    ``x_j = x_left + j * dx``,  ``dx = (x_right - x_left) / N``,  ``j = 0..N-1``.
    The right endpoint is excluded, which is the grid natural to a discrete
    Fourier transform on ``N`` points and hence to a QFT register.

Dirichlet midpoint grid
    ``x_j = (j + 1/2) * dx``,  ``dx = L / N``,  ``j = 0..N-1``.
    Both walls are excluded and the samples sit at cell midpoints.  This is the
    grid on which the orthonormal DST-II diagonalises the Dirichlet Laplacian
    exactly (see :mod:`boundary_aware_dynamics.transforms`).

Notation follows the repository convention: ``N`` grid points, ``n_q = log2 N``
data qubits, ``nu = 1..N`` sine-mode index, ``k`` unsigned FFT bin index and
``k_tilde`` the signed frequency index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

BoundaryKind = Literal["periodic", "dirichlet"]


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def validate_power_of_two(n_grid: int) -> None:
    """Reject grid sizes that cannot be addressed by a whole qubit register."""
    if not is_power_of_two(int(n_grid)):
        raise ValueError(f"Grid size must be a power of two, received {n_grid}.")


@dataclass(frozen=True)
class Grid:
    """A one-dimensional sampling grid together with its boundary model."""

    positions: np.ndarray
    spacing: float
    boundary: BoundaryKind
    extent: tuple[float, float]

    @property
    def n_grid(self) -> int:
        return int(self.positions.size)

    @property
    def n_data_qubits(self) -> int:
        return int(np.log2(self.n_grid))

    @property
    def length(self) -> float:
        return float(self.extent[1] - self.extent[0])


def periodic_grid(x_left: float, x_right: float, n_grid: int) -> Grid:
    """Return the periodic grid matched to an ``N``-point DFT / QFT register."""
    validate_power_of_two(n_grid)
    if x_right <= x_left:
        raise ValueError("x_right must be larger than x_left.")
    spacing = (x_right - x_left) / n_grid
    positions = x_left + spacing * np.arange(n_grid, dtype=float)
    return Grid(positions, spacing, "periodic", (float(x_left), float(x_right)))


def dirichlet_midpoint_grid(length: float, n_grid: int) -> Grid:
    """Return the midpoint grid on ``(0, L)`` matched to the orthonormal DST-II."""
    validate_power_of_two(n_grid)
    if length <= 0:
        raise ValueError("length must be positive.")
    spacing = length / n_grid
    positions = (np.arange(n_grid, dtype=float) + 0.5) * spacing
    return Grid(positions, spacing, "dirichlet", (0.0, float(length)))


def signed_frequency_indices(n_grid: int) -> np.ndarray:
    """Return ``k_tilde``: ``k`` for ``k < N/2`` and ``k - N`` otherwise.

    This is the signed index ordering used by :func:`numpy.fft.fftfreq`, and is
    the ordering a QFT register must reproduce for the kinetic phase to be
    applied to the correct momentum.
    """
    validate_power_of_two(n_grid)
    k = np.arange(n_grid)
    return np.where(k < n_grid // 2, k, k - n_grid).astype(float)


def fft_momenta(n_grid: int, spacing: float, hbar: float) -> np.ndarray:
    """Return ``p_k = (2 pi hbar / L) * k_tilde`` in FFT bin order, ``L = N dx``."""
    if spacing <= 0:
        raise ValueError("spacing must be positive.")
    return 2.0 * np.pi * hbar * signed_frequency_indices(n_grid) / (n_grid * spacing)


def fft_kinetic_energies(n_grid: int, spacing: float, mass: float, hbar: float) -> np.ndarray:
    """Return ``p^2 / 2m`` on the periodic momentum grid, in FFT bin order."""
    if mass <= 0:
        raise ValueError("mass must be positive.")
    momenta = fft_momenta(n_grid, spacing, hbar)
    return momenta**2 / (2.0 * mass)


def sine_mode_indices(n_modes: int) -> np.ndarray:
    """Return the Dirichlet sine-mode indices ``nu = 1, ..., n_modes``.

    The offset matters: DST-II output bin ``k`` carries mode ``nu = k + 1``,
    because the constant mode is not a Dirichlet eigenfunction.

    No power-of-two constraint applies here.  That constraint belongs to grids
    and qubit registers; a reference solution may use any number of sine modes,
    and usually needs more than the simulation grid can hold.
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be at least 1, received {n_modes}.")
    return np.arange(1, n_modes + 1, dtype=float)


def sine_mode_energies(n_modes: int, length: float, mass: float, hbar: float) -> np.ndarray:
    """Return ``E_nu = hbar^2 pi^2 nu^2 / (2 m L^2)`` for ``nu = 1..n_modes``.

    These are the continuum hard-wall eigenvalues.  Using them makes the
    zero-potential Dirichlet propagator exact within the truncated sine basis,
    which is why that benchmark is a control rather than a Trotter test.
    """
    if mass <= 0:
        raise ValueError("mass must be positive.")
    if length <= 0:
        raise ValueError("length must be positive.")
    nu = sine_mode_indices(n_modes)
    return (hbar**2 * np.pi**2 * nu**2) / (2.0 * mass * length**2)

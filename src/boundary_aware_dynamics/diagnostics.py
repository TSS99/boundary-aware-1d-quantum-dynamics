"""Diagnostics beyond fidelity.

Fidelity alone is a poor diagnostic here: for the zero-potential well it is
identically one by construction, and near unity it compresses exactly the range
where the interesting differences live.  These functions therefore report
infidelity, phase-aligned L2 state error and density errors alongside it, plus
observables and boundary-specific quantities.

Terminology
-----------
For a hard-wall system, probability near the wall is *not* leakage: the exact
solution has probability there too.  It is reported as ``near_wall_probability``.
What would indicate a wrong boundary model is ``wrap_around_probability``,
i.e. amplitude that has left one end of the box and reappeared at the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grids import Grid
from .states import align_global_phase

__all__ = [
    "StateErrors",
    "boundary_diagnostics",
    "density_errors",
    "energy_expectation",
    "fidelity",
    "momentum_expectation",
    "position_expectation",
    "state_errors",
    "wrap_around_probability",
]


@dataclass(frozen=True)
class StateErrors:
    """A family of error measures between a numerical and a reference state."""

    fidelity: float
    infidelity: float
    l2_state_error: float
    max_amplitude_error: float
    density_l1_error: float
    density_linf_error: float
    norm_error: float


def fidelity(reference: np.ndarray, state: np.ndarray, spacing: float) -> float:
    """Return ``|<ref|state>|^2`` with the quadrature weight, clipped to [0, 1]."""
    overlap = spacing * np.vdot(reference, state)
    return float(np.clip(np.abs(overlap) ** 2, 0.0, 1.0))


def state_errors(reference: np.ndarray, state: np.ndarray, spacing: float) -> StateErrors:
    """Return fidelity plus the error measures that stay informative near unity."""
    value = fidelity(reference, state, spacing)
    aligned = align_global_phase(state, reference)
    difference = aligned - reference

    reference_density = np.abs(reference) ** 2
    state_density = np.abs(state) ** 2
    density_difference = np.abs(state_density - reference_density)

    return StateErrors(
        fidelity=value,
        infidelity=max(1.0 - value, 0.0),
        l2_state_error=float(np.sqrt(spacing * np.vdot(difference, difference).real)),
        max_amplitude_error=float(np.max(np.abs(difference))),
        density_l1_error=float(spacing * np.sum(density_difference)),
        density_linf_error=float(np.max(density_difference)),
        norm_error=float(abs(spacing * np.vdot(state, state).real - 1.0)),
    )


def position_expectation(state: np.ndarray, grid: Grid) -> tuple[float, float, float]:
    """Return ``(<x>, <x^2>, Var[x])``."""
    density = np.abs(state) ** 2
    mean = float(grid.spacing * np.sum(grid.positions * density))
    mean_square = float(grid.spacing * np.sum(grid.positions**2 * density))
    return mean, mean_square, max(mean_square - mean**2, 0.0)


def momentum_expectation(state: np.ndarray, grid: Grid, hbar: float) -> float:
    """Return ``<p>`` via a spectral derivative appropriate to the boundary.

    On a periodic grid the FFT derivative is exact.  On a Dirichlet grid the
    sine basis is used, which is exact for states satisfying the wall condition.
    """
    from .transforms import fft_forward, fft_inverse

    if grid.boundary == "periodic":
        momenta = 2.0 * np.pi * hbar * np.fft.fftfreq(grid.n_grid, d=grid.spacing)
        derivative = fft_inverse(1j * momenta / hbar * fft_forward(state))
    else:
        # The derivative of a sine series is a *cosine* series, which the inverse
        # DST cannot represent, so project onto the sine modes explicitly and
        # differentiate mode by mode.
        nu = np.arange(1, grid.n_grid + 1, dtype=float)[:, None]
        argument = nu * np.pi * grid.positions[None, :] / grid.length
        sine_modes = np.sqrt(2.0 / grid.length) * np.sin(argument)
        cosine_modes = np.sqrt(2.0 / grid.length) * (nu * np.pi / grid.length) * np.cos(argument)
        coefficients = grid.spacing * (sine_modes @ state)
        derivative = coefficients @ cosine_modes
    return float(np.real(grid.spacing * np.vdot(state, -1j * hbar * derivative)))


def energy_expectation(
    state: np.ndarray,
    grid: Grid,
    potential: np.ndarray,
    mass: float,
    hbar: float,
) -> tuple[float, float, float]:
    """Return ``(kinetic, potential, total)`` energy expectations."""
    from .grids import fft_kinetic_energies, sine_mode_energies
    from .transforms import dst2_forward, fft_forward

    if grid.boundary == "periodic":
        spectrum = fft_forward(state)
        energies = fft_kinetic_energies(grid.n_grid, grid.spacing, mass, hbar)
    else:
        spectrum = dst2_forward(state)
        energies = sine_mode_energies(grid.n_grid, grid.length, mass, hbar)

    kinetic = float(grid.spacing * np.sum(energies * np.abs(spectrum) ** 2))
    potential_energy = float(grid.spacing * np.sum(potential * np.abs(state) ** 2))
    return kinetic, potential_energy, kinetic + potential_energy


def wrap_around_probability(state: np.ndarray, grid: Grid, fraction: float = 0.05) -> float:
    """Return the probability found in the outermost ``fraction`` of each end.

    For a hard-wall problem propagated with the wrong (periodic) topology,
    amplitude leaving one end reappears at the other; comparing this quantity
    between the two propagators makes the topology error visible.
    """
    if not 0.0 < fraction < 0.5:
        raise ValueError("fraction must lie in (0, 0.5).")
    width = max(1, int(round(fraction * grid.n_grid)))
    density = np.abs(state) ** 2
    return float(grid.spacing * (np.sum(density[:width]) + np.sum(density[-width:])))


def boundary_diagnostics(
    state: np.ndarray,
    grid: Grid,
    hbar: float,
    mass: float,
    fraction: float = 0.05,
) -> dict[str, float]:
    """Return boundary-sensitive quantities for one state.

    ``wall_residual`` extrapolates the state to the walls from the two nearest
    midpoints; for an exact Dirichlet solution it should be small, whereas a
    periodic propagator has no reason to keep it small.
    """
    density = np.abs(state) ** 2
    left_wall = 1.5 * state[0] - 0.5 * state[1]
    right_wall = 1.5 * state[-1] - 0.5 * state[-2]

    width = max(1, int(round(fraction * grid.n_grid)))
    probability_current = float(
        (hbar / mass) * np.imag(np.conjugate(state[0]) * (state[1] - state[0]) / grid.spacing)
    )

    return {
        "near_wall_probability": float(grid.spacing * (np.sum(density[:width]) + np.sum(density[-width:]))),
        "wall_residual": float(max(abs(left_wall), abs(right_wall))),
        "wrap_around_probability": wrap_around_probability(state, grid, fraction),
        "left_wall_current": probability_current,
    }


def density_errors(reference: np.ndarray, state: np.ndarray, spacing: float) -> tuple[float, float]:
    """Return ``(L1, Linf)`` errors between the two probability densities."""
    difference = np.abs(np.abs(state) ** 2 - np.abs(reference) ** 2)
    return float(spacing * np.sum(difference)), float(np.max(difference))

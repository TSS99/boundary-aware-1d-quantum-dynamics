"""Second-order (Strang) split-operator propagators.

Two propagators differing only in which spectral transform sits between the
potential half-steps, and therefore in which boundary condition they impose:

Periodic::

    U(dt) = e^{-iV dt / 2 hbar} F^{-1} e^{-iT dt / hbar} F e^{-iV dt / 2 hbar}

Dirichlet::

    U(dt) = e^{-iV dt / 2 hbar} S^{-1} e^{-iT_D dt / hbar} S e^{-iV dt / 2 hbar}

With ``V = 0`` the Dirichlet propagator is *exact*, not second-order: the sine
transform diagonalises the Dirichlet Laplacian, so there is no non-commuting
splitting left to make an error.  This is why the zero-potential well is a
control benchmark and the tilted well is the Trotter benchmark.

Composition note
----------------
These functions apply the full symmetric step each time, which stores the state
at every step boundary.  Resource counting must *not* copy that structure: the
adjacent half-potential phases of consecutive steps merge, so ``r`` steps need
one initial half-phase, ``r - 1`` full phases and one final half-phase.  See
:mod:`boundary_aware_dynamics.circuits.resources`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .grids import Grid, fft_kinetic_energies, sine_mode_energies
from .states import normalise_physical, quadrature_norm
from .transforms import dst2_forward, dst2_inverse, fft_forward, fft_inverse

NORM_TOLERANCE = 1e-10


@dataclass
class Propagation:
    """States and norms from one split-operator run."""

    grid: Grid
    times: np.ndarray
    states: np.ndarray
    norms: np.ndarray
    boundary: str
    n_steps: int
    time_step: float
    potential: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_state(self) -> np.ndarray:
        return self.states[-1]

    @property
    def max_norm_error(self) -> float:
        return float(np.max(np.abs(self.norms - 1.0)))


def harmonic_potential(positions: np.ndarray, mass: float, omega: float) -> np.ndarray:
    """``V(x) = m omega^2 x^2 / 2``."""
    return 0.5 * mass * omega**2 * positions**2


def tilted_potential(positions: np.ndarray, length: float, tilt_force: float) -> np.ndarray:
    """``V(x) = F (x - L/2)``, the linear tilt used for Benchmark C.

    Chosen because it is nonzero in the interior (so ``[T, V] != 0`` and the
    Strang splitting has genuine second-order error) while remaining compatible
    with hard walls and having a closed-form sine-basis matrix element.
    """
    return tilt_force * (positions - 0.5 * length)


def _kinetic_energies(grid: Grid, mass: float, hbar: float) -> np.ndarray:
    if grid.boundary == "periodic":
        return fft_kinetic_energies(grid.n_grid, grid.spacing, mass, hbar)
    return sine_mode_energies(grid.n_grid, grid.length, mass, hbar)


def _transform_pair(boundary: str):
    if boundary == "periodic":
        return fft_forward, fft_inverse
    return dst2_forward, dst2_inverse


def strang_step(
    psi: np.ndarray,
    half_potential_phase: np.ndarray,
    kinetic_phase: np.ndarray,
    boundary: str,
) -> np.ndarray:
    """Apply one symmetric split-operator step."""
    forward, inverse = _transform_pair(boundary)
    psi = half_potential_phase * psi
    psi = inverse(forward(psi) * kinetic_phase)
    return half_potential_phase * psi


def split_operator_evolution(
    psi0: np.ndarray,
    grid: Grid,
    potential: np.ndarray | None,
    t_max: float,
    n_steps: int,
    mass: float,
    hbar: float,
) -> Propagation:
    """Propagate ``psi0`` for ``n_steps`` Strang steps on ``grid``.

    The transform family follows ``grid.boundary``, so the boundary condition is
    selected by the grid rather than by the caller remembering to pass a
    matching transform.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    potential_values = (
        np.zeros(grid.n_grid) if potential is None else np.asarray(potential, dtype=float)
    )
    if potential_values.shape != (grid.n_grid,):
        raise ValueError(
            f"potential must have shape ({grid.n_grid},), got {potential_values.shape}."
        )

    times = np.linspace(0.0, t_max, n_steps + 1)
    time_step = float(times[1] - times[0])
    kinetic = _kinetic_energies(grid, mass, hbar)

    half_potential_phase = np.exp(-0.5j * potential_values * time_step / hbar)
    kinetic_phase = np.exp(-1j * kinetic * time_step / hbar)

    psi = normalise_physical(psi0, grid.spacing)
    states = np.empty((n_steps + 1, grid.n_grid), dtype=np.complex128)
    norms = np.empty(n_steps + 1, dtype=float)
    states[0], norms[0] = psi, quadrature_norm(psi, grid.spacing) ** 2

    for step in range(1, n_steps + 1):
        psi = strang_step(psi, half_potential_phase, kinetic_phase, grid.boundary)
        states[step] = psi
        norms[step] = quadrature_norm(psi, grid.spacing) ** 2

    propagation = Propagation(
        grid=grid,
        times=times,
        states=states,
        norms=norms,
        boundary=grid.boundary,
        n_steps=n_steps,
        time_step=time_step,
        potential=potential_values,
        metadata={
            "transform": "FFT" if grid.boundary == "periodic" else "DST-II",
            "exact_for_zero_potential": grid.boundary == "dirichlet",
        },
    )
    if propagation.max_norm_error > NORM_TOLERANCE:
        raise RuntimeError(
            f"Split-operator norm drift {propagation.max_norm_error:.3e} exceeds "
            f"tolerance {NORM_TOLERANCE:.1e}."
        )
    return propagation


def one_step_operator(
    grid: Grid,
    potential: np.ndarray | None,
    time_step: float,
    mass: float,
    hbar: float,
) -> np.ndarray:
    """Return the explicit ``N x N`` matrix of a single Strang step.

    Used to check the loop implementation against an independent construction of
    the same operator.
    """
    potential_values = (
        np.zeros(grid.n_grid) if potential is None else np.asarray(potential, dtype=float)
    )
    forward, _ = _transform_pair(grid.boundary)
    kinetic = _kinetic_energies(grid, mass, hbar)

    transform = np.asarray([forward(row) for row in np.eye(grid.n_grid)]).T
    half_phase = np.exp(-0.5j * potential_values * time_step / hbar)
    kinetic_phase = np.exp(-1j * kinetic * time_step / hbar)

    kinetic_operator = transform.conj().T @ (kinetic_phase[:, None] * transform)
    return half_phase[:, None] * kinetic_operator * half_phase[None, :]

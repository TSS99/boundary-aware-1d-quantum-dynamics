"""Independent reference solutions.

A reference is only useful if it is independent of the method under test.  Three
are provided, and which one is appropriate depends on the benchmark:

``harmonic_reference``
    Analytical Hermite eigenbasis on a dense grid.  Independent of the FFT
    split-operator propagator being tested.

``sine_galerkin_reference``
    Sine-Galerkin Hamiltonian matrix, diagonalised exactly and propagated by
    ``U exp(-i w t) U^T``.  For the tilted well the potential matrix elements are
    available in closed form, so no quadrature error enters.  This shares the
    sine *basis* with the Dirichlet propagator but not the *method*: it never
    splits the Hamiltonian, so it measures Trotter error.

``finite_difference_reference``
    Second-order finite-difference Hamiltonian with Dirichlet rows removed,
    diagonalised exactly.  This shares neither basis nor method with the sine
    propagator and is the genuinely independent check required for the
    zero-potential well, where a sine-basis reference would be circular.

Truncation is reported rather than hidden: coefficient norms before
renormalisation, retained mode counts and discarded tail weights are all
returned so that reference error can be entered into the error budget.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .grids import Grid, sine_mode_energies
from .states import normalise_physical

__all__ = [
    "ReferenceSolution",
    "exact_discrete_propagation",
    "finite_difference_reference",
    "harmonic_reference",
    "hermite_basis",
    "sine_basis",
    "sine_galerkin_hamiltonian",
    "sine_galerkin_reference",
    "tilt_matrix_elements",
]


@dataclass
class ReferenceSolution:
    """Reference states on the simulation grid, with truncation diagnostics."""

    states: np.ndarray
    times: np.ndarray
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def final_state(self) -> np.ndarray:
        return self.states[-1]


# --------------------------------------------------------------- bases ------


def hermite_basis(positions: np.ndarray, n_modes: int, mass: float, omega: float, hbar: float) -> np.ndarray:
    """Return normalised harmonic eigenfunctions by stable upward recurrence."""
    alpha = mass * omega / hbar
    scaled = np.sqrt(alpha) * positions
    basis = np.empty((n_modes, positions.size), dtype=float)
    basis[0] = (alpha / np.pi) ** 0.25 * np.exp(-0.5 * scaled**2)
    if n_modes > 1:
        basis[1] = np.sqrt(2.0) * scaled * basis[0]
        for n in range(1, n_modes - 1):
            basis[n + 1] = (
                np.sqrt(2.0 / (n + 1)) * scaled * basis[n] - np.sqrt(n / (n + 1)) * basis[n - 1]
            )
    return basis


def sine_basis(positions: np.ndarray, length: float, n_modes: int) -> np.ndarray:
    """Return ``sqrt(2/L) sin(nu pi x / L)`` for ``nu = 1..n_modes``."""
    nu = np.arange(1, n_modes + 1, dtype=float)[:, None]
    return np.sqrt(2.0 / length) * np.sin(nu * np.pi * positions[None, :] / length)


# ------------------------------------------------------ sine-Galerkin -------


def tilt_matrix_elements(n_modes: int, length: float, tilt_force: float) -> np.ndarray:
    """Return ``<nu| F (x - L/2) |nu'>`` in the Dirichlet sine basis, in closed form.

    Evaluating

        (2/L) int_0^L sin(nu pi x/L) sin(nu' pi x/L) F (x - L/2) dx

    with the product-to-sum identity gives

        V[nu, nu'] = (2 F L / pi^2) [ (nu + nu')^-2 - (nu - nu')^-2 ]

    when ``nu + nu'`` is odd, and zero otherwise.  In particular the diagonal
    vanishes, because ``sin^2`` is symmetric about ``L/2`` while the tilt is
    antisymmetric there.
    """
    nu = np.arange(1, n_modes + 1, dtype=float)
    total = nu[:, None] + nu[None, :]
    difference = nu[:, None] - nu[None, :]

    # nu + nu' odd implies nu != nu', so the difference never vanishes where the
    # element is nonzero; substitute 1 elsewhere purely to keep the divide safe.
    odd = (total.astype(int) % 2) == 1
    safe_difference = np.where(odd, difference, 1.0)
    elements = (2.0 * tilt_force * length / np.pi**2) * (1.0 / total**2 - 1.0 / safe_difference**2)
    return np.where(odd, elements, 0.0)


def sine_galerkin_hamiltonian(
    n_modes: int,
    length: float,
    mass: float,
    hbar: float,
    tilt_force: float = 0.0,
) -> np.ndarray:
    """Return the Dirichlet Hamiltonian in the sine basis: diagonal ``T`` plus tilt ``V``."""
    hamiltonian = np.diag(sine_mode_energies(n_modes, length, mass, hbar))
    if tilt_force:
        hamiltonian = hamiltonian + tilt_matrix_elements(n_modes, length, tilt_force)
    return hamiltonian


def sine_galerkin_reference(
    initial_state: Callable[[np.ndarray], np.ndarray],
    grid: Grid,
    times: np.ndarray,
    mass: float,
    hbar: float,
    n_modes: int,
    tilt_force: float = 0.0,
    dense_grid_size: int = 4097,
) -> ReferenceSolution:
    """Propagate exactly in a truncated sine basis via eigendecomposition.

    ``initial_state`` is a callable evaluated on a dense grid, so the projection
    is accurate for ``n_modes`` well beyond the simulation grid size.  Projecting
    on the simulation grid instead would alias every mode above ``N``.  The
    Hamiltonian is then diagonalised and each requested time evaluated as
    ``U exp(-i w t) U^T c``; no time stepping is involved, so this carries no
    Trotter error.
    """
    length = grid.length
    dense_positions = np.linspace(0.0, length, dense_grid_size)
    dense_state = np.asarray(initial_state(dense_positions), dtype=np.complex128)
    dense_state /= np.sqrt(np.trapezoid(np.abs(dense_state) ** 2, dense_positions))

    dense_basis = sine_basis(dense_positions, length, n_modes)
    coefficients = np.trapezoid(dense_basis * dense_state, dense_positions, axis=1)

    raw_weight = float(np.sum(np.abs(coefficients) ** 2))
    tail_weight = max(1.0 - raw_weight, 0.0)
    basis = sine_basis(grid.positions, length, n_modes)

    hamiltonian = sine_galerkin_hamiltonian(n_modes, length, mass, hbar, tilt_force)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    spectral_coefficients = eigenvectors.T @ coefficients

    phases = np.exp(-1j * eigenvalues[None, :] * times[:, None] / hbar)
    evolved = (phases * spectral_coefficients[None, :]) @ eigenvectors.T
    states = evolved @ basis

    return ReferenceSolution(
        states=np.asarray(
            [normalise_physical(state, grid.spacing) for state in states], dtype=np.complex128
        ),
        times=times,
        method="sine_galerkin_exact_diagonalisation",
        diagnostics={
            "n_modes": n_modes,
            "raw_coefficient_norm": raw_weight,
            "tail_weight": tail_weight,
            "tilt_force": tilt_force,
            "shares_basis_with_dirichlet_propagator": True,
            "shares_method_with_split_operator": False,
        },
    )


def exact_discrete_propagation(
    psi0: np.ndarray,
    grid: Grid,
    times: np.ndarray,
    mass: float,
    hbar: float,
    potential: np.ndarray | None = None,
) -> ReferenceSolution:
    """Propagate the *discretised* Hamiltonian exactly, without splitting it.

    Builds ``H = S^T diag(E_nu) S + diag(V(x_j))`` (or the FFT analogue on a
    periodic grid) and evaluates ``exp(-i H t / hbar)`` by eigendecomposition.

    This is the reference that isolates **Trotter error alone**: it uses exactly
    the same spatial discretisation as the split-operator propagator, so grid
    resolution, basis truncation and pseudospectral aliasing all cancel and only
    the splitting error remains.  Comparing against a continuum reference
    instead measures the *total* error, which is the right quantity for the
    error budget but the wrong one for verifying the order of the method.
    """
    from .grids import fft_kinetic_energies
    from .transforms import analytical_dst2_matrix, dft_matrix

    potential_values = (
        np.zeros(grid.n_grid) if potential is None else np.asarray(potential, dtype=float)
    )
    if grid.boundary == "periodic":
        transform = dft_matrix(grid.n_grid)
        energies = fft_kinetic_energies(grid.n_grid, grid.spacing, mass, hbar)
    else:
        transform = analytical_dst2_matrix(grid.n_grid).astype(complex)
        energies = sine_mode_energies(grid.n_grid, grid.length, mass, hbar)

    hamiltonian = transform.conj().T @ (energies[:, None] * transform) + np.diag(potential_values)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)   # symmetrise away round-off
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    coefficients = eigenvectors.conj().T @ normalise_physical(psi0, grid.spacing)
    phases = np.exp(-1j * eigenvalues[None, :] * times[:, None] / hbar)
    states = (phases * coefficients[None, :]) @ eigenvectors.T

    return ReferenceSolution(
        states=np.asarray(
            [normalise_physical(state, grid.spacing) for state in states], dtype=np.complex128
        ),
        times=times,
        method="exact_diagonalisation_of_the_discrete_hamiltonian",
        diagnostics={
            "isolates": "trotter_error_only",
            "shares_discretisation_with_propagator": True,
            "shares_method_with_split_operator": False,
        },
    )


# -------------------------------------------------- finite difference -------


def finite_difference_reference(
    initial_state: Callable[[np.ndarray], np.ndarray],
    grid: Grid,
    times: np.ndarray,
    mass: float,
    hbar: float,
    potential: Callable[[np.ndarray], np.ndarray] | None = None,
    refinement: int = 8,
) -> ReferenceSolution:
    """Propagate exactly on a refined finite-difference Dirichlet grid.

    Uses the standard three-point Laplacian on an interior grid with the wall
    values held at zero, diagonalised exactly.  This shares neither the sine
    basis nor the split-operator method with the propagator under test, so it is
    the appropriate reference for the zero-potential well, where a sine-basis
    reference would be circular.

    The fine grid has spacing ``L / (2 N k)`` so that every simulation midpoint
    ``(j + 1/2) L / N`` coincides exactly with fine node ``(2j + 1) k``.  The
    result is read off at those nodes, so no interpolation error is introduced
    in either direction.
    """
    if refinement < 1:
        raise ValueError("refinement must be at least 1.")

    length = grid.length
    spacing = length / (2 * grid.n_grid * refinement)
    n_fine = 2 * grid.n_grid * refinement - 1
    positions = spacing * np.arange(1, n_fine + 1)
    sample_nodes = (2 * np.arange(grid.n_grid) + 1) * refinement - 1

    off_diagonal = -np.ones(n_fine - 1)
    laplacian = (
        np.diag(2.0 * np.ones(n_fine)) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    ) / spacing**2
    hamiltonian = (hbar**2 / (2.0 * mass)) * laplacian
    if potential is not None:
        hamiltonian = hamiltonian + np.diag(np.asarray(potential(positions), dtype=float))

    psi0_fine = normalise_physical(
        np.asarray(initial_state(positions), dtype=np.complex128), spacing
    )

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    coefficients = eigenvectors.T @ psi0_fine
    phases = np.exp(-1j * eigenvalues[None, :] * times[:, None] / hbar)
    evolved_fine = (phases * coefficients[None, :]) @ eigenvectors.T

    states = np.asarray(
        [normalise_physical(state[sample_nodes], grid.spacing) for state in evolved_fine],
        dtype=np.complex128,
    )
    return ReferenceSolution(
        states=states,
        times=times,
        method="finite_difference_exact_diagonalisation",
        diagnostics={
            "refinement": refinement,
            "n_fine": n_fine,
            "fine_spacing": spacing,
            "shares_basis_with_dirichlet_propagator": False,
            "shares_method_with_split_operator": False,
        },
    )


# ------------------------------------------------------------ harmonic ------


def harmonic_reference(
    grid: Grid,
    times: np.ndarray,
    centre: float,
    momentum: float,
    sigma: float,
    mass: float,
    omega: float,
    hbar: float,
    dense_grid_size: int,
    basis_cap: int,
    tail_tolerance: float,
) -> ReferenceSolution:
    """Expand the initial Gaussian in Hermite eigenstates and evolve analytically."""
    from .states import gaussian_wavepacket

    x_left, x_right = grid.extent
    dense_positions = np.linspace(x_left, x_right, dense_grid_size)
    dense_spacing = float(dense_positions[1] - dense_positions[0])
    psi0_dense = gaussian_wavepacket(dense_positions, dense_spacing, centre, momentum, sigma)

    dense_basis = hermite_basis(dense_positions, basis_cap, mass, omega, hbar)
    coefficients = np.trapezoid(dense_basis * psi0_dense, dense_positions, axis=1)

    weights = np.abs(coefficients) ** 2
    total_weight = float(np.sum(weights))
    cumulative = np.cumsum(weights)
    n_keep = int(np.searchsorted(cumulative, (1.0 - tail_tolerance) * total_weight) + 1)
    if n_keep >= basis_cap - 2:
        raise RuntimeError(
            f"Hermite basis cap {basis_cap} exhausted (kept {n_keep}); increase basis_cap."
        )
    tail_weight = max(total_weight - float(cumulative[n_keep - 1]), 0.0)

    kept = coefficients[:n_keep].astype(np.complex128)
    energies = hbar * omega * (np.arange(n_keep, dtype=float) + 0.5)
    basis_sim = hermite_basis(grid.positions, n_keep, mass, omega, hbar)

    phases = np.exp(-1j * energies[None, :] * times[:, None] / hbar)
    states = (phases * kept[None, :]) @ basis_sim

    return ReferenceSolution(
        states=np.asarray(
            [normalise_physical(state, grid.spacing) for state in states], dtype=np.complex128
        ),
        times=times,
        method="hermite_eigenbasis_expansion",
        diagnostics={
            "dense_grid_size": dense_grid_size,
            "basis_cap": basis_cap,
            "n_modes": n_keep,
            "raw_coefficient_norm": total_weight,
            "tail_weight": tail_weight,
            "box_truncation_note": "eigenstates are truncated to the finite periodic box",
        },
    )

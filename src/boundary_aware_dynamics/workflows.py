"""Benchmark workflows built from configuration.

Everything a notebook needs is assembled here from a :class:`RunConfig`, so a
notebook never redefines a physical parameter and two notebooks cannot disagree.

Three benchmarks, with distinct and clearly separated purposes:

A ``harmonic``
    Periodic box.  Validates Strang splitting, the FFT/QFT convention and
    Trotter convergence where the reference is an analytical eigenbasis.

B ``infinite_well``
    Hard walls, zero interior potential.  A *control*: the Dirichlet propagator
    is exact here, so this benchmark measures boundary topology, not accuracy of
    the time integrator.  Its headline result is the direct periodic-versus-
    Dirichlet comparison.

C ``tilted_well``
    Hard walls plus ``V(x) = F (x - L/2)``.  The genuine Trotter benchmark:
    ``[T, V] != 0``, so the splitting has real second-order error to measure
    under Dirichlet boundary conditions.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Benchmark, RunConfig
from .diagnostics import (
    StateErrors,
    boundary_diagnostics,
    energy_expectation,
    momentum_expectation,
    position_expectation,
    state_errors,
)
from .grids import Grid, dirichlet_midpoint_grid, periodic_grid
from .propagators import (
    Propagation,
    harmonic_potential,
    split_operator_evolution,
    tilted_potential,
)
from .references import (
    ReferenceSolution,
    exact_discrete_propagation,
    finite_difference_reference,
    harmonic_reference,
    sine_galerkin_reference,
)
from .states import gaussian_wavepacket, sine_windowed_gaussian

__all__ = [
    "BenchmarkResult",
    "BoundaryComparison",
    "ConvergenceStudy",
    "build_grid",
    "build_initial_state",
    "build_potential",
    "build_state_callable",
    "fit_convergence_slope",
    "grid_convergence_study",
    "run_benchmark",
    "run_boundary_comparison",
    "trotter_convergence_study",
]


# --------------------------------------------------------- construction ----


def build_grid(benchmark: Benchmark, n_grid: int | None = None) -> Grid:
    """Return the grid implied by the benchmark's boundary model."""
    size = benchmark.domain.n_grid if n_grid is None else n_grid
    if benchmark.domain.boundary == "periodic":
        return periodic_grid(benchmark.domain.x_left, benchmark.domain.x_right, size)
    return dirichlet_midpoint_grid(benchmark.domain.length, size)


def build_state_callable(benchmark: Benchmark) -> Callable[[np.ndarray], np.ndarray]:
    """Return the initial state as a function of position, for dense references."""
    spec = benchmark.initial_state
    length = benchmark.domain.length

    def state(positions: np.ndarray) -> np.ndarray:
        envelope = np.exp(-((positions - spec.centre) ** 2) / (4.0 * spec.sigma**2))
        packet = envelope * np.exp(1j * spec.momentum * (positions - spec.centre))
        if spec.sine_window:
            packet = packet * np.sin(np.pi * positions / length)
        return packet

    return state


def build_initial_state(benchmark: Benchmark, grid: Grid) -> np.ndarray:
    """Return the normalised initial state sampled on ``grid``."""
    spec = benchmark.initial_state
    if spec.sine_window:
        return sine_windowed_gaussian(
            grid.positions, grid.spacing, benchmark.domain.length,
            spec.centre, spec.momentum, spec.sigma,
        )
    return gaussian_wavepacket(
        grid.positions, grid.spacing, spec.centre, spec.momentum, spec.sigma
    )


def build_potential(benchmark: Benchmark, grid: Grid, mass: float) -> np.ndarray:
    """Return the potential on ``grid``; zero for the free well."""
    if benchmark.omega is not None:
        return harmonic_potential(grid.positions, mass, benchmark.omega)
    if benchmark.tilt_force:
        return tilted_potential(grid.positions, benchmark.domain.length, benchmark.tilt_force)
    return np.zeros(grid.n_grid)


def build_reference(
    benchmark: Benchmark,
    grid: Grid,
    times: np.ndarray,
    config: RunConfig,
) -> ReferenceSolution:
    """Return the continuum reference appropriate to the benchmark."""
    physics = config.physics
    settings = benchmark.reference

    if benchmark.omega is not None:
        spec = benchmark.initial_state
        return harmonic_reference(
            grid, times, spec.centre, spec.momentum, spec.sigma,
            physics.mass, benchmark.omega, physics.hbar,
            settings.dense_grid_size, settings.basis_cap, settings.tail_tolerance,
        )
    return sine_galerkin_reference(
        build_state_callable(benchmark), grid, times,
        physics.mass, physics.hbar, settings.basis_cap,
        tilt_force=benchmark.tilt_force or 0.0,
        dense_grid_size=settings.dense_grid_size,
    )


# ------------------------------------------------------------- results -----


@dataclass
class BenchmarkResult:
    """One benchmark run with its reference, diagnostics and provenance."""

    name: str
    grid: Grid
    propagation: Propagation
    reference: ReferenceSolution
    errors: list[StateErrors]
    observables: dict[str, np.ndarray]
    boundary: list[dict[str, float]]
    config_hash: str
    runtime_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def times(self) -> np.ndarray:
        return self.propagation.times

    @property
    def final_errors(self) -> StateErrors:
        return self.errors[-1]

    @property
    def infidelity(self) -> np.ndarray:
        return np.array([error.infidelity for error in self.errors])

    @property
    def l2_state_error(self) -> np.ndarray:
        return np.array([error.l2_state_error for error in self.errors])


@dataclass
class ConvergenceStudy:
    """A swept-parameter study with its fitted slope and fit window."""

    name: str
    parameter: str
    values: np.ndarray
    step_sizes: np.ndarray
    l2_state_error: np.ndarray
    infidelity: np.ndarray
    fit: dict[str, Any]
    reference_kind: str


@dataclass
class BoundaryComparison:
    """Periodic versus Dirichlet propagation of the same hard-wall problem."""

    grid: Grid
    times: np.ndarray
    reference_states: np.ndarray
    dirichlet_states: np.ndarray
    periodic_states: np.ndarray
    dirichlet_errors: list[StateErrors]
    periodic_errors: list[StateErrors]
    dirichlet_boundary: list[dict[str, float]]
    periodic_boundary: list[dict[str, float]]
    reference_method: str
    config_hash: str

    @property
    def cross_fidelity(self) -> np.ndarray:
        """Fidelity between the two propagations, as a direct topology measure."""
        from .diagnostics import fidelity

        return np.array(
            [
                fidelity(a, b, self.grid.spacing)
                for a, b in zip(self.dirichlet_states, self.periodic_states)
            ]
        )


# ------------------------------------------------------------- runners -----


def run_benchmark(
    config: RunConfig,
    name: str,
    n_steps: int | None = None,
    n_grid: int | None = None,
) -> BenchmarkResult:
    """Run one benchmark end to end and collect every diagnostic."""
    started = time.perf_counter()
    benchmark = config.benchmark(name)
    physics = config.physics

    grid = build_grid(benchmark, n_grid)
    steps = benchmark.time_grid.n_steps if n_steps is None else n_steps
    psi0 = build_initial_state(benchmark, grid)
    potential = build_potential(benchmark, grid, physics.mass)

    propagation = split_operator_evolution(
        psi0, grid, potential, benchmark.time_grid.t_max, steps, physics.mass, physics.hbar
    )
    reference = build_reference(benchmark, grid, propagation.times, config)

    errors = [
        state_errors(ref, state, grid.spacing)
        for ref, state in zip(reference.states, propagation.states)
    ]
    boundary = [
        boundary_diagnostics(state, grid, physics.hbar, physics.mass)
        for state in propagation.states
    ]

    positions = np.array([position_expectation(s, grid) for s in propagation.states])
    energies = np.array(
        [energy_expectation(s, grid, potential, physics.mass, physics.hbar) for s in propagation.states]
    )
    observables = {
        "position_mean": positions[:, 0],
        "position_mean_square": positions[:, 1],
        "position_variance": positions[:, 2],
        "momentum_mean": np.array(
            [momentum_expectation(s, grid, physics.hbar) for s in propagation.states]
        ),
        "kinetic_energy": energies[:, 0],
        "potential_energy": energies[:, 1],
        "total_energy": energies[:, 2],
        "energy_drift": energies[:, 2] - energies[0, 2],
    }

    return BenchmarkResult(
        name=name,
        grid=grid,
        propagation=propagation,
        reference=reference,
        errors=errors,
        observables=observables,
        boundary=boundary,
        config_hash=config.config_hash,
        runtime_seconds=time.perf_counter() - started,
        metadata={
            "profile": config.profile,
            "n_grid": grid.n_grid,
            "n_data_qubits": grid.n_data_qubits,
            "n_steps": steps,
            "time_step": propagation.time_step,
            "boundary": grid.boundary,
            "transform": propagation.metadata["transform"],
            "reference_method": reference.method,
            "reference_diagnostics": reference.diagnostics,
        },
    )


def fit_convergence_slope(
    step_sizes: np.ndarray,
    errors: np.ndarray,
    fit_from: int = 0,
) -> dict[str, Any]:
    """Fit ``log error = slope * log dt + c`` over a stated window.

    ``fit_from`` drops the leading points, which is how the pre-asymptotic
    transient is excluded.  The window and the goodness of fit are returned so
    that a reported slope can never be detached from the interval it was fitted
    on.
    """
    x = np.log(np.asarray(step_sizes, dtype=float)[fit_from:])
    y = np.log(np.asarray(errors, dtype=float)[fit_from:])
    if x.size < 2:
        raise ValueError("At least two points are needed to fit a slope.")

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": 1.0 - residual / total if total > 0 else float("nan"),
        "fit_from_index": fit_from,
        "fit_interval_dt": (float(np.exp(x.min())), float(np.exp(x.max()))),
        "n_points": int(x.size),
    }


def trotter_convergence_study(
    config: RunConfig,
    name: str,
    fit_from: int = 1,
) -> ConvergenceStudy:
    """Sweep the Trotter step count against the exact discrete propagator.

    The reference is exact diagonalisation of the *same* discrete Hamiltonian,
    so spatial discretisation cancels and the fitted slope measures the order of
    the time integrator alone.
    """
    benchmark = config.benchmark(name)
    physics = config.physics
    grid = build_grid(benchmark)
    psi0 = build_initial_state(benchmark, grid)
    potential = build_potential(benchmark, grid, physics.mass)
    t_max = benchmark.time_grid.t_max

    reference = exact_discrete_propagation(
        psi0, grid, np.array([0.0, t_max]), physics.mass, physics.hbar, potential
    ).final_state

    step_counts = np.array(benchmark.time_grid.step_sweep, dtype=int)
    step_sizes = t_max / step_counts
    errors, infidelities = [], []
    for steps in step_counts:
        run = split_operator_evolution(
            psi0, grid, potential, t_max, int(steps), physics.mass, physics.hbar
        )
        measured = state_errors(reference, run.final_state, grid.spacing)
        errors.append(measured.l2_state_error)
        infidelities.append(measured.infidelity)

    errors = np.array(errors)
    floor = 1e-13
    fit = (
        fit_convergence_slope(step_sizes, errors, fit_from)
        if np.all(errors[fit_from:] > floor)
        else {"slope": float("nan"), "note": "all sampled errors are at the round-off floor"}
    )
    return ConvergenceStudy(
        name=name,
        parameter="n_steps",
        values=step_counts,
        step_sizes=step_sizes,
        l2_state_error=errors,
        infidelity=np.array(infidelities),
        fit=fit,
        reference_kind="exact_discrete_hamiltonian",
    )


def grid_convergence_study(config: RunConfig, name: str) -> ConvergenceStudy:
    """Sweep the grid size against the continuum reference at fixed step count."""
    benchmark = config.benchmark(name)
    sizes = np.array(benchmark.grid_sweep, dtype=int)

    errors, infidelities, spacings = [], [], []
    for size in sizes:
        result = run_benchmark(config, name, n_grid=int(size))
        errors.append(result.final_errors.l2_state_error)
        infidelities.append(result.final_errors.infidelity)
        spacings.append(result.grid.spacing)

    return ConvergenceStudy(
        name=name,
        parameter="n_grid",
        values=sizes,
        step_sizes=np.array(spacings),
        l2_state_error=np.array(errors),
        infidelity=np.array(infidelities),
        fit={"note": "spatial error saturates against the reference; no slope fitted"},
        reference_kind="continuum_reference",
    )


def run_boundary_comparison(config: RunConfig, name: str = "infinite_well") -> BoundaryComparison:
    """Propagate the same hard-wall state under both boundary topologies.

    The two runs share the initial state, the box, the grid resolution, the time
    interval and the step count.  They differ only in which spectral transform
    sits inside the split step, so any divergence is attributable to boundary
    topology alone.  Both are compared against a finite-difference hard-wall
    reference, which shares neither basis nor method with either propagator.
    """
    benchmark = config.benchmark(name)
    physics = config.physics
    if benchmark.domain.boundary != "dirichlet":
        raise ValueError(f"Benchmark {name!r} is not a hard-wall problem.")

    length = benchmark.domain.length
    n_grid = benchmark.domain.n_grid
    steps = benchmark.time_grid.n_steps
    t_max = benchmark.time_grid.t_max

    dirichlet_grid = dirichlet_midpoint_grid(length, n_grid)
    # The ring uses the identical sample points, so the two states are directly
    # comparable point by point without any interpolation.
    ring_grid = Grid(
        dirichlet_grid.positions, dirichlet_grid.spacing, "periodic", (0.0, length)
    )

    psi0 = build_initial_state(benchmark, dirichlet_grid)
    potential = build_potential(benchmark, dirichlet_grid, physics.mass)

    by_dirichlet = split_operator_evolution(
        psi0, dirichlet_grid, potential, t_max, steps, physics.mass, physics.hbar
    )
    by_periodic = split_operator_evolution(
        psi0, ring_grid, potential, t_max, steps, physics.mass, physics.hbar
    )

    potential_fn = None
    if benchmark.tilt_force:
        tilt = benchmark.tilt_force
        potential_fn = lambda x: tilt * (x - 0.5 * length)  # noqa: E731

    reference = finite_difference_reference(
        build_state_callable(benchmark), dirichlet_grid, by_dirichlet.times,
        physics.mass, physics.hbar, potential_fn, refinement=8,
    )

    spacing = dirichlet_grid.spacing
    return BoundaryComparison(
        grid=dirichlet_grid,
        times=by_dirichlet.times,
        reference_states=reference.states,
        dirichlet_states=by_dirichlet.states,
        periodic_states=by_periodic.states,
        dirichlet_errors=[
            state_errors(r, s, spacing) for r, s in zip(reference.states, by_dirichlet.states)
        ],
        periodic_errors=[
            state_errors(r, s, spacing) for r, s in zip(reference.states, by_periodic.states)
        ],
        dirichlet_boundary=[
            boundary_diagnostics(s, dirichlet_grid, physics.hbar, physics.mass)
            for s in by_dirichlet.states
        ],
        periodic_boundary=[
            boundary_diagnostics(s, ring_grid, physics.hbar, physics.mass)
            for s in by_periodic.states
        ],
        reference_method=reference.method,
        config_hash=config.config_hash,
    )

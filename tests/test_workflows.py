"""Config-driven benchmark assembly and the boundary-comparison experiment."""

from __future__ import annotations

import numpy as np
import pytest

from boundary_aware_dynamics.config import load_config
from boundary_aware_dynamics.workflows import (
    build_grid,
    build_initial_state,
    build_potential,
    build_state_callable,
    run_benchmark,
    run_boundary_comparison,
)

CONFIG = load_config("configs/smoke.yaml")


# -------------------------------------------------------- construction -----


def test_grids_follow_the_configured_boundary_model():
    assert build_grid(CONFIG.benchmark("harmonic")).boundary == "periodic"
    assert build_grid(CONFIG.benchmark("infinite_well")).boundary == "dirichlet"
    assert build_grid(CONFIG.benchmark("tilted_well")).boundary == "dirichlet"


def test_only_the_tilted_well_gets_a_nonzero_interior_potential():
    for name, expect_zero in (("infinite_well", True), ("tilted_well", False)):
        benchmark = CONFIG.benchmark(name)
        potential = build_potential(benchmark, build_grid(benchmark), CONFIG.physics.mass)
        assert np.allclose(potential, 0.0) is expect_zero


def test_tilt_potential_is_antisymmetric_about_the_box_centre():
    benchmark = CONFIG.benchmark("tilted_well")
    grid = build_grid(benchmark)
    potential = build_potential(benchmark, grid, CONFIG.physics.mass)
    assert np.allclose(potential, -potential[::-1], atol=1e-12)
    assert potential.sum() == pytest.approx(0.0, abs=1e-10)


def test_state_callable_agrees_with_the_sampled_state():
    benchmark = CONFIG.benchmark("tilted_well")
    grid = build_grid(benchmark)
    sampled = build_initial_state(benchmark, grid)
    from_callable = build_state_callable(benchmark)(grid.positions)
    from_callable /= np.sqrt(grid.spacing * np.vdot(from_callable, from_callable).real)
    assert np.allclose(sampled, from_callable, atol=1e-12)


# ------------------------------------------------------------- results -----


@pytest.mark.parametrize("name", ["harmonic", "infinite_well", "tilted_well"])
def test_benchmarks_run_and_report_provenance(name):
    result = run_benchmark(CONFIG, name)
    assert result.config_hash == CONFIG.config_hash
    assert result.metadata["profile"] == "smoke"
    assert result.propagation.max_norm_error < 1e-12
    assert len(result.errors) == len(result.times)
    assert result.metadata["n_data_qubits"] == int(np.log2(result.grid.n_grid))


@pytest.mark.parametrize("name", ["harmonic", "infinite_well", "tilted_well"])
def test_energy_error_stays_bounded(name):
    # Strang splitting does not conserve energy exactly: the error is an O(dt^2)
    # excursion that stays bounded rather than accumulating. Asserting exact
    # conservation would be false; the order is checked by the test below.
    result = run_benchmark(CONFIG, name)
    drift = np.abs(result.observables["energy_drift"])
    scale = abs(result.observables["total_energy"][0]) + 1.0
    assert np.max(drift) / scale < 5e-2


def test_finer_steps_shrink_the_energy_error_quadratically():
    coarse = run_benchmark(CONFIG, "harmonic", n_steps=20)
    fine = run_benchmark(CONFIG, "harmonic", n_steps=80)
    coarse_peak = np.max(np.abs(coarse.observables["energy_drift"]))
    fine_peak = np.max(np.abs(fine.observables["energy_drift"]))
    assert fine_peak < coarse_peak / 8.0


def test_free_well_run_is_essentially_exact():
    result = run_benchmark(CONFIG, "infinite_well")
    assert result.final_errors.infidelity < 1e-10
    assert result.metadata["transform"] == "DST-II"


def test_harmonic_run_reports_hermite_reference_truncation():
    result = run_benchmark(CONFIG, "harmonic")
    diagnostics = result.metadata["reference_diagnostics"]
    assert diagnostics["tail_weight"] < 1e-8
    assert 0 < diagnostics["n_modes"] < diagnostics["basis_cap"]


# ------------------------------------------------- boundary comparison -----


def test_boundary_comparison_shares_everything_but_the_transform():
    comparison = run_boundary_comparison(CONFIG)
    # Same grid points, same times, same starting state.
    assert comparison.dirichlet_states.shape == comparison.periodic_states.shape
    assert np.allclose(comparison.dirichlet_states[0], comparison.periodic_states[0])
    assert comparison.cross_fidelity[0] == pytest.approx(1.0, abs=1e-12)


def test_dirichlet_tracks_the_independent_reference_and_periodic_does_not():
    # The central quantitative claim of the project. The reference is a
    # finite-difference hard-wall solution sharing neither basis nor method
    # with either propagator, so this is not circular.
    comparison = run_boundary_comparison(CONFIG)
    assert comparison.reference_method == "finite_difference_exact_diagonalisation"

    dirichlet_final = comparison.dirichlet_errors[-1].infidelity
    periodic_final = comparison.periodic_errors[-1].infidelity
    assert dirichlet_final < 1e-3
    assert periodic_final > 0.5
    assert periodic_final / dirichlet_final > 1e3


def test_the_two_topologies_diverge_once_the_packet_reaches_the_wall():
    comparison = run_boundary_comparison(CONFIG)
    cross = comparison.cross_fidelity
    assert cross[0] == pytest.approx(1.0, abs=1e-12)
    assert cross.min() < 0.2       # they end up nearly orthogonal


def test_periodic_propagation_violates_the_wall_condition():
    comparison = run_boundary_comparison(CONFIG)
    late = slice(len(comparison.times) // 2, None)
    dirichlet_residual = max(d["wall_residual"] for d in comparison.dirichlet_boundary[late])
    periodic_residual = max(d["wall_residual"] for d in comparison.periodic_boundary[late])
    assert periodic_residual > 3.0 * dirichlet_residual


def test_boundary_comparison_rejects_a_periodic_benchmark():
    with pytest.raises(ValueError, match="not a hard-wall problem"):
        run_boundary_comparison(CONFIG, "harmonic")

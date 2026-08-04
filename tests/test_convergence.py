"""Convergence orders, fitted over a stated window.

Ranges are deliberately loose enough not to be fragile, but tight enough that a
first-order method or a broken reference would fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from boundary_aware_dynamics.config import load_config
from boundary_aware_dynamics.workflows import (
    fit_convergence_slope,
    grid_convergence_study,
    trotter_convergence_study,
)

CONFIG = load_config("configs/paper.yaml")


# ------------------------------------------------------------ slope fit -----


def test_slope_fit_recovers_a_known_power_law():
    step_sizes = np.array([0.1, 0.05, 0.025, 0.0125])
    fit = fit_convergence_slope(step_sizes, 3.0 * step_sizes**2)
    assert fit["slope"] == pytest.approx(2.0, abs=1e-10)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert fit["n_points"] == 4


def test_slope_fit_reports_the_window_it_used():
    step_sizes = np.array([0.4, 0.1, 0.05, 0.025])
    fit = fit_convergence_slope(step_sizes, step_sizes**2, fit_from=1)
    assert fit["fit_from_index"] == 1
    assert fit["n_points"] == 3
    assert fit["fit_interval_dt"] == pytest.approx((0.025, 0.1))


def test_slope_fit_needs_at_least_two_points():
    with pytest.raises(ValueError, match="At least two points"):
        fit_convergence_slope(np.array([0.1]), np.array([0.01]))


# ---------------------------------------------------------- Benchmark A -----


def test_harmonic_state_error_is_second_order():
    study = trotter_convergence_study(CONFIG, "harmonic")
    assert study.fit["slope"] == pytest.approx(2.0, abs=0.15)
    assert study.fit["r_squared"] > 0.999


def test_harmonic_infidelity_is_fourth_order():
    study = trotter_convergence_study(CONFIG, "harmonic")
    fit = fit_convergence_slope(study.step_sizes, study.infidelity, fit_from=1)
    assert fit["slope"] == pytest.approx(4.0, abs=0.3)
    assert fit["r_squared"] > 0.999


def test_harmonic_error_decreases_monotonically_with_step_count():
    study = trotter_convergence_study(CONFIG, "harmonic")
    assert np.all(np.diff(study.l2_state_error) < 0.0)


# ---------------------------------------------------------- Benchmark B -----


def test_free_well_has_no_trotter_error_to_fit():
    # The honest outcome for a control benchmark: every sampled error sits at
    # the round-off floor, so no slope is reported rather than a meaningless
    # number being fitted to numerical noise.
    study = trotter_convergence_study(CONFIG, "infinite_well")
    assert np.all(study.l2_state_error < 1e-12)
    assert np.isnan(study.fit["slope"])
    assert "round-off floor" in study.fit["note"]


def test_free_well_error_is_flat_across_the_whole_sweep():
    study = trotter_convergence_study(CONFIG, "infinite_well")
    spread = study.l2_state_error.max() / study.l2_state_error.min()
    assert spread < 10.0     # noise only, not a convergence trend


# ---------------------------------------------------------- Benchmark C -----


def test_tilted_well_state_error_is_second_order_under_dirichlet_walls():
    study = trotter_convergence_study(CONFIG, "tilted_well")
    assert study.fit["slope"] == pytest.approx(2.0, abs=0.1)
    assert study.fit["r_squared"] > 0.999
    # The fit must exclude the pre-asymptotic first point.
    assert study.fit["fit_from_index"] == 1


def test_tilted_well_infidelity_is_fourth_order():
    study = trotter_convergence_study(CONFIG, "tilted_well")
    fit = fit_convergence_slope(study.step_sizes, study.infidelity, fit_from=1)
    assert fit["slope"] == pytest.approx(4.0, abs=0.3)


def test_tilted_well_spans_a_useful_error_range():
    # A convergence plot is only informative if the error moves over decades.
    study = trotter_convergence_study(CONFIG, "tilted_well")
    decades = np.log10(study.l2_state_error.max() / study.l2_state_error.min())
    assert decades > 3.0


def test_tilted_well_uses_the_discrete_hamiltonian_reference():
    study = trotter_convergence_study(CONFIG, "tilted_well")
    assert study.reference_kind == "exact_discrete_hamiltonian"


# --------------------------------------------------------------- spatial ----


def test_harmonic_spatial_error_falls_before_saturating():
    study = grid_convergence_study(CONFIG, "harmonic")
    errors = study.l2_state_error
    # Coarse grids must improve; the finest grids saturate against the
    # reference and the fixed Trotter step, so only the early trend is asserted.
    assert errors[0] > errors[1] > errors[2]
    assert study.values[0] < study.values[-1]


def test_free_well_spatial_error_stays_at_the_floor():
    # Increasing N cannot fix a wrong boundary topology and does not need to
    # fix anything here: the Dirichlet propagator is already exact.
    study = grid_convergence_study(CONFIG, "infinite_well")
    assert np.all(study.infidelity < 1e-6)

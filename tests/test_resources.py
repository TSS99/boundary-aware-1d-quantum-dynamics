"""Resource accounting: composition, ancillas, synthesis model and connectivity."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from boundary_aware_dynamics.circuits.resources import (
    ResourceRow,
    approximate_qft_study,
    count_resources,
    forward_transform,
    measure_qft_approximation_error,
    propagation_resources,
    scaling_table,
)
from boundary_aware_dynamics.circuits.state_preparation import (
    exact_state_preparation_circuit,
    measurement_note,
    shot_budget_for_density,
)
from boundary_aware_dynamics.config import load_config
from boundary_aware_dynamics.grids import dirichlet_midpoint_grid
from boundary_aware_dynamics.states import sine_windowed_gaussian
from boundary_aware_dynamics.transforms import dft_matrix

CONFIG = load_config("configs/paper.yaml")


# ------------------------------------------------------------ convention ---


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
def test_forward_transform_is_the_numpy_dft(n_qubits):
    observed = Operator(forward_transform(n_qubits)).data
    assert np.allclose(observed, dft_matrix(2**n_qubits), atol=1e-12)


# ------------------------------------------------------------- ancillas ----


def test_dirichlet_rows_report_two_ancillas_and_periodic_rows_report_none():
    # The earlier tables listed n_qubits = 6 for a circuit that occupied 8 wires.
    periodic = propagation_resources(CONFIG, "harmonic", n_steps=1)
    dirichlet = propagation_resources(CONFIG, "tilted_well", n_steps=1)

    assert periodic.n_ancilla_qubits == 0
    assert periodic.n_total_qubits == periodic.n_data_qubits == 6

    assert dirichlet.n_ancilla_qubits == 2
    assert dirichlet.n_data_qubits == 6
    assert dirichlet.n_total_qubits == 8


def test_every_row_states_its_scope_and_assumptions():
    row = propagation_resources(CONFIG, "tilted_well", n_steps=4)
    assert row.scope == "propagation_core_only"
    assert row.includes_state_preparation is False
    assert row.includes_measurement is False
    assert row.connectivity == "all_to_all"
    assert row.optimisation_level == CONFIG.circuits.optimisation_level
    assert row.seed == CONFIG.circuits.seed
    assert set(ResourceRow.__dataclass_fields__) <= set(row.as_dict())


# ---------------------------------------------------------- composition ----


def test_potential_phases_are_merged_across_steps():
    # r steps need 1 half + (r-1) full + 1 half potential phases, not 2r halves.
    for n_steps in (1, 2, 4, 8):
        row = propagation_resources(CONFIG, "tilted_well", n_steps=n_steps)
        assert f"1 half + {max(n_steps - 1, 0)} full + 1 half" in row.notes


def test_merging_half_phases_is_exact_and_cheaper_than_naive_repetition():
    # The naive circuit repeats the full five-block step r times, leaving two
    # adjacent half-potential phases at every interior boundary. Merging them is
    # an exact identity, so the two circuits must implement the same unitary
    # while the merged one uses strictly fewer gates.
    from boundary_aware_dynamics.circuits.phases import (
        harmonic_position_expansion,
        signed_momentum_expansion,
    )
    from boundary_aware_dynamics.circuits.resources import build_propagation_core

    n_grid, n_steps = 8, 3
    spacing, time_step = 16.0 / n_grid, 0.05
    kinetic = signed_momentum_expansion(n_grid, spacing, 1.0, 1.0, time_step)
    full = harmonic_position_expansion(n_grid, -8.0, spacing, 1.0, 1.0, 1.0, time_step)
    half = harmonic_position_expansion(n_grid, -8.0, spacing, 1.0, 1.0, 1.0, 0.5 * time_step)

    merged = build_propagation_core("periodic", n_grid, n_steps, full, kinetic, half)
    single = build_propagation_core("periodic", n_grid, 1, full, kinetic, half)
    naive = single.copy()
    for _ in range(n_steps - 1):
        naive.compose(single, inplace=True)

    assert np.allclose(Operator(merged).data, Operator(naive).data, atol=1e-10)
    assert merged.size() < naive.size()


def test_kinetic_blocks_scale_linearly_with_step_count():
    rows = [propagation_resources(CONFIG, "tilted_well", n_steps=r) for r in (2, 4, 8)]
    counts = [row.two_qubit_gates for row in rows]
    assert counts[1] == pytest.approx(2 * counts[0], rel=0.05)
    assert counts[2] == pytest.approx(2 * counts[1], rel=0.05)


def test_free_well_needs_no_potential_phases():
    row = propagation_resources(CONFIG, "infinite_well", n_steps=4)
    assert "zero interior potential" in row.notes


# ------------------------------------------------------------- synthesis ---


@pytest.mark.parametrize("name", ["harmonic", "tilted_well"])
def test_structured_synthesis_beats_the_generic_diagonal(name):
    # The generic DiagonalGate is kept only as an explicitly labelled upper
    # bound; it must not be the number that gets reported as the cost.
    structured = propagation_resources(CONFIG, name, n_steps=1, synthesis="structured")
    generic = propagation_resources(CONFIG, name, n_steps=1, synthesis="generic_diagonal")

    assert structured.two_qubit_gates < generic.two_qubit_gates
    assert structured.total_depth < generic.total_depth
    assert structured.synthesis_model == "structured"
    assert generic.synthesis_model == "generic_diagonal"


# ---------------------------------------------------------- connectivity ---


def test_linear_connectivity_costs_more_than_all_to_all():
    all_to_all = propagation_resources(CONFIG, "tilted_well", n_steps=1, connectivity="all_to_all")
    linear = propagation_resources(CONFIG, "tilted_well", n_steps=1, connectivity="linear")
    assert linear.two_qubit_gates > all_to_all.two_qubit_gates
    assert linear.connectivity == "linear"


def test_unknown_connectivity_is_rejected():
    from qiskit import QuantumCircuit

    with pytest.raises(ValueError, match="Unknown connectivity"):
        count_resources(QuantumCircuit(2), CONFIG, "heavy_hex_ish")


# --------------------------------------------------------------- scaling ---


def test_scaling_is_reported_across_register_sizes_not_one_qubit_count():
    rows = scaling_table(CONFIG, "tilted_well", (8, 16, 32, 64, 128), n_steps=1)
    assert [row.n_data_qubits for row in rows] == [3, 4, 5, 6, 7]
    counts = [row.two_qubit_gates for row in rows]
    assert counts == sorted(counts)


def test_two_qubit_cost_grows_polynomially_not_exponentially():
    rows = scaling_table(CONFIG, "tilted_well", (16, 32, 64, 128, 256), n_steps=1)
    counts = np.array([row.two_qubit_gates for row in rows], dtype=float)
    qubits = np.array([row.n_total_qubits for row in rows], dtype=float)
    slope = np.polyfit(np.log(qubits), np.log(counts), 1)[0]
    # Quadratic-ish growth; exponential would give a slope far above this.
    assert 1.0 < slope < 3.0


# ------------------------------------------------------ approximate QFT ----


def test_exact_qft_has_zero_approximation_error():
    assert measure_qft_approximation_error(5, 0) == pytest.approx(0.0, abs=1e-12)


def test_truncating_more_rotations_increases_the_operator_error():
    errors = [measure_qft_approximation_error(6, degree) for degree in (1, 2, 3)]
    assert errors[0] < errors[1] < errors[2]


def test_approximate_qft_trade_off_reports_both_saving_and_error():
    rows = approximate_qft_study(CONFIG, "harmonic", degrees=(0, 1, 2, 3), n_steps=1)
    assert rows[0]["transform_operator_error"] == pytest.approx(0.0, abs=1e-12)
    assert rows[0]["two_qubit_reduction"] == pytest.approx(0.0, abs=1e-12)
    # Gates are saved, but the error grows: both must be present in every row.
    assert rows[-1]["two_qubit_gates"] < rows[0]["two_qubit_gates"]
    assert rows[-1]["transform_operator_error"] > rows[0]["transform_operator_error"]
    for row in rows:
        assert {"two_qubit_reduction", "transform_operator_error"} <= set(row)


# ------------------------------------------- state preparation and shots ---


def test_state_preparation_is_costed_separately_and_is_not_free():
    grid = dirichlet_midpoint_grid(10.0, 64)
    psi = sine_windowed_gaussian(grid.positions, grid.spacing, 10.0, 5.0, 2.0, 0.8)
    circuit = exact_state_preparation_circuit(psi, grid.spacing)
    counts = count_resources(circuit, CONFIG)

    assert circuit.num_qubits == 6
    # Exact amplitude encoding is exponential in the qubit count, so it must not
    # be folded silently into the propagation core.
    assert counts["two_qubit_gates"] > 10
    assert propagation_resources(CONFIG, "infinite_well", n_steps=1).includes_state_preparation is False


def test_state_preparation_produces_the_right_amplitudes():
    from qiskit.quantum_info import Statevector

    grid = dirichlet_midpoint_grid(10.0, 16)
    psi = sine_windowed_gaussian(grid.positions, grid.spacing, 10.0, 5.0, 2.0, 0.8)
    prepared = Statevector(exact_state_preparation_circuit(psi, grid.spacing)).data

    expected = np.sqrt(grid.spacing) * psi
    overlap = abs(np.vdot(expected, prepared))
    assert overlap == pytest.approx(1.0, abs=1e-10)


def test_state_preparation_rejects_non_power_of_two_lengths():
    with pytest.raises(ValueError, match="not a power of two"):
        exact_state_preparation_circuit(np.ones(6), 1.0)


def test_measurement_caveats_are_explicit_about_hardware():
    note = measurement_note()
    assert note["hardware_runs_performed"] is False
    assert "simulator diagnostic" in note["statevector_fidelity"]


def test_shot_budget_grows_as_inverse_square_precision():
    coarse = shot_budget_for_density(0.01)
    fine = shot_budget_for_density(0.005)
    assert fine == pytest.approx(4 * coarse, rel=0.01)
    with pytest.raises(ValueError, match="precision must lie"):
        shot_budget_for_density(0.0)

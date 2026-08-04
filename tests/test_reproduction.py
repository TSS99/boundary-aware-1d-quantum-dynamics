"""End-to-end reproduction: does the pipeline produce valid, deterministic output?

These tests run the smoke profile into a temporary directory, so they never touch
``results/`` and never depend on it.  They are slower than the rest of the suite
but they are the only checks that exercise the whole path from configuration to
exported artefact.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TABLES = (
    "benchmark_parameters.csv", "benchmark_errors.csv", "benchmark_observables.csv",
    "boundary_comparison.csv", "convergence.csv", "resource_single_step.csv",
    "resource_scaling.csv", "resource_vs_steps.csv", "approximate_qft.csv",
)


@pytest.fixture(scope="module")
def reproduced(tmp_path_factory) -> Path:
    """Run the smoke profile once into a temporary output directory."""
    output = tmp_path_factory.mktemp("reproduction")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reproduce.py"),
         "--profile", "smoke", "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return output


def test_all_required_tables_are_written(reproduced):
    for name in REQUIRED_TABLES:
        path = reproduced / "tables" / name
        assert path.exists(), f"missing {name}"
        assert not pd.read_csv(path).empty, f"{name} is empty"


def test_all_required_metadata_is_written(reproduced):
    for name in ("provenance.json", "key_results.json", "config.json",
                 "paper_values.tex", "figure_manifest.csv", "figure_manifest.json"):
        assert (reproduced / "metadata" / name).exists(), f"missing {name}"


def test_every_manifest_figure_exists_and_has_a_caption(reproduced):
    manifest = pd.read_csv(reproduced / "metadata" / "figure_manifest.csv")
    assert len(manifest) >= 8
    for _, row in manifest.iterrows():
        matches = list((reproduced / "figures").glob(f"{row['filename']}.*"))
        assert matches, f"no file for figure {row['figure_id']}"
        assert len(str(row["caption"])) > 20, f"figure {row['figure_id']} has no real caption"
        assert row["config_hash"], f"figure {row['figure_id']} has no config hash"


def test_resource_tables_record_their_assumptions(reproduced):
    frame = pd.read_csv(reproduced / "tables" / "resource_single_step.csv")
    required = {
        "n_data_qubits", "n_ancilla_qubits", "n_total_qubits", "synthesis_model",
        "connectivity", "basis_gates", "optimisation_level", "seed",
        "approximation_degree", "includes_state_preparation",
        "includes_measurement", "scope",
    }
    assert required <= set(frame.columns)
    # A Dirichlet row must carry its two ancillas.
    dirichlet = frame[frame.boundary == "dirichlet"]
    assert (dirichlet.n_ancilla_qubits == 2).all()
    assert (dirichlet.n_total_qubits == dirichlet.n_data_qubits + 2).all()


def test_no_fidelity_value_exceeds_one(reproduced):
    frame = pd.read_csv(reproduced / "tables" / "boundary_comparison.csv")
    assert frame["cross_fidelity"].max() <= 1.0 + 1e-9


def test_boundary_comparison_supports_the_central_claim(reproduced):
    frame = pd.read_csv(reproduced / "tables" / "boundary_comparison.csv")
    assert frame["dirichlet_infidelity"].iloc[-1] < 1e-3
    assert frame["periodic_infidelity"].iloc[-1] > 0.5
    assert frame["cross_fidelity"].min() < 0.3


def test_convergence_slopes_are_recorded_with_their_fit_window(reproduced):
    frame = pd.read_csv(reproduced / "tables" / "convergence.csv")
    trotter = frame[frame.study == "trotter"]
    for name in ("harmonic", "tilted_well"):
        rows = trotter[trotter.benchmark == name]
        assert abs(rows["fitted_slope"].iloc[0] - 2.0) < 0.15
        assert rows["fit_r_squared"].iloc[0] > 0.999
        assert rows["fit_from_index"].iloc[0] >= 1        # transient excluded
        assert rows["reference_kind"].iloc[0] == "exact_discrete_hamiltonian"


def test_free_well_reports_no_slope_rather_than_fitting_noise(reproduced):
    frame = pd.read_csv(reproduced / "tables" / "convergence.csv")
    rows = frame[(frame.study == "trotter") & (frame.benchmark == "infinite_well")]
    assert rows["fitted_slope"].isna().all()
    assert (rows["l2_state_error"] < 1e-12).all()


def test_paper_macros_are_generated_not_typed(reproduced):
    text = (reproduced / "metadata" / "paper_values.tex").read_text(encoding="utf-8")
    for macro in ("\\ConfigHash", "\\BoundaryPeriodicInfidelity",
                  "\\TiltedTrotterSlope", "\\TiltedTotalQubits"):
        assert macro in text
    assert "Do not edit by hand" in text


def test_provenance_is_complete(reproduced):
    payload = json.loads((reproduced / "metadata" / "provenance.json").read_text(encoding="utf-8"))
    for field in ("config_hash", "source_hash", "git_commit", "git_dirty",
                  "python_version", "platform_name", "dependencies", "seeds",
                  "output_hashes", "volatile"):
        assert field in payload, f"provenance missing {field}"
    assert payload["output_hashes"], "no output hashes recorded"
    assert "generated_at_utc" in payload["volatile"]
    # Volatile fields are segregated so identical work gives identical
    # deterministic fields.
    assert "generated_at_utc" not in payload


def test_rerunning_reproduces_identical_deterministic_output(reproduced, tmp_path):
    """A second run must give byte-identical outputs apart from volatile fields."""
    second = tmp_path / "again"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reproduce.py"),
         "--profile", "smoke", "--output", str(second)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]

    first_hashes = json.loads(
        (reproduced / "metadata" / "provenance.json").read_text(encoding="utf-8")
    )["output_hashes"]
    second_hashes = json.loads(
        (second / "metadata" / "provenance.json").read_text(encoding="utf-8")
    )["output_hashes"]

    assert set(first_hashes) == set(second_hashes)
    differing = {k for k in first_hashes if first_hashes[k] != second_hashes[k]}
    # Tables must be bit-identical; PDF output embeds a creation date, so only
    # the data artefacts are required to match exactly.
    assert not {k for k in differing if k.endswith(".csv")}, differing


def test_stale_results_are_detected_from_provenance(reproduced):
    from boundary_aware_dynamics.config import load_config
    from boundary_aware_dynamics.provenance import (
        check_staleness, collect_provenance, read_provenance,
    )

    config = load_config(ROOT / "configs" / "smoke.yaml")
    stored = read_provenance(reproduced / "metadata")
    assert not check_staleness(stored, collect_provenance(config))["stale"]

    # A configuration change must be reported, with a reason.
    other = load_config(ROOT / "configs" / "paper.yaml")
    status = check_staleness(stored, collect_provenance(other))
    assert status["stale"]
    assert any("configuration changed" in reason for reason in status["reasons"])


def test_missing_provenance_is_treated_as_stale():
    from boundary_aware_dynamics.provenance import check_staleness

    status = check_staleness(None, None)  # type: ignore[arg-type]
    assert status["stale"]
    assert "no stored provenance record" in status["reasons"]

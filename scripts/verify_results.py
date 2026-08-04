"""Local quality gate: tests, schemas, provenance and figure checks.

Usage::

    python scripts/verify_results.py                 # fast checks
    python scripts/verify_results.py --full          # adds notebook execution

Replaces file-existence checking with provenance-aware verification: a result is
stale when the configuration, the package source, the dependency versions or the
seeds have moved since it was produced, not merely when a file is missing.

This is a local script by design.  The project deliberately does not depend on a
hosted CI service, so verification can be run from any clone with no network.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boundary_aware_dynamics.config import load_config  # noqa: E402
from boundary_aware_dynamics.provenance import (  # noqa: E402
    check_staleness,
    collect_provenance,
    read_provenance,
    verify_output_hashes,
)

REQUIRED_TABLES = (
    "benchmark_parameters.csv",
    "benchmark_errors.csv",
    "benchmark_observables.csv",
    "boundary_comparison.csv",
    "convergence.csv",
    "resource_single_step.csv",
    "resource_scaling.csv",
    "resource_vs_steps.csv",
    "approximate_qft.csv",
)
REQUIRED_METADATA = (
    "provenance.json",
    "key_results.json",
    "config.json",
    "paper_values.tex",
    "figure_manifest.csv",
)
RESOURCE_REQUIRED_COLUMNS = {
    "n_data_qubits", "n_ancilla_qubits", "n_total_qubits", "synthesis_model",
    "connectivity", "optimisation_level", "seed", "approximation_degree",
    "includes_state_preparation", "includes_measurement", "scope",
}


class Report:
    """Accumulates pass/fail lines and decides the exit status."""

    def __init__(self) -> None:
        self.lines: list[tuple[bool, str]] = []

    def check(self, ok: bool, message: str) -> bool:
        self.lines.append((ok, message))
        print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
        return ok

    @property
    def failed(self) -> int:
        return sum(1 for ok, _ in self.lines if not ok)


def run_tests(report: Report) -> None:
    print("\n== unit and physics tests ==")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "no output"
    report.check(result.returncode == 0, f"pytest: {tail}")


def check_outputs(report: Report, profile: str) -> None:
    config = load_config(ROOT / "configs" / f"{profile}.yaml")
    output_root = ROOT / config.output_root
    print(f"\n== outputs for profile '{profile}' ==")

    if not report.check(output_root.exists(), f"{output_root.relative_to(ROOT)} exists"):
        print("      run: python scripts/reproduce.py --profile " + profile)
        return

    tables, metadata, figures = (output_root / n for n in ("tables", "metadata", "figures"))
    for name in REQUIRED_TABLES:
        report.check((tables / name).exists(), f"table {name}")
    for name in REQUIRED_METADATA:
        report.check((metadata / name).exists(), f"metadata {name}")

    check_schemas(report, tables)
    check_figures(report, figures, metadata)
    check_provenance(report, config, metadata, output_root)


def check_schemas(report: Report, tables: Path) -> None:
    import pandas as pd

    print("\n== table schemas ==")
    path = tables / "resource_single_step.csv"
    if path.exists():
        columns = set(pd.read_csv(path).columns)
        missing = RESOURCE_REQUIRED_COLUMNS - columns
        report.check(not missing, f"resource table records its assumptions (missing: {missing or 'none'})")

    path = tables / "boundary_comparison.csv"
    if path.exists():
        frame = pd.read_csv(path)
        report.check(
            frame["dirichlet_infidelity"].iloc[-1] < frame["periodic_infidelity"].iloc[-1],
            "boundary comparison: Dirichlet beats periodic against the hard-wall reference",
        )
        report.check(
            frame["cross_fidelity"].max() <= 1.0 + 1e-9,
            "no fidelity value exceeds one",
        )

    path = tables / "convergence.csv"
    if path.exists():
        frame = pd.read_csv(path)
        trotter = frame[frame.study == "trotter"]
        for name, expected in (("harmonic", 2.0), ("tilted_well", 2.0)):
            slopes = trotter[trotter.benchmark == name]["fitted_slope"].dropna().unique()
            report.check(
                len(slopes) > 0 and abs(float(slopes[0]) - expected) < 0.15,
                f"{name} Trotter slope {float(slopes[0]) if len(slopes) else float('nan'):.3f} "
                f"is near {expected}",
            )


def check_figures(report: Report, figures: Path, metadata: Path) -> None:
    import pandas as pd

    print("\n== figures ==")
    manifest_path = metadata / "figure_manifest.csv"
    if not manifest_path.exists():
        report.check(False, "figure manifest present")
        return

    manifest = pd.read_csv(manifest_path)
    for _, row in manifest.iterrows():
        found = [
            suffix for suffix in str(row["formats"]).strip("[]").replace("'", "").split(", ")
            if (figures / f"{row['filename']}.{suffix}").exists()
        ]
        report.check(bool(found), f"figure {row['figure_id']} written ({', '.join(found) or 'none'})")
    report.check(
        manifest["caption"].notna().all() and (manifest["caption"].str.len() > 20).all(),
        "every figure carries a caption in the manifest",
    )


def check_provenance(report: Report, config, metadata: Path, output_root: Path) -> None:
    print("\n== provenance ==")
    stored = read_provenance(metadata)
    current = collect_provenance(config)
    staleness = check_staleness(stored, current)

    report.check(not staleness["stale"], f"results are current ({'; '.join(staleness['reasons']) or 'no drift'})")
    if stored is not None:
        hashes = verify_output_hashes(stored, output_root)
        report.check(hashes["ok"], f"output hashes match ({hashes['n_checked']} files checked)")
        if stored.git_dirty:
            print("      NOTE: produced from a dirty working tree (labelled, not a failure)")


def execute_notebooks(report: Report) -> None:
    print("\n== notebook execution ==")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "execute_notebooks.py")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    report.check(result.returncode == 0, "all notebooks execute from a fresh kernel")
    if result.returncode != 0:
        print(result.stdout[-2000:])


def check_local_links(report: Report) -> None:
    print("\n== local links ==")
    broken = []
    import re

    for path in list(ROOT.glob("*.md")) + list((ROOT / "docs").glob("*.md")):
        for target in re.findall(r"\]\(([^)#][^)]*)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (path.parent / target).exists():
                broken.append(f"{path.name} -> {target}")
    report.check(not broken, f"no broken local links ({len(broken)} broken)")
    for entry in broken:
        print(f"      {entry}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--full", action="store_true", help="also execute the notebooks")
    parser.add_argument("--skip-tests", action="store_true")
    arguments = parser.parse_args()

    report = Report()
    if not arguments.skip_tests:
        run_tests(report)
    check_outputs(report, arguments.profile)
    check_local_links(report)
    if arguments.full:
        execute_notebooks(report)

    print(f"\n{len(report.lines) - report.failed}/{len(report.lines)} checks passed")
    if report.failed:
        print(json.dumps([m for ok, m in report.lines if not ok], indent=2))
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

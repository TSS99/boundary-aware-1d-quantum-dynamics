"""Regenerate every figure, table and metadata file for a profile.

Usage::

    python scripts/reproduce.py --profile smoke
    python scripts/reproduce.py --profile paper

Everything written under ``results/<profile>/`` is derived from
``configs/<profile>.yaml`` and the package source, and is stamped with a
provenance record so that stale outputs are detectable.  Nothing here contacts a
network or a remote repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boundary_aware_dynamics import plotting  # noqa: E402
from boundary_aware_dynamics.circuits.resources import (  # noqa: E402
    approximate_qft_study,
    propagation_resources,
    scaling_table,
)
from boundary_aware_dynamics.circuits.state_preparation import (  # noqa: E402
    measurement_note,
    shot_budget_for_density,
)
from boundary_aware_dynamics.config import config_to_dict, load_config  # noqa: E402
from boundary_aware_dynamics.plotting import FigureManifest, FigureRecord, save_figure  # noqa: E402
from boundary_aware_dynamics.provenance import (  # noqa: E402
    collect_provenance,
    hash_directory,
    write_provenance,
)
from boundary_aware_dynamics.workflows import (  # noqa: E402
    fit_convergence_slope,
    grid_convergence_study,
    run_benchmark,
    run_boundary_comparison,
    trotter_convergence_study,
)

BENCHMARKS = ("harmonic", "infinite_well", "tilted_well")
NUMERICAL_FLOOR = 1e-13


def write_table(frame: pd.DataFrame, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    frame.to_csv(path, index=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke", help="configs/<profile>.yaml")
    parser.add_argument("--output", default=None, help="override the output directory")
    arguments = parser.parse_args()

    started = time.perf_counter()
    config = load_config(ROOT / "configs" / f"{arguments.profile}.yaml")
    output_root = Path(arguments.output) if arguments.output else ROOT / config.output_root
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    metadata_dir = output_root / "metadata"
    for directory in (figures_dir, tables_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    plotting.apply_style(config.plotting.mode)
    formats = tuple(config.plotting.formats)
    dpi = config.plotting.dpi
    manifest = FigureManifest()
    key_results: dict[str, object] = {"profile": config.profile, "config_hash": config.config_hash}

    def record(figure_id: str, stem: str, source_data: str, caption: str, **parameters) -> None:
        manifest.add(
            FigureRecord(
                figure_id=figure_id,
                filename=f"{stem}",
                formats=list(formats),
                source_notebook=parameters.pop("notebook", "scripts/reproduce.py"),
                source_data=source_data,
                config_hash=config.config_hash,
                width_mm=parameters.pop("width_mm", 85.0),
                height_mm=parameters.pop("height_mm", 65.0),
                caption=caption,
                generation_command=f"python scripts/reproduce.py --profile {config.profile}",
                key_parameters=parameters,
            )
        )

    # ---------------------------------------------------------- benchmarks --
    print(f"[reproduce] profile={config.profile} hash={config.config_hash}")
    parameter_rows, error_rows, observable_rows = [], [], []
    runtimes: dict[str, float] = {}

    for name in BENCHMARKS:
        result = run_benchmark(config, name)
        print(
            f"  {name:14s} N={result.grid.n_grid:4d} r={result.metadata['n_steps']:5d} "
            f"{result.metadata['transform']:6s} infidelity={result.final_errors.infidelity:.3e}"
        )
        benchmark = config.benchmark(name)
        parameter_rows.append(
            {
                "benchmark": name,
                "boundary": result.grid.boundary,
                "transform": result.metadata["transform"],
                "n_grid": result.grid.n_grid,
                "n_data_qubits": result.grid.n_data_qubits,
                "n_steps": result.metadata["n_steps"],
                "time_step": result.metadata["time_step"],
                "t_max": benchmark.time_grid.t_max,
                "x_left": benchmark.domain.x_left,
                "x_right": benchmark.domain.x_right,
                "grid_spacing": result.grid.spacing,
                "centre": benchmark.initial_state.centre,
                "momentum": benchmark.initial_state.momentum,
                "sigma_density_stdev": benchmark.initial_state.sigma,
                "omega": benchmark.omega,
                "tilt_force": benchmark.tilt_force,
                "hbar": config.physics.hbar,
                "mass": config.physics.mass,
                "reference_method": result.metadata["reference_method"],
                "final_infidelity": result.final_errors.infidelity,
                "final_l2_state_error": result.final_errors.l2_state_error,
                "max_norm_error": result.propagation.max_norm_error,
                # Runtime is deliberately NOT recorded here: it varies between
                # runs and would make a deterministic table differ every time.
                # It lives in the `volatile` section of provenance.json instead.
            }
        )
        runtimes[name] = round(result.runtime_seconds, 3)
        for index, time_value in enumerate(result.times):
            errors = result.errors[index]
            error_rows.append(
                {
                    "benchmark": name, "time": time_value,
                    "infidelity": errors.infidelity,
                    "l2_state_error": errors.l2_state_error,
                    "density_l1_error": errors.density_l1_error,
                    "density_linf_error": errors.density_linf_error,
                    "norm_error": errors.norm_error,
                    "wall_residual": result.boundary[index]["wall_residual"],
                    "near_wall_probability": result.boundary[index]["near_wall_probability"],
                }
            )
        for index, time_value in enumerate(result.times):
            observable_rows.append(
                {"benchmark": name, "time": time_value,
                 **{key: values[index] for key, values in result.observables.items()}}
            )

        indices = np.unique(np.linspace(0, len(result.times) - 1, 4).astype(int))
        figure = plotting.plot_density_snapshots(
            result.grid.positions, result.times,
            {"reference": result.reference.states,
             "dirichlet" if result.grid.boundary == "dirichlet" else "periodic":
                 result.propagation.states},
            indices,
        )
        save_figure(figure, f"{name}_density_snapshots", figures_dir, formats, dpi)
        matplotlib.pyplot.close(figure)
        record(
            f"{name}_density", f"{name}_density_snapshots", f"tables/{name}_errors.csv",
            f"Probability density of the {name.replace('_', ' ')} benchmark at four times, "
            f"numerical propagation against the {result.metadata['reference_method']} reference.",
            n_grid=result.grid.n_grid, n_steps=result.metadata["n_steps"],
        )

        figure = plotting.plot_error_vs_time(
            result.times,
            {"dirichlet" if result.grid.boundary == "dirichlet" else "periodic":
                 np.array([e.infidelity for e in result.errors])},
            floor=NUMERICAL_FLOOR,
        )
        save_figure(figure, f"{name}_infidelity_vs_time", figures_dir, formats, dpi)
        matplotlib.pyplot.close(figure)
        record(
            f"{name}_infidelity", f"{name}_infidelity_vs_time", f"tables/{name}_errors.csv",
            f"Infidelity against time for the {name.replace('_', ' ')} benchmark. "
            f"Plotted logarithmically because the values approach the numerical floor.",
        )

    write_table(pd.DataFrame(parameter_rows), tables_dir, "benchmark_parameters.csv")
    write_table(pd.DataFrame(error_rows), tables_dir, "benchmark_errors.csv")
    write_table(pd.DataFrame(observable_rows), tables_dir, "benchmark_observables.csv")

    # ------------------------------------------------ boundary comparison --
    print("  boundary comparison ...")
    comparison = run_boundary_comparison(config)
    dirichlet_infidelity = np.array([e.infidelity for e in comparison.dirichlet_errors])
    periodic_infidelity = np.array([e.infidelity for e in comparison.periodic_errors])
    dirichlet_wall = np.array([d["wall_residual"] for d in comparison.dirichlet_boundary])
    periodic_wall = np.array([d["wall_residual"] for d in comparison.periodic_boundary])
    cross = comparison.cross_fidelity

    write_table(
        pd.DataFrame(
            {
                "time": comparison.times,
                "dirichlet_infidelity": dirichlet_infidelity,
                "periodic_infidelity": periodic_infidelity,
                "cross_fidelity": cross,
                "dirichlet_wall_residual": dirichlet_wall,
                "periodic_wall_residual": periodic_wall,
                "dirichlet_wrap_around": [d["wrap_around_probability"] for d in comparison.dirichlet_boundary],
                "periodic_wrap_around": [d["wrap_around_probability"] for d in comparison.periodic_boundary],
            }
        ),
        tables_dir, "boundary_comparison.csv",
    )

    figure = plotting.plot_boundary_comparison(
        comparison.times, dirichlet_infidelity, periodic_infidelity, cross, dirichlet_wall, periodic_wall
    )
    save_figure(figure, "boundary_comparison", figures_dir, formats, dpi)
    matplotlib.pyplot.close(figure)
    record(
        "boundary_comparison", "boundary_comparison", "tables/boundary_comparison.csv",
        "Periodic versus Dirichlet propagation of the same hard-wall problem, identical in every "
        "respect but the spectral transform. (a) infidelity of each against an independent "
        "finite-difference hard-wall reference; (b) divergence between the two propagations; "
        "(c) residual wall amplitude. The periodic representation fails once the packet reaches "
        "the wall.",
        width_mm=170.0, height_mm=52.0,
    )

    snapshots = np.unique(np.linspace(0, len(comparison.times) - 1, 4).astype(int))
    figure = plotting.plot_density_snapshots(
        comparison.grid.positions, comparison.times,
        {"reference": comparison.reference_states,
         "dirichlet": comparison.dirichlet_states,
         "periodic": comparison.periodic_states},
        snapshots,
    )
    save_figure(figure, "boundary_density_snapshots", figures_dir, formats, dpi)
    matplotlib.pyplot.close(figure)
    record(
        "boundary_density", "boundary_density_snapshots", "tables/boundary_comparison.csv",
        "Densities under both boundary topologies against the independent hard-wall reference. "
        "The periodic solution wraps around the box edge; the Dirichlet solution reflects.",
    )

    key_results["boundary_comparison"] = {
        "final_dirichlet_infidelity": float(dirichlet_infidelity[-1]),
        "final_periodic_infidelity": float(periodic_infidelity[-1]),
        "minimum_cross_fidelity": float(cross.min()),
        "max_dirichlet_wall_residual": float(dirichlet_wall.max()),
        "max_periodic_wall_residual": float(periodic_wall.max()),
        "reference_method": comparison.reference_method,
    }

    # ----------------------------------------------------- convergence -----
    print("  convergence studies ...")
    convergence_rows = []
    for name in BENCHMARKS:
        study = trotter_convergence_study(config, name)
        for index, steps in enumerate(study.values):
            convergence_rows.append(
                {
                    "benchmark": name, "study": "trotter", "n_steps": int(steps),
                    "step_size": study.step_sizes[index],
                    "l2_state_error": study.l2_state_error[index],
                    "infidelity": study.infidelity[index],
                    "fitted_slope": study.fit.get("slope", float("nan")),
                    "fit_r_squared": study.fit.get("r_squared", float("nan")),
                    "fit_from_index": study.fit.get("fit_from_index", -1),
                    "reference_kind": study.reference_kind,
                }
            )
        key_results.setdefault("trotter_convergence", {})[name] = {
            "slope": study.fit.get("slope"),
            "r_squared": study.fit.get("r_squared"),
            "note": study.fit.get("note", ""),
        }
        if np.isfinite(study.fit.get("slope", np.nan)):
            figure = plotting.plot_convergence(
                study.step_sizes, study.l2_state_error, study.fit, expected_slope=2.0,
                floor=NUMERICAL_FLOOR,
            )
            save_figure(figure, f"{name}_trotter_convergence", figures_dir, formats, dpi)
            matplotlib.pyplot.close(figure)
            record(
                f"{name}_trotter", f"{name}_trotter_convergence", "tables/convergence.csv",
                f"Trotter convergence of the {name.replace('_', ' ')} benchmark against exact "
                f"diagonalisation of the same discrete Hamiltonian, so that spatial discretisation "
                f"cancels. Fitted slope {study.fit['slope']:.2f} over the shaded interval.",
            )

        grid_study = grid_convergence_study(config, name)
        for index, size in enumerate(grid_study.values):
            convergence_rows.append(
                {
                    "benchmark": name, "study": "grid", "n_grid": int(size),
                    "step_size": grid_study.step_sizes[index],
                    "l2_state_error": grid_study.l2_state_error[index],
                    "infidelity": grid_study.infidelity[index],
                    "reference_kind": grid_study.reference_kind,
                }
            )
    write_table(pd.DataFrame(convergence_rows), tables_dir, "convergence.csv")

    # ------------------------------------------------------- resources -----
    print("  resource analysis ...")
    resource_rows = []
    for name in ("harmonic", "tilted_well"):
        for synthesis in ("structured", "generic_diagonal"):
            for connectivity in ("all_to_all", "linear"):
                row = propagation_resources(
                    config, name, n_steps=1, synthesis=synthesis, connectivity=connectivity
                )
                resource_rows.append(row.as_dict())
    write_table(pd.DataFrame(resource_rows), tables_dir, "resource_single_step.csv")

    scaling_rows = []
    for name in ("harmonic", "tilted_well"):
        for row in scaling_table(config, name, (8, 16, 32, 64, 128), n_steps=1):
            scaling_rows.append(row.as_dict())
    scaling_frame = pd.DataFrame(scaling_rows)
    write_table(scaling_frame, tables_dir, "resource_scaling.csv")

    figure = plotting.plot_resource_scaling(
        scaling_frame[scaling_frame.benchmark == "tilted_well"].n_total_qubits.values,
        {
            "Dirichlet (QST, +2 ancillas)":
                scaling_frame[scaling_frame.benchmark == "tilted_well"].two_qubit_gates.values,
            "Periodic (QFT, no ancillas)":
                scaling_frame[scaling_frame.benchmark == "harmonic"].two_qubit_gates.values,
        },
    )
    save_figure(figure, "resource_scaling", figures_dir, formats, dpi)
    matplotlib.pyplot.close(figure)
    record(
        "resource_scaling", "resource_scaling", "tables/resource_scaling.csv",
        "Two-qubit gate count of one propagation step against total register size, with "
        "structured phase synthesis and all-to-all connectivity. The Dirichlet register includes "
        "the two QST ancillas.",
    )

    approximation_rows = approximate_qft_study(config, "harmonic", degrees=(0, 1, 2, 3), n_steps=1)
    write_table(pd.DataFrame(approximation_rows), tables_dir, "approximate_qft.csv")

    steps_rows = [
        propagation_resources(config, "tilted_well", n_steps=steps).as_dict()
        for steps in (1, 2, 4, 8, 16)
    ]
    write_table(pd.DataFrame(steps_rows), tables_dir, "resource_vs_steps.csv")

    # -------------------------------------------------- graphical abstract --
    figure = plotting.plot_graphical_abstract()
    save_figure(figure, "graphical_abstract", figures_dir, formats, dpi)
    matplotlib.pyplot.close(figure)
    record(
        "graphical_abstract", "graphical_abstract", "n/a",
        "Boundary condition determines grid, transform and circuit structure, and hence the "
        "physics the propagator represents.",
        width_mm=170.0, height_mm=46.0,
    )

    # -------------------------------------------------------- metadata -----
    manifest.write(metadata_dir)
    single_step = pd.DataFrame(resource_rows)
    structured = single_step[
        (single_step.synthesis_model == "structured") & (single_step.connectivity == "all_to_all")
    ]
    key_results["resources"] = {
        "tilted_well_two_qubit_gates_single_step": int(
            structured[structured.benchmark == "tilted_well"].two_qubit_gates.iloc[0]
        ),
        "tilted_well_total_qubits": int(
            structured[structured.benchmark == "tilted_well"].n_total_qubits.iloc[0]
        ),
        "harmonic_two_qubit_gates_single_step": int(
            structured[structured.benchmark == "harmonic"].two_qubit_gates.iloc[0]
        ),
    }
    key_results["measurement"] = measurement_note()
    key_results["shots_for_one_percent_density"] = shot_budget_for_density(0.01)

    (metadata_dir / "key_results.json").write_text(
        json.dumps(key_results, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    (metadata_dir / "config.json").write_text(
        json.dumps(config_to_dict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    write_paper_macros(metadata_dir / "paper_values.tex", key_results)

    provenance = collect_provenance(config, runtime_seconds=time.perf_counter() - started)
    provenance.volatile["benchmark_runtime_seconds"] = runtimes
    provenance.output_hashes = {
        **{f"figures/{k}": v for k, v in hash_directory(figures_dir).items()},
        **{f"tables/{k}": v for k, v in hash_directory(tables_dir).items()},
    }
    write_provenance(provenance, metadata_dir)

    if not provenance.is_reproducible_artefact:
        print("  NOTE: generated from a dirty working tree; labelled in provenance.json")
    print(f"[reproduce] done in {time.perf_counter() - started:.1f}s -> {output_root}")
    return 0


def write_paper_macros(path: Path, key_results: dict) -> None:
    """Emit LaTeX macros so manuscript numbers are never typed by hand."""
    boundary = key_results["boundary_comparison"]
    trotter = key_results["trotter_convergence"]
    resources = key_results["resources"]

    def macro(name: str, value: object) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}\n"

    lines = [
        "% Generated by scripts/reproduce.py. Do not edit by hand.\n",
        f"% profile={key_results['profile']} config_hash={key_results['config_hash']}\n",
        macro("ConfigHash", key_results["config_hash"]),
        macro("BoundaryDirichletInfidelity", f"{boundary['final_dirichlet_infidelity']:.2e}"),
        macro("BoundaryPeriodicInfidelity", f"{boundary['final_periodic_infidelity']:.3f}"),
        macro("BoundaryMinCrossFidelity", f"{boundary['minimum_cross_fidelity']:.3f}"),
        macro("BoundaryDirichletWall", f"{boundary['max_dirichlet_wall_residual']:.3f}"),
        macro("BoundaryPeriodicWall", f"{boundary['max_periodic_wall_residual']:.3f}"),
        macro("HarmonicTrotterSlope", f"{trotter['harmonic']['slope']:.2f}"),
        macro("TiltedTrotterSlope", f"{trotter['tilted_well']['slope']:.2f}"),
        macro("TiltedTwoQubitGates", resources["tilted_well_two_qubit_gates_single_step"]),
        macro("TiltedTotalQubits", resources["tilted_well_total_qubits"]),
        macro("HarmonicTwoQubitGates", resources["harmonic_two_qubit_gates_single_step"]),
        macro("DensityShotBudget", key_results["shots_for_one_percent_density"]),
    ]
    path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

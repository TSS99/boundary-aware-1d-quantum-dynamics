"""Run the propagation circuits on IBM hardware and compare against the ideal.

This is a standalone study that sits beside the simulator results: nothing under
``results/paper`` depends on it, and it is never part of ``reproduce.py``.  It
answers one question -- how far a real device drifts from the noiseless circuit
as the Trotter step count grows -- on the smallest grid that still carries every
structural feature of the method.

Usage::

    export IBM_QUANTUM_TOKEN=...          # never stored in this repository
    export IBM_QUANTUM_INSTANCE=...       # optional; defaults to 'auto'
    python scripts/hardware_run.py submit --backend ibm_marrakesh
    python scripts/hardware_run.py fetch

Token and instance are read from the environment only.  Neither is written to
``results/hardware/``: the token is a credential and the instance names an
account, and this directory is committed.  Job identifiers, tags and all
returned counts are written there so the analysis can be repeated without
re-running the device.

Experiment
----------
Every circuit evolves to the same final time; ``r`` sets how finely that
interval is split.  Increasing ``r`` therefore *reduces* Trotter error and
*increases* device error, and the interesting quantity is where those two cross.
Three circuit groups make that separable:

``propagate``
    Preparation, ``r`` Strang steps, measurement.  Compared against the
    noiseless simulation of the same circuit (device error) and against exact
    diagonalisation of the same discrete Hamiltonian (total error).
``echo``
    Preparation, ``r`` steps, the inverse of those steps, measurement.  The
    ideal result is the initial density exactly, so any deviation is device
    error alone, with the Trotter error cancelled by construction.
``baseline``
    Preparation and measurement only: state preparation and readout error with
    no propagation at all.

On the hard-wall benchmarks the two QST ancillas return to |00> exactly after
every step, so they are measured as well and used as an error-detection flag.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qiskit import ClassicalRegister, QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402

from boundary_aware_dynamics.circuits.resources import _benchmark_circuit  # noqa: E402
from boundary_aware_dynamics.circuits.state_preparation import (  # noqa: E402
    exact_state_preparation_circuit,
)
from boundary_aware_dynamics.config import load_config  # noqa: E402
from boundary_aware_dynamics.grids import (  # noqa: E402
    dirichlet_midpoint_grid,
    fft_kinetic_energies,
    periodic_grid,
    sine_mode_energies,
)
from boundary_aware_dynamics.propagators import harmonic_potential, tilted_potential  # noqa: E402
from boundary_aware_dynamics.states import gaussian_wavepacket, sine_windowed_gaussian  # noqa: E402
from boundary_aware_dynamics.transforms import dft_matrix, dst2_matrix  # noqa: E402

N_GRID = 8
PLAN = {"harmonic": (1, 2, 4, 8), "infinite_well": (1, 2, 4), "tilted_well": (1, 2, 4)}
PROPAGATE_SHOTS = 8192
CONTROL_SHOTS = 4096
OUTPUT = ROOT / "results" / "hardware"
JOB_RECORD = OUTPUT / "jobs.json"
STUDY_TAG = "boundary-aware-dynamics"


@dataclass
class CircuitSpec:
    """One submitted circuit and everything needed to interpret its counts."""

    label: str
    benchmark: str
    group: str
    n_steps: int
    n_grid: int
    n_qubits: int
    n_data_qubits: int
    boundary: str
    transform: str
    shots: int
    two_qubit_gates: int
    depth: int


def initial_state(config, name: str, grid) -> np.ndarray:
    """The benchmark's own initial state, sampled on the display grid."""
    benchmark = config.benchmark(name)
    state = benchmark.initial_state
    if state.sine_window:
        return sine_windowed_gaussian(
            grid.positions, grid.spacing, benchmark.domain.length,
            state.centre, state.momentum, state.sigma,
        )
    return gaussian_wavepacket(
        grid.positions, grid.spacing, state.centre, state.momentum, state.sigma
    )


def benchmark_grid(config, name: str):
    """The display grid for one benchmark."""
    benchmark = config.benchmark(name)
    if benchmark.domain.boundary == "periodic":
        return periodic_grid(benchmark.domain.x_left, benchmark.domain.x_right, N_GRID)
    return dirichlet_midpoint_grid(benchmark.domain.length, N_GRID)


def logical_circuit(config, name: str, n_steps: int, group: str) -> tuple[QuantumCircuit, CircuitSpec]:
    """Preparation, propagation and measurement, before transpilation."""
    grid = benchmark_grid(config, name)
    psi = initial_state(config, name, grid)
    built = _benchmark_circuit(config, name, n_steps, N_GRID, "structured", 0)
    core = built.circuit

    n_total = core.num_qubits
    n_data = built.n_data_qubits
    # The Dirichlet construction embeds the data register between the ancillas.
    data_qubits = list(range(n_data)) if built.boundary == "periodic" else list(range(1, n_data + 1))
    ancillas = [q for q in range(n_total) if q not in data_qubits]

    circuit = QuantumCircuit(n_total, name=f"{name}_{group}_r{n_steps}")
    circuit.compose(exact_state_preparation_circuit(psi, grid.spacing), qubits=data_qubits, inplace=True)
    if group != "baseline":
        circuit.compose(core, inplace=True)
    if group == "echo":
        circuit.compose(core.inverse(), inplace=True)

    data_bits = ClassicalRegister(n_data, "position")
    circuit.add_register(data_bits)
    circuit.measure(data_qubits, data_bits)
    if ancillas:
        ancilla_bits = ClassicalRegister(len(ancillas), "ancilla")
        circuit.add_register(ancilla_bits)
        circuit.measure(ancillas, ancilla_bits)

    spec = CircuitSpec(
        label=circuit.name,
        benchmark=name,
        group=group,
        n_steps=0 if group == "baseline" else n_steps,
        n_grid=N_GRID,
        n_qubits=n_total,
        n_data_qubits=n_data,
        boundary=built.boundary,
        transform=built.transform,
        shots=PROPAGATE_SHOTS if group == "propagate" else CONTROL_SHOTS,
        two_qubit_gates=0,
        depth=0,
    )
    return circuit, spec


def build_all(config) -> list[tuple[QuantumCircuit, CircuitSpec]]:
    """The full inventory: propagation, echo controls and preparation baselines."""
    items = []
    for name, steps in PLAN.items():
        for r in steps:
            items.append(logical_circuit(config, name, r, "propagate"))
        for r in steps:
            items.append(logical_circuit(config, name, r, "echo"))
        items.append(logical_circuit(config, name, 1, "baseline"))
    return items


def ideal_density(circuit: QuantumCircuit, spec: CircuitSpec) -> np.ndarray:
    """Noiseless density over the data register, ancillas postselected on zero."""
    stripped = circuit.remove_final_measurements(inplace=False)
    probabilities = Statevector.from_instruction(stripped).probabilities_dict()
    density = np.zeros(2 ** spec.n_data_qubits)
    data_qubits = (
        list(range(spec.n_data_qubits))
        if spec.boundary == "periodic"
        else list(range(1, spec.n_data_qubits + 1))
    )
    ancillas = [q for q in range(spec.n_qubits) if q not in data_qubits]
    for bitstring, probability in probabilities.items():
        bits = bitstring[::-1]  # Qiskit prints the highest qubit first
        if any(bits[q] == "1" for q in ancillas):
            continue
        index = sum(int(bits[q]) << position for position, q in enumerate(data_qubits))
        density[index] += probability
    return density


def exact_density(config, name: str) -> np.ndarray:
    """Density from exact diagonalisation of the *same discrete* Hamiltonian.

    Not a continuum solution: at eight points the spatial discretisation error
    is large, and mixing it in would hide the two errors this study is about.
    Splitting error is what separates this from the circuit; device noise is
    what separates the circuit from the hardware.
    """
    benchmark = config.benchmark(name)
    grid = benchmark_grid(config, name)
    physics = config.physics

    if benchmark.domain.boundary == "periodic":
        potential = harmonic_potential(grid.positions, physics.mass, benchmark.omega or 0.0)
        transform = dft_matrix(N_GRID)
        kinetic = fft_kinetic_energies(N_GRID, grid.spacing, physics.mass, physics.hbar)
    else:
        potential = (
            tilted_potential(grid.positions, benchmark.domain.length, benchmark.tilt_force)
            if benchmark.tilt_force
            else np.zeros_like(grid.positions)
        )
        transform = dst2_matrix(N_GRID)
        kinetic = sine_mode_energies(
            N_GRID, benchmark.domain.length, physics.mass, physics.hbar
        )

    hamiltonian = transform.conj().T @ np.diag(kinetic) @ transform + np.diag(potential)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phase = np.exp(-1j * eigenvalues * benchmark.time_grid.t_max / physics.hbar)
    propagator = eigenvectors @ np.diag(phase) @ eigenvectors.conj().T

    state = propagator @ initial_state(config, name, grid)
    density = np.abs(state) ** 2
    return density / density.sum()


def total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(first) - np.asarray(second)).sum())


def connect():
    """Open the runtime service from the environment.

    Both the token and the instance are read from the environment and neither is
    written to disk. The instance names an account, so recording it here would
    put an account identifier into a public repository for no benefit; ``auto``
    selects it at run time instead.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        raise SystemExit("Set IBM_QUANTUM_TOKEN in the environment; it is never stored in the repo.")
    return QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=os.environ.get("IBM_QUANTUM_INSTANCE", "auto"),
    )


def write_figures(frame, densities: dict) -> None:
    """Two figures: where the two error sources cross, and what the device returns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from boundary_aware_dynamics import plotting

    plotting.apply_style("publication")
    palette = plotting.PALETTE
    figures_dir = OUTPUT / "figures"
    titles = {
        "harmonic": "harmonic (QFT, 3 qubits)",
        "infinite_well": "free well (QST, 5 qubits)",
        "tilted_well": "tilted well (QST, 5 qubits)",
    }
    labels = ("(a)", "(b)", "(c)")

    # --- error against step count -------------------------------------------
    figure, axes = plt.subplots(
        1, 3, figsize=(plotting.millimetres(170.0), plotting.millimetres(62.0)),
        constrained_layout=True,
    )
    propagate = frame[frame.group == "propagate"]
    for axis, (name, panel) in zip(axes, ((n, propagate[propagate.benchmark == n]) for n in PLAN)):
        panel = panel.sort_values("n_steps")
        axis.semilogy(
            panel.n_steps, panel.tvd_hardware_vs_ideal_circuit, color=palette["periodic"],
            linestyle="--", marker="s", markerfacecolor="white", markeredgewidth=0.9,
            label="device error (hardware vs ideal circuit)",
        )
        axis.semilogy(
            panel.n_steps, panel.tvd_ideal_circuit_vs_exact, color=palette["dirichlet"],
            linestyle="-", marker="o", markerfacecolor="white", markeredgewidth=0.9,
            label="splitting error (ideal circuit vs exact)",
        )
        axis.semilogy(
            panel.n_steps, panel.tvd_hardware_vs_exact, color=palette["reference"],
            linestyle="-.", marker="^", markerfacecolor="white", markeredgewidth=0.9,
            label="total error (hardware vs exact)",
        )
        # The free well has no splitting error at all -- [T, V] = 0 makes the
        # circuit exact -- so its curve sits at machine precision. Letting that
        # set the axis would squash the two curves that carry information into
        # one decade, so the axis is scaled to the measurable series and the
        # exact one is stated in words instead.
        measurable = np.concatenate([
            panel.tvd_hardware_vs_ideal_circuit.to_numpy(),
            panel.tvd_hardware_vs_exact.to_numpy(),
            panel.tvd_ideal_circuit_vs_exact.to_numpy(),
        ])
        measurable = measurable[measurable > 1e-12]
        axis.set_ylim(measurable.min() / 3.0, measurable.max() * 4.0)
        if (panel.tvd_ideal_circuit_vs_exact < 1e-12).all():
            axis.text(
                0.5, 0.06,
                "splitting error $<10^{-12}$:\ncircuit is exact here",
                transform=axis.transAxes, ha="center", va="bottom",
                fontsize=matplotlib.rcParams["font.size"] - 1.5, color=palette["dirichlet"],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(panel.n_steps.unique()))
        axis.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axis.set_xlabel("Trotter steps $r$")
        axis.set_title(titles[name], fontsize=matplotlib.rcParams["font.size"] - 0.5)
        axis.grid(True, color=palette["grid"], linewidth=0.4, alpha=0.7)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Total variation distance")
    for axis, label in zip(axes, labels):
        axis.text(-0.18, 1.10, label, transform=axis.transAxes,
                  fontsize=matplotlib.rcParams["font.size"] + 1, fontweight="bold")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=3, columnspacing=1.6)
    plotting.save_figure(figure, "hardware_error_vs_steps", figures_dir)
    plt.close(figure)

    # --- every measured density, hardware against the ideal simulator --------
    widest = max(len(steps) for steps in PLAN.values())
    figure, axes = plt.subplots(
        len(PLAN), widest,
        figsize=(plotting.millimetres(170.0), plotting.millimetres(105.0)),
        constrained_layout=True, sharex=True,
    )
    for row_index, name in enumerate(PLAN):
        panel = propagate[propagate.benchmark == name].sort_values("n_steps")
        for column_index in range(widest):
            axis = axes[row_index, column_index]
            if column_index >= len(panel):
                axis.set_axis_off()
                continue
            entry = panel.iloc[column_index]
            data = densities[entry.label]
            centres = np.arange(len(data["exact"]))
            axis.bar(centres, data["exact"], width=0.85, color=palette["grid"],
                     edgecolor="none", label="exact (discrete)")
            axis.step(centres, data["ideal_circuit"], where="mid", color=palette["dirichlet"],
                      linewidth=1.4, label="ideal simulator")
            axis.step(centres, data["hardware"], where="mid", color=palette["periodic"],
                      linewidth=1.4, linestyle="--", label="hardware")
            axis.set_title(
                f"$r={int(entry.n_steps)}$,  {int(entry.two_qubit_gates)} two-qubit gates",
                fontsize=matplotlib.rcParams["font.size"] - 1.0,
            )
            axis.text(
                0.03, 0.97,
                f"device TVD {entry.tvd_hardware_vs_ideal_circuit:.3f}",
                transform=axis.transAxes, ha="left", va="top",
                fontsize=matplotlib.rcParams["font.size"] - 2.0, color=palette["periodic"],
            )
            axis.set_ylim(0.0, max(max(data["exact"]), max(data["hardware"]),
                                   max(data["ideal_circuit"])) * 1.38)
            axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
            axis.grid(True, axis="y", color=palette["grid"], linewidth=0.4, alpha=0.7)
            axis.set_axisbelow(True)
            # Shared x hides tick labels on every axis that has another below it,
            # including axes whose lower neighbours were switched off, so the
            # bottom-most *populated* axis of each column re-enables them.
            below = [
                len(propagate[propagate.benchmark == other].index)
                for other in list(PLAN)[row_index + 1:]
            ]
            if all(column_index >= count for count in below):
                axis.set_xlabel("grid point $j$")
                axis.tick_params(labelbottom=True)
        axes[row_index, 0].set_ylabel(f"{titles[name].split(' (')[0]}\nprobability")
    for axis, label in zip(axes[:, 0], labels):
        axis.text(-0.34, 1.12, label, transform=axis.transAxes,
                  fontsize=matplotlib.rcParams["font.size"] + 1, fontweight="bold")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=3, columnspacing=1.6)
    plotting.save_figure(figure, "hardware_density_comparison", figures_dir)
    plt.close(figure)

    # --- what was actually executed: gate counts -----------------------------
    system_colour = {
        "harmonic": palette["third"],
        "infinite_well": palette["dirichlet"],
        "tilted_well": palette["periodic"],
    }
    figure, axes = plt.subplots(
        1, 3, figsize=(plotting.millimetres(170.0), plotting.millimetres(62.0)),
        constrained_layout=True,
    )
    for name in PLAN:
        for group, style, alpha in (("propagate", "-", 1.0), ("echo", "--", 0.55)):
            panel = frame[(frame.benchmark == name) & (frame.group == group)].sort_values("n_steps")
            axes[0].plot(panel.n_steps, panel.two_qubit_gates, style, color=system_colour[name],
                         marker="o", markerfacecolor="white", markeredgewidth=0.9, alpha=alpha,
                         label=f"{titles[name].split(' (')[0]} ({group})")
            axes[1].plot(panel.n_steps, panel.depth, style, color=system_colour[name],
                         marker="o", markerfacecolor="white", markeredgewidth=0.9, alpha=alpha)
    for axis, ylabel in ((axes[0], "Two-qubit gates (cz)"), (axes[1], "Circuit depth")):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(sorted(frame[frame.group == "propagate"].n_steps.unique()))
        axis.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axis.set_xlabel("Trotter steps $r$")
        axis.set_ylabel(ylabel)
        axis.grid(True, color=palette["grid"], linewidth=0.4, alpha=0.7)
        axis.set_axisbelow(True)

    # Measurement is not a gate, so it is excluded from the composition.
    gate_columns = [
        column for column in frame.columns
        if column.startswith("gate_") and column != "gate_measure"
    ]
    propagate_sorted = frame[frame.group == "propagate"].sort_values(["benchmark", "n_steps"])
    # Bars are grouped by system with a gap between groups, so the system can be
    # named once per group instead of once per bar, which does not fit.
    positions, group_spans = [], {}
    cursor = 0.0
    for name in ("harmonic", "infinite_well", "tilted_well"):
        members = propagate_sorted[propagate_sorted.benchmark == name]
        start = cursor
        for _ in range(len(members)):
            positions.append(cursor)
            cursor += 1.0
        group_spans[name] = (start, cursor - 1.0)
        cursor += 0.8
    positions = np.array(positions)
    bottom = np.zeros(len(propagate_sorted))
    gate_shades = {
        "gate_cz": palette["reference"], "gate_rz": palette["dirichlet"],
        "gate_sx": palette["periodic"], "gate_x": palette["third"],
    }
    for column in sorted(gate_columns):
        values = propagate_sorted[column].fillna(0).to_numpy(dtype=float)
        if values.sum() == 0:
            continue
        axes[2].bar(positions, values, bottom=bottom, width=0.75,
                    color=gate_shades.get(column, palette["floor"]),
                    edgecolor="none", label=column.replace("gate_", ""))
        bottom += values
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(
        [f"$r$={int(row.n_steps)}" for row in propagate_sorted.itertuples()],
        fontsize=matplotlib.rcParams["font.size"] - 2.0,
    )
    short_names = {"harmonic": "harmonic", "infinite_well": "free well", "tilted_well": "tilted well"}
    for name, (start, end) in group_spans.items():
        axes[2].text(
            0.5 * (start + end), -0.16, short_names[name], transform=axes[2].get_xaxis_transform(),
            ha="center", va="top", fontsize=matplotlib.rcParams["font.size"] - 1.5,
            color=system_colour[name], fontweight="semibold",
        )
    axes[2].set_ylabel("Gates in executed circuit")
    axes[2].grid(True, axis="y", color=palette["grid"], linewidth=0.4, alpha=0.7)
    axes[2].set_axisbelow(True)
    axes[2].legend(loc="upper left", ncol=2, fontsize=matplotlib.rcParams["font.size"] - 2.0)
    for axis, label in zip(axes, labels):
        axis.text(-0.20, 1.06, label, transform=axis.transAxes,
                  fontsize=matplotlib.rcParams["font.size"] + 1, fontweight="bold")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=3,
                  columnspacing=1.4, fontsize=matplotlib.rcParams["font.size"] - 1.5)
    plotting.save_figure(figure, "hardware_gate_counts", figures_dir)
    plt.close(figure)

    # --- controls: echo, preparation floor, error detection ------------------
    figure, axes = plt.subplots(
        1, 2, figsize=(plotting.millimetres(170.0), plotting.millimetres(58.0)),
        constrained_layout=True,
    )
    for name in PLAN:
        echo = frame[(frame.benchmark == name) & (frame.group == "echo")].sort_values("two_qubit_gates")
        baseline = frame[(frame.benchmark == name) & (frame.group == "baseline")]
        axes[0].plot(echo.two_qubit_gates, echo.tvd_hardware_vs_ideal_circuit, "-o",
                     color=system_colour[name], markerfacecolor="white", markeredgewidth=0.9,
                     label=titles[name].split(" (")[0])
        axes[0].plot(baseline.two_qubit_gates, baseline.tvd_hardware_vs_ideal_circuit, "*",
                     color=system_colour[name], markersize=9)
        wells = frame[(frame.benchmark == name) & (frame.ancilla_retention < 0.999)]
        if not wells.empty:
            wells = wells.sort_values("two_qubit_gates")
            axes[1].plot(wells.two_qubit_gates, wells.ancilla_retention, "-o",
                         color=system_colour[name], markerfacecolor="white", markeredgewidth=0.9,
                         label=titles[name].split(" (")[0])
    axes[0].set_xlabel("Two-qubit gates in executed circuit")
    axes[0].set_ylabel("Echo error, hardware vs ideal")
    axes[0].text(0.97, 0.06, "$\\star$  preparation + readout floor", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=matplotlib.rcParams["font.size"] - 1.5,
                 color=palette["floor"])
    axes[1].set_xlabel("Two-qubit gates in executed circuit")
    axes[1].set_ylabel("Ancilla postselection retention")
    axes[1].set_ylim(0.0, 1.05)
    for axis in axes:
        axis.grid(True, color=palette["grid"], linewidth=0.4, alpha=0.7)
        axis.set_axisbelow(True)
        axis.legend(loc="best", fontsize=matplotlib.rcParams["font.size"] - 1.5)
    for axis, label in zip(axes, labels):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes,
                  fontsize=matplotlib.rcParams["font.size"] + 1, fontweight="bold")
    plotting.save_figure(figure, "hardware_controls", figures_dir)
    plt.close(figure)
    print(f"[hardware] wrote figures to {figures_dir}")


def submit(arguments: argparse.Namespace) -> int:
    from qiskit_ibm_runtime import SamplerV2

    config = load_config(ROOT / "configs" / "paper.yaml")
    items = build_all(config)

    service = connect()
    backend = service.backend(arguments.backend)
    print(f"[hardware] backend={backend.name}")

    pass_manager = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=config.circuits.seed
    )
    pubs, specs = [], []
    for circuit, spec in items:
        isa = pass_manager.run(circuit)
        operations = isa.count_ops()
        spec.two_qubit_gates = sum(
            count for gate, count in operations.items()
            if gate in ("cz", "cx", "ecr")
        )
        spec.depth = int(isa.depth())
        pubs.append((isa,))
        specs.append(spec)
        print(f"  {spec.label:34s} qubits={spec.n_qubits} 2q={spec.two_qubit_gates:5d} "
              f"depth={spec.depth:5d} shots={spec.shots}")

    if len({spec.shots for spec in specs}) != 1:
        # SamplerV2 takes one shot count per pub, so state it per pub explicitly.
        pubs = [(isa, None, spec.shots) for (isa, ), spec in zip(pubs, specs)]

    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    sampler.options.environment.job_tags = [
        STUDY_TAG,
        "hardware-vs-ideal",
        f"grid-{N_GRID}",
        "trotter-sweep",
        f"backend-{backend.name}",
    ]

    job = sampler.run(pubs)
    print(f"[hardware] submitted job {job.job_id()}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    JOB_RECORD.write_text(
        json.dumps(
            {
                "job_id": job.job_id(),
                "backend": backend.name,
                "job_tags": sampler.options.environment.job_tags,
                "submitted_utc": datetime.now(timezone.utc).isoformat(),
                "n_grid": N_GRID,
                "config_hash": config.config_hash,
                "specs": [asdict(spec) for spec in specs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[hardware] wrote {JOB_RECORD}")
    return 0


def fetch(arguments: argparse.Namespace) -> int:
    import pandas as pd

    record = json.loads(JOB_RECORD.read_text(encoding="utf-8"))
    service = connect()
    job = service.job(record["job_id"])
    status = job.status()
    print(f"[hardware] job {record['job_id']} status={status}")
    if str(status) != "DONE":
        print("[hardware] not finished yet; re-run fetch later.")
        return 1

    config = load_config(ROOT / "configs" / "paper.yaml")
    items = {spec.label: (circuit, spec) for circuit, spec in build_all(config)}
    result = job.result()

    # Re-derive the gate composition of what was actually executed. Transpilation
    # is seeded, so this reproduces the submitted circuits; the two-qubit counts
    # stored at submission are checked against it rather than trusted.
    backend = service.backend(record["backend"])
    pass_manager = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=config.circuits.seed
    )
    compositions = {}
    for label, (circuit, _) in items.items():
        operations = {
            gate: int(count) for gate, count in pass_manager.run(circuit).count_ops().items()
        }
        compositions[label] = operations
    for stored in record["specs"]:
        rebuilt = sum(
            count for gate, count in compositions[stored["label"]].items()
            if gate in ("cz", "cx", "ecr")
        )
        if rebuilt != stored["two_qubit_gates"]:
            print(
                f"  WARNING {stored['label']}: re-transpiled to {rebuilt} two-qubit gates, "
                f"{stored['two_qubit_gates']} were submitted."
            )

    rows, densities = [], {}
    exact = {name: exact_density(config, name) for name in PLAN}
    for pub_result, stored in zip(result, record["specs"]):
        label = stored["label"]
        circuit, spec = items[label]
        counts = pub_result.data.position.get_counts()
        shots = sum(counts.values())
        retained = shots
        if hasattr(pub_result.data, "ancilla"):
            # Postselect shot by shot: the ancillas return to |00> exactly, so
            # any other ancilla outcome is a detected error. The two registers
            # must be paired per shot -- joining them into one bitstring and
            # splitting on whitespace does not work, there is no separator.
            position_shots = pub_result.data.position.get_bitstrings()
            ancilla_shots = pub_result.data.ancilla.get_bitstrings()
            kept: dict[str, int] = {}
            for position_bits, ancilla_bits in zip(position_shots, ancilla_shots):
                if set(ancilla_bits) == {"0"}:
                    kept[position_bits] = kept.get(position_bits, 0) + 1
            retained = sum(kept.values())
            if retained:
                counts = kept

        measured = np.zeros(2 ** spec.n_data_qubits)
        for bitstring, count in counts.items():
            measured[int(bitstring, 2)] += count
        measured = measured / measured.sum()

        ideal = ideal_density(circuit, spec)
        row = {
            "label": label,
            "benchmark": spec.benchmark,
            "group": spec.group,
            "n_steps": spec.n_steps,
            "n_qubits": spec.n_qubits,
            "two_qubit_gates": stored["two_qubit_gates"],
            "depth": stored["depth"],
            "shots": shots,
            "ancilla_retention": retained / shots if shots else float("nan"),
            "tvd_hardware_vs_ideal_circuit": total_variation(measured, ideal),
        }
        if spec.group == "propagate":
            row["tvd_hardware_vs_exact"] = total_variation(measured, exact[spec.benchmark])
            row["tvd_ideal_circuit_vs_exact"] = total_variation(ideal, exact[spec.benchmark])
        row.update({f"gate_{gate}": count for gate, count in compositions[label].items()})
        rows.append(row)
        densities[label] = {
            "hardware": measured.tolist(),
            "ideal_circuit": ideal.tolist(),
            "exact": exact[spec.benchmark].tolist(),
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "hardware_comparison.csv", index=False)
    (OUTPUT / "densities.json").write_text(json.dumps(densities, indent=2), encoding="utf-8")
    metrics = job.metrics()
    (OUTPUT / "job_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    write_figures(frame, densities)
    print(frame.to_string(index=False))
    print(f"[hardware] QPU seconds used: {metrics.get('usage', {}).get('quantum_seconds')}")
    return 0


def circuits(arguments: argparse.Namespace) -> int:
    """Draw the circuits that the device actually executed.

    These are the transpiled, backend-mapped circuits including preparation and
    measurement -- not the logical construction -- so what the figure shows is
    what ran: three data qubits, the two QST ancillas on the hard-wall
    benchmarks, and nothing but the device's own basis gates.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from boundary_aware_dynamics import plotting

    config = load_config(ROOT / "configs" / "paper.yaml")
    record = json.loads(JOB_RECORD.read_text(encoding="utf-8"))
    submitted = {spec["label"]: spec for spec in record["specs"]}

    service = connect()
    backend = service.backend(record["backend"])
    pass_manager = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=config.circuits.seed
    )

    figures_dir = OUTPUT / "figures"
    for name in PLAN:
        circuit, spec = logical_circuit(config, name, arguments.steps, "propagate")
        isa = pass_manager.run(circuit)
        two_qubit = sum(
            count for gate, count in isa.count_ops().items() if gate in ("cz", "cx", "ecr")
        )
        expected = submitted.get(f"{name}_propagate_r{arguments.steps}", {})
        agrees = expected.get("two_qubit_gates") == two_qubit
        print(
            f"  {name:14s} qubits={isa.num_qubits} 2q={two_qubit:4d} depth={isa.depth():5d} "
            f"{'matches the executed circuit' if agrees else 'DIFFERS from the executed circuit'}"
        )
        figure = plotting.plot_circuit_diagram(isa, fold=arguments.fold)
        plotting.save_figure(figure, f"{name}_hardware_circuit", figures_dir, ("pdf",))
        plt.close(figure)
    print(f"[hardware] wrote circuit figures to {figures_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    submit_parser = sub.add_parser("submit", help="build, transpile and submit the circuits")
    submit_parser.add_argument("--backend", default="ibm_marrakesh")
    submit_parser.set_defaults(handler=submit)

    fetch_parser = sub.add_parser("fetch", help="retrieve results and write the comparison")
    fetch_parser.set_defaults(handler=fetch)

    circuits_parser = sub.add_parser(
        "circuits", help="draw the transpiled circuits the device executed"
    )
    circuits_parser.add_argument("--steps", type=int, default=1)
    circuits_parser.add_argument("--fold", type=int, default=30)
    circuits_parser.set_defaults(handler=circuits)

    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())

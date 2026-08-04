"""Resource accounting for the full propagation, by component.

Three things this module is careful about, because the earlier version of this
repository got each of them wrong:

Step composition
    ``r`` Strang steps are **not** ``r`` copies of the five-block single step.
    Adjacent half-potential phases merge, ``D_V(dt/2) D_V(dt/2) = D_V(dt)``, so
    the sequence is one initial half-phase, ``r - 1`` full phases, one final
    half-phase, and ``r`` kinetic blocks.  Multiplying a single-step count by
    ``r`` overstates the potential phases by roughly a factor of two.

Ancillas
    The QST needs two ancillas beyond the data register.  Reporting the data
    qubit count as the total silently omits them.  Every row here carries data,
    ancilla and total qubit counts separately.

Barriers
    Barriers block the transpiler from cancelling gates across block boundaries,
    which inflates counts.  Counting circuits contain no barriers; barriers
    belong only in display circuits.

Counts are additionally separated by synthesis model (structured versus generic
``DiagonalGate``) and by connectivity (all-to-all versus a linear nearest-
neighbour coupling map), because a single number without those qualifiers is not
comparable to anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit.synthesis import synth_qft_full
from qiskit.transpiler import CouplingMap

from ..config import RunConfig
from ..grids import validate_power_of_two
from .phases import (
    BitExpansion,
    evaluate_bit_expansion,
    folded_sine_kinetic_expansion,
    harmonic_position_expansion,
    linear_tilt_expansion,
    signed_momentum_expansion,
    structured_phase_circuit,
)
from .qst import extended_register_size, odd_extension_circuit

__all__ = [
    "ResourceRow",
    "SynthesisModel",
    "approximate_qft_study",
    "build_propagation_core",
    "count_resources",
    "measure_qft_approximation_error",
    "propagation_resources",
    "scaling_table",
]

SynthesisModel = Literal["structured", "generic_diagonal"]


def forward_transform(n_qubits: int, approximation_degree: int = 0) -> QuantumCircuit:
    """Forward DFT as a circuit: Qiskit's QFT carries the opposite sign."""
    return synth_qft_full(
        n_qubits, do_swaps=True, approximation_degree=approximation_degree, inverse=True
    )


def inverse_transform(n_qubits: int, approximation_degree: int = 0) -> QuantumCircuit:
    """Inverse DFT as a circuit (Qiskit's plain QFT)."""
    return synth_qft_full(
        n_qubits, do_swaps=True, approximation_degree=approximation_degree, inverse=False
    )


def _phase_block(
    expansion: BitExpansion,
    synthesis: SynthesisModel,
    name: str,
) -> QuantumCircuit:
    """Emit a diagonal phase either structurally or as a generic DiagonalGate."""
    if synthesis == "structured":
        return structured_phase_circuit(expansion, name=name)
    circuit = QuantumCircuit(expansion.n_qubits, name=name)
    circuit.append(DiagonalGate(list(evaluate_bit_expansion(expansion))), range(expansion.n_qubits))
    return circuit


# ------------------------------------------------------------ assembly -----


def build_propagation_core(
    boundary: str,
    n_grid: int,
    n_steps: int,
    potential_expansion: BitExpansion | None,
    kinetic_expansion: BitExpansion,
    half_potential_expansion: BitExpansion | None,
    synthesis: SynthesisModel = "structured",
    approximation_degree: int = 0,
) -> QuantumCircuit:
    """Assemble ``r`` Strang steps with adjacent potential phases merged.

    Contains no barriers and no state preparation or measurement, so the count
    it yields is the propagation core alone.
    """
    validate_power_of_two(n_grid)
    n_data = int(np.log2(n_grid))

    if boundary == "periodic":
        n_total = n_data
        data_qubits = list(range(n_data))
    else:
        n_data, n_total = extended_register_size(n_grid)
        data_qubits = list(range(1, n_data + 1))

    circuit = QuantumCircuit(n_total, name=f"propagation_r{n_steps}")

    def apply_potential(expansion: BitExpansion | None, label: str) -> None:
        if expansion is None:
            return
        circuit.compose(_phase_block(expansion, synthesis, label), qubits=data_qubits, inplace=True)

    def apply_kinetic() -> None:
        if boundary == "periodic":
            circuit.compose(forward_transform(n_total, approximation_degree), inplace=True)
            circuit.compose(_phase_block(kinetic_expansion, synthesis, "kinetic"), inplace=True)
            circuit.compose(inverse_transform(n_total, approximation_degree), inplace=True)
        else:
            extension = odd_extension_circuit(n_grid)
            circuit.compose(extension, inplace=True)
            circuit.compose(forward_transform(n_total, approximation_degree), inplace=True)
            circuit.compose(_phase_block(kinetic_expansion, synthesis, "kinetic"), inplace=True)
            circuit.compose(inverse_transform(n_total, approximation_degree), inplace=True)
            circuit.compose(extension.inverse(), inplace=True)

    apply_potential(half_potential_expansion, "V_half")
    for step in range(n_steps):
        apply_kinetic()
        if step < n_steps - 1:
            apply_potential(potential_expansion, "V_full")
    apply_potential(half_potential_expansion, "V_half")
    return circuit


# ------------------------------------------------------------ counting -----


@dataclass
class ResourceRow:
    """One fully-qualified row of the resource table."""

    benchmark: str
    boundary: str
    transform: str
    n_grid: int
    n_data_qubits: int
    n_ancilla_qubits: int
    n_total_qubits: int
    n_steps: int
    one_qubit_gates: int
    two_qubit_gates: int
    two_qubit_depth: int
    total_depth: int
    synthesis_model: str
    connectivity: str
    basis_gates: str
    optimisation_level: int
    approximation_degree: int
    seed: int
    includes_state_preparation: bool
    includes_measurement: bool
    scope: str
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def count_resources(
    circuit: QuantumCircuit,
    config: RunConfig,
    connectivity: str = "all_to_all",
) -> dict[str, int]:
    """Transpile and return one-/two-qubit counts, two-qubit depth and depth."""
    settings = config.circuits
    coupling = None
    if connectivity == "linear":
        coupling = CouplingMap.from_line(circuit.num_qubits)
    elif connectivity != "all_to_all":
        raise ValueError(f"Unknown connectivity {connectivity!r}.")

    transpiled = transpile(
        circuit,
        basis_gates=list(settings.basis_gates),
        coupling_map=coupling,
        optimization_level=settings.optimisation_level,
        seed_transpiler=settings.seed,
    )

    one_qubit = two_qubit = 0
    two_qubit_circuit = transpiled.copy_empty_like()
    for instruction in transpiled.data:
        width = instruction.operation.num_qubits
        if instruction.operation.name == "barrier":
            continue
        if width == 1:
            one_qubit += 1
        elif width == 2:
            two_qubit += 1
            two_qubit_circuit.append(instruction)
        else:
            raise RuntimeError(
                f"Transpiled circuit still contains a {width}-qubit gate "
                f"'{instruction.operation.name}'; counts would be meaningless."
            )

    return {
        "one_qubit_gates": one_qubit,
        "two_qubit_gates": two_qubit,
        "two_qubit_depth": int(two_qubit_circuit.depth()),
        "total_depth": int(transpiled.depth()),
    }


def propagation_resources(
    config: RunConfig,
    name: str,
    n_steps: int | None = None,
    n_grid: int | None = None,
    synthesis: SynthesisModel = "structured",
    connectivity: str = "all_to_all",
    approximation_degree: int = 0,
) -> ResourceRow:
    """Return the resource row for the propagation core of one benchmark."""
    benchmark = config.benchmark(name)
    physics = config.physics
    grid_size = benchmark.domain.n_grid if n_grid is None else n_grid
    steps = benchmark.time_grid.n_steps if n_steps is None else n_steps
    time_step = benchmark.time_grid.time_step(steps)
    length = benchmark.domain.length
    boundary = benchmark.domain.boundary

    n_data = int(np.log2(grid_size))
    if boundary == "periodic":
        spacing = length / grid_size
        kinetic = signed_momentum_expansion(grid_size, spacing, physics.mass, physics.hbar, time_step)
        full = harmonic_position_expansion(
            grid_size, benchmark.domain.x_left, spacing, physics.mass,
            benchmark.omega or 0.0, physics.hbar, time_step,
        )
        half = harmonic_position_expansion(
            grid_size, benchmark.domain.x_left, spacing, physics.mass,
            benchmark.omega or 0.0, physics.hbar, 0.5 * time_step,
        )
        n_ancilla, transform = 0, "QFT"
    else:
        kinetic = folded_sine_kinetic_expansion(
            grid_size, length, physics.mass, physics.hbar, time_step
        )
        if benchmark.tilt_force:
            full = linear_tilt_expansion(
                grid_size, length, benchmark.tilt_force, physics.hbar, time_step
            )
            half = linear_tilt_expansion(
                grid_size, length, benchmark.tilt_force, physics.hbar, 0.5 * time_step
            )
        else:
            full = half = None
        n_ancilla, transform = 2, "QST"

    circuit = build_propagation_core(
        boundary, grid_size, steps, full, kinetic, half, synthesis, approximation_degree
    )
    counts = count_resources(circuit, config, connectivity)

    return ResourceRow(
        benchmark=name,
        boundary=boundary,
        transform=transform,
        n_grid=grid_size,
        n_data_qubits=n_data,
        n_ancilla_qubits=n_ancilla,
        n_total_qubits=n_data + n_ancilla,
        n_steps=steps,
        synthesis_model=synthesis,
        connectivity=connectivity,
        basis_gates="+".join(config.circuits.basis_gates),
        optimisation_level=config.circuits.optimisation_level,
        approximation_degree=approximation_degree,
        seed=config.circuits.seed,
        includes_state_preparation=False,
        includes_measurement=False,
        scope="propagation_core_only",
        notes=(
            f"{steps} kinetic blocks; 1 half + {max(steps - 1, 0)} full + 1 half potential phases"
            if full is not None
            else f"{steps} kinetic blocks; zero interior potential so no potential phases"
        ),
        **counts,
    )


def scaling_table(
    config: RunConfig,
    name: str,
    grid_sizes: tuple[int, ...],
    n_steps: int = 1,
    synthesis: SynthesisModel = "structured",
    connectivity: str = "all_to_all",
) -> list[ResourceRow]:
    """Return resource rows across register sizes, not just one qubit count."""
    return [
        propagation_resources(config, name, n_steps, size, synthesis, connectivity)
        for size in grid_sizes
    ]


# -------------------------------------------------- approximate QFT --------


def measure_qft_approximation_error(n_qubits: int, approximation_degree: int) -> float:
    """Return the spectral-norm error of a truncated QFT against the exact one."""
    from qiskit.quantum_info import Operator

    exact = Operator(synth_qft_full(n_qubits, do_swaps=True)).data
    approximate = Operator(
        synth_qft_full(n_qubits, do_swaps=True, approximation_degree=approximation_degree)
    ).data
    return float(np.linalg.norm(exact - approximate, 2))


def approximate_qft_study(
    config: RunConfig,
    name: str,
    degrees: tuple[int, ...],
    n_steps: int = 1,
) -> list[dict[str, Any]]:
    """Trade-off between omitted small rotations, gate count and accuracy.

    Reports operator error alongside the gate saving, so a claim that the
    approximate QFT is worthwhile has to be supported by both numbers.
    """
    benchmark = config.benchmark(name)
    n_total = int(np.log2(benchmark.domain.n_grid))
    if benchmark.domain.boundary != "periodic":
        n_total += 2

    rows = []
    baseline = None
    for degree in degrees:
        row = propagation_resources(
            config, name, n_steps=n_steps, approximation_degree=degree
        )
        baseline = baseline or row.two_qubit_gates
        rows.append(
            {
                "approximation_degree": degree,
                "n_total_qubits": row.n_total_qubits,
                "two_qubit_gates": row.two_qubit_gates,
                "one_qubit_gates": row.one_qubit_gates,
                "total_depth": row.total_depth,
                "two_qubit_reduction": 1.0 - row.two_qubit_gates / baseline,
                "transform_operator_error": measure_qft_approximation_error(n_total, degree),
                "n_steps": n_steps,
            }
        )
    return rows

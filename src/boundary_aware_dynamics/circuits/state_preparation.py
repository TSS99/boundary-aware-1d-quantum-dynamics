"""Initial-state preparation, costed honestly and reported separately.

What this module does **not** claim
-----------------------------------
Amplitude encoding of an arbitrary ``N``-sample wavefunction into ``log2 N``
qubits is not free and is not known to be efficient in general.  Qiskit's
``StatePreparation`` performs exact preparation with a cost that grows like
``O(N) = O(2^n)`` two-qubit gates.  Any end-to-end speedup claim that quietly
assumes free state preparation is unsupported, so preparation cost is always
reported as its own row rather than folded into the propagation core.

For the specific states used in this project a cheaper route exists in
principle: a discretised Gaussian is a structured function and can be prepared
approximately by known techniques with cost polynomial in the number of qubits.
That is not implemented here, and no such saving is claimed anywhere in the
resource tables.  The Gaussian preparation cost is reported only as the exact
``StatePreparation`` upper bound, explicitly labelled as such.

Measurement
-----------
A position-density measurement is a computational-basis measurement of the data
register and costs no gates, but it yields samples of ``|psi(x_j)|^2`` and not
the complex amplitudes.  The statevector fidelities reported elsewhere in this
repository are **simulator diagnostics**: obtaining them on hardware would need
an overlap protocol (such as a swap test or a Hadamard test) or full state
tomography, with a shot cost that grows with the precision required.  Nothing in
this repository measures fidelity on hardware, and no hardware run is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

from ..states import physical_to_amplitudes

__all__ = [
    "PreparationCost",
    "exact_state_preparation_circuit",
    "measurement_note",
    "shot_budget_for_density",
]


@dataclass
class PreparationCost:
    """Cost of preparing the initial state, kept apart from propagation cost."""

    n_data_qubits: int
    n_grid: int
    one_qubit_gates: int
    two_qubit_gates: int
    total_depth: int
    method: str
    is_efficient: bool
    notes: str

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def exact_state_preparation_circuit(psi: np.ndarray, spacing: float) -> QuantumCircuit:
    """Return an exact amplitude-encoding circuit for physical samples ``psi``.

    The samples are converted to register amplitudes first: a quadrature-
    normalised wavefunction is not a legal statevector.
    """
    amplitudes = physical_to_amplitudes(psi, spacing)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    n_qubits = int(np.log2(amplitudes.size))
    if 2**n_qubits != amplitudes.size:
        raise ValueError(f"State length {amplitudes.size} is not a power of two.")

    circuit = QuantumCircuit(n_qubits, name="state_preparation")
    circuit.append(StatePreparation(amplitudes), range(n_qubits))
    return circuit


def measurement_note() -> dict[str, Any]:
    """Return the standing caveats about what measurement does and does not give."""
    return {
        "position_density": "computational-basis sampling of the data register; zero gate cost",
        "complex_amplitudes": "not directly observable",
        "statevector_fidelity": "simulator diagnostic only, not a hardware-measurable quantity",
        "hardware_fidelity_route": "overlap protocol (swap or Hadamard test) or full tomography",
        "shot_cost": "scales as 1/epsilon^2 for density estimation to precision epsilon",
        "hardware_runs_performed": False,
    }


def shot_budget_for_density(precision: float, confidence: float = 0.99) -> int:
    """Return a Hoeffding shot count for estimating a bin probability to ``precision``.

    Included so that measurement cost appears as a number rather than as an
    unquantified aside.
    """
    if not 0.0 < precision < 1.0:
        raise ValueError("precision must lie in (0, 1).")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    return int(np.ceil(np.log(2.0 / (1.0 - confidence)) / (2.0 * precision**2)))

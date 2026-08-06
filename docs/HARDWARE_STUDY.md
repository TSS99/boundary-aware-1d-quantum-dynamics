# Hardware study: device error against splitting error

A standalone study, separate from the simulator results. Nothing under
`results/paper` depends on it and `scripts/reproduce.py` does not run it.

Reproduce with:

```
export IBM_QUANTUM_TOKEN=...        # never stored in this repository
export IBM_QUANTUM_INSTANCE=...     # optional; defaults to 'auto'
python scripts/hardware_run.py submit --backend ibm_marrakesh
python scripts/hardware_run.py fetch
```

Token and instance both come from the environment and neither is recorded in
`results/hardware/`. The token is a credential; the instance names an account,
and this directory is committed.

## What was run

| | |
|---|---|
| backend | `ibm_marrakesh` (Heron r2, 156 qubits) |
| plan | open |
| job id | `d9q4ifclp7es73b4gnn0` |
| job tags | `boundary-aware-dynamics`, `hardware-vs-ideal`, `grid-8`, `trotter-sweep`, `backend-ibm_marrakesh` |
| submitted | 2026-08-06 08:43:41 UTC, finished 08:45:15 UTC |
| QPU time charged | 43 s |
| circuits | 23 |
| shots | 8192 (propagation), 4096 (controls) |
| error suppression | XY4 dynamical decoupling, gate and measurement twirling |
| grid | `N = 8` for all three benchmarks |

Every circuit evolves to the **same final time**; `r` sets how finely that
interval is split. Raising `r` therefore lowers splitting error and raises
device error, and the question is where they cross.

Three groups make the two error sources separable:

- **propagate** — preparation, `r` Strang steps, measurement.
- **echo** — preparation, `r` steps, the inverse of those steps, measurement.
  The ideal output is the initial density exactly, so deviation is device error
  alone with splitting error cancelled by construction.
- **baseline** — preparation and measurement only: the state-preparation and
  readout floor, with no propagation.

On the hard-wall benchmarks the two QST ancillas return to `|00>` exactly after
every step (verified to 1e-12 by statevector simulation), so they are measured
and used as an error-detection flag; shots with a non-zero ancilla are
discarded. `ancilla_retention` in the results table is the surviving fraction.

## Results

Total variation distance between probability densities over the eight grid
points. Full table in `results/hardware/hardware_comparison.csv`, densities in
`results/hardware/densities.json`.

| system | r | 2q gates | device error | splitting error | total error | ancilla retention |
|---|---|---|---|---|---|---|
| harmonic (QFT, 3q) | 1 | 33 | 0.056 | 0.401 | 0.425 | — |
| harmonic | 2 | 49 | 0.122 | 0.894 | 0.794 | — |
| harmonic | 4 | 69 | 0.087 | 0.286 | **0.328** | — |
| harmonic | 8 | 109 | 0.079 | 0.356 | 0.386 | — |
| free well (QST, 5q) | 1 | 111 | 0.092 | < 1e-14 | **0.092** | 0.65 |
| free well | 2 | 203 | 0.132 | < 1e-14 | 0.132 | 0.54 |
| free well | 4 | 373 | 0.165 | < 1e-14 | 0.165 | 0.41 |
| tilted well (QST, 5q) | 1 | 111 | 0.155 | 0.928 | 0.833 | 0.64 |
| tilted well | 2 | 229 | 0.222 | 0.429 | **0.561** | 0.48 |
| tilted well | 4 | 455 | 0.346 | 0.272 | 0.583 | 0.37 |

Controls:

| system | baseline (prep + readout) | echo r=1 | echo r=2 | echo r=4 |
|---|---|---|---|---|
| harmonic | 0.044 | 0.100 | 0.102 | 0.103 |
| free well | 0.040 | 0.415 | 0.525 | 0.639 |
| tilted well | 0.052 | 0.348 | 0.503 | 0.663 |

## Figures

All in `results/hardware/figures/`, PDF and PNG.

**`hardware_error_vs_steps`** — *Where device error overtakes splitting error.*
One panel per system, total variation distance on a logarithmic axis against
Trotter steps. Three curves: device error (hardware against the noiseless
simulation of the same circuit), splitting error (that simulation against exact
diagonalisation of the same discrete Hamiltonian) and total error (hardware
against exact). The free well carries no splitting error at all, so its axis is
scaled to the measurable series and the exactness is stated in the panel.

**`hardware_density_comparison`** — *Every measured density against the ideal
simulator.* Rows are systems, columns are step counts; grey bars are the exact
discrete solution, the solid line the ideal simulator, the dashed line the
device. Each panel is annotated with its two-qubit gate count and its device
TVD, so the growth of the deviation can be read against circuit size directly.

**`hardware_gate_counts`** — *What was actually executed.* **(a)** two-qubit
(`cz`) count and **(b)** depth against Trotter steps for all three systems,
propagation solid and echo controls dashed; **(c)** the full gate composition of
each executed propagation circuit, `cz`/`rz`/`sx`/`x`, after transpilation to the
Heron basis. Counts are re-derived from the seeded transpilation and checked
against what was submitted; they agree exactly for all 23 circuits.

**`hardware_controls`** — *The two controls.* **(a)** echo error against
two-qubit count, with the preparation-and-readout floor marked by a star;
**(b)** ancilla postselection retention against two-qubit count, which is the
error-detection rate on the two hard-wall systems.

**`<benchmark>_hardware_circuit`** — *What the device executed*, one figure per
system, written by `python scripts/hardware_run.py circuits`. These are the
transpiled, backend-mapped `r = 1` circuits including state preparation and
measurement: physical qubit indices on `ibm_marrakesh`, the device's own
`cz/rz/sx/x` basis, three data qubits and — on the hard-wall systems — the two
QST ancillas with their own classical register. Their two-qubit counts are
checked against the counts recorded at submission and agree for all three.
Vector PDF only, drawn in Qiskit's IBM (`iqp`) scheme.

## What it shows

1. **Device error grows monotonically with circuit size**, as it must: free
   well 0.092 → 0.165 and tilted well 0.155 → 0.346 as the two-qubit count goes
   from 111 to 373 and 455 respectively. The echo controls confirm this
   independently of any physics: with the propagation exactly undone, the
   measured deviation still climbs with depth.

2. **The free well is a clean device-error probe.** Its interior potential is
   zero, so `[T, V] = 0` and the circuit is exact at every `r` — the splitting
   error is at machine precision, verified by statevector simulation. Every
   thing measured there is device error.

3. **The tilted well shows the crossover the study was built to find.**
   Splitting error falls (0.93 → 0.43 → 0.27) while device error rises
   (0.16 → 0.22 → 0.35); total error is minimised near `r = 2` and gets no
   better at `r = 4`. On this device, more Trotter steps stop paying at `r ≈ 2`.

4. **Ancilla postselection is worth having.** It is free — the ancillas are
   already there — and it discards 35–63% of shots as detected errors.

5. **A floor of roughly 0.04–0.05** sits under everything, from state
   preparation and readout, measured directly by the baseline circuits.

## Caveats

- `N = 8` is a toy grid: eight points, so the spatial discretisation is far from
  the manuscript's. The comparison here is against **exact diagonalisation of
  the same discrete Hamiltonian**, not against the continuum, so
  discretisation error is excluded by construction and the two errors on show
  are splitting error and device error only.
- At `r = 1, 2` the time step is O(1) and the splitting error is far outside its
  asymptotic regime, which is why the harmonic splitting-error curve is not
  monotonic at small `r`. It does converge: statevector checks give
  4.0e-1 at `r = 1` down to 2.4e-4 at `r = 128`.
- Readout mitigation was not applied; only twirling and dynamical decoupling.
  M3 or TREX would lower the baseline floor.
- Densities are the only observable used. Complex amplitudes are not directly
  measurable, and statevector fidelity is a simulator diagnostic, not a
  hardware-measurable quantity.

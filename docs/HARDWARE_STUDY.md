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

**`hardware_zne_density_grid`** — *Every benchmark, raw and mitigated.* Rows are
systems, columns are step counts, matching `hardware_density_comparison`. Grey bars
are the exact discrete solution, blue the ideal simulator, orange dashed the raw
hardware, faint orange the noise-amplified runs and green the extrapolation. The
dotted line is the uniform distribution, which is where depolarising noise drives
everything — the faint curves should approach it in order, and where they do not the
extrapolation has nothing to work with. Each panel carries its λ range and both
distances.

**`hardware_zne_densities`** — the four-panel pilot that preceded the grid, with
global folding at λ = 1, 3, 5. Kept because its free-well panel is the clearest
illustration of a saturated extrapolation failing visibly.

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

## Error mitigation

Two techniques were tried after the main campaign, in increasing order of cost.
Both are reproducible from `scripts/hardware_run.py`.

### Readout correction — tried, and mostly does not help

`python scripts/hardware_run.py mitigate`, **0 QPU seconds**: the raw outcomes are
re-fetched from the completed job and corrected in post-processing with a tensored
assignment matrix built from the backend's per-qubit readout probabilities. The
correction is applied on the joint register and only then is the ancilla
postselection applied, so that a misread ancilla cannot discard a good shot before
the correction can repair it.

The prediction going in was that most of the 0.040–0.052 baseline floor was readout,
and that correction would take it to about 0.015. **That was wrong.**

| | raw | corrected | gain |
|---|---|---|---|
| baselines (mean) | 0.0452 | 0.0386 | 0.007 |
| best case anywhere | — | — | 0.018 |
| typical | — | — | 0.004–0.010 |

Readout is roughly **15% of the floor**, not the bulk of it. Working back from the
calibration, the baseline circuits carry ~1.5% gate error from five two-qubit gates
and ~1.5–3% readout error, which sums to the observed ~4.5%. Ancilla retention
barely moved (0.6453 → 0.6423), so misread ancillas were not costing good shots
either. The result is recorded because it is the kind of obvious first move that
looks like it should work, and knowing it does not saves the next person the effort.

### Zero-noise extrapolation — works, and moves the crossover

Qiskit Runtime offers ZNE only through the Estimator, which returns expectation
values and therefore cannot postselect the ancillas. Since postselection discards a
third to two thirds of shots as detected errors, the folding is done explicitly
instead and the extrapolation is applied per density bin after postselection.

**Noise amplification.** Individual `cz` gates are repeated in place. `cz` is its own
inverse, so a fold is three copies where there was one and no basis inverse has to be
constructed; folding the two-qubit gates alone is also what makes non-integer noise
factors reachable. That matters, because global folding triples the circuit at its
smallest step, which pushes the deeper benchmarks past the point where the returned
distribution is indistinguishable from uniform. Noise factors are therefore chosen
per circuit so that the most amplified copy stays under 600 two-qubit gates:

| benchmark | r | 2q at λ=1 | λ range | 2q at λ_max |
|---|---|---|---|---|
| harmonic | 1, 2, 4, 8 | 33–111 | 1 → 5.00 | 165–555 |
| free well | 1 | 111 | 1 → 5.00 | 555 |
| free well | 2 | 203 | 1 → 2.95 | 599 |
| free well | 4 | 373 | 1 → 1.61 | 599 |
| tilted well | 1 | 111 | 1 → 5.00 | 555 |
| tilted well | 2 | 229 | 1 → 2.62 | 601 |
| tilted well | 4 | 455 | 1 → 1.32 | 599 |

**The estimator matters more than the folding.** Fitting each density bin separately
to an exponential needs two free parameters per bin against three noise factors, and
where the lever arm is short — the deepest circuits reach only λ = 1.32 — those fits
diverge. On this data they returned distances of 0.776 and 0.834 where the raw
measurements were 0.142 and 0.184: confident nonsense. Depolarising noise acts on the
whole distribution at once, so the faithful model has a single parameter: the
deviation from uniform decays as `exp(-Gamma * lambda)` with its shape preserved.
One constant fitted jointly across every bin and every noise factor is stable where
per-bin fitting is not, and it turned four broken panels into four working ones from
the same data.

**Results**, `python scripts/hardware_run.py zne-grid` then `zne-grid-fetch`,
**79 QPU seconds** for 30 circuits (10 benchmarks × 3 noise factors, 8192 shots):

| benchmark | r | raw | mitigated | gain | amplification |
|---|---|---|---|---|---|
| harmonic | 1 | 0.052 | 0.030 | 0.021 | 1.10 |
| harmonic | 2 | 0.128 | 0.069 | 0.059 | 1.15 |
| harmonic | 4 | 0.094 | 0.041 | 0.052 | 1.25 |
| harmonic | 8 | 0.083 | 0.043 | 0.040 | 1.25 |
| free well | 1 | 0.108 | 0.076 | 0.032 | 1.34 |
| free well | 2 | 0.142 | 0.065 | 0.077 | 2.65 |
| free well | 4 | 0.184 | 0.157 | 0.028 | 2.50 |
| tilted well | 1 | 0.171 | 0.048 | 0.123 | 1.59 |
| tilted well | 2 | 0.233 | 0.032 | 0.201 | 2.37 |
| tilted well | 4 | 0.367 | 0.150 | 0.217 | 4.00 (capped) |

Every benchmark improved. Clipped negative mass is below 0.02 everywhere, so no
result rests on repairing an unphysical extrapolation. The one qualification is
tilted well `r = 4`, where the fit reached the amplification cap of 4.0: it wanted to
extrapolate further than the data supports, so its 0.150 is a lower bound on the
correction rather than a measurement.

**The crossover moves.** Against exact diagonalisation of the same discrete
Hamiltonian, for the tilted well:

| | r=1 | r=2 | r=4 |
|---|---|---|---|
| raw hardware | 0.814 | 0.574 | 0.598 |
| mitigated | 0.897 | 0.424 | 0.322 |
| splitting error alone | 0.928 | 0.429 | 0.272 |

Raw hardware bottoms out at `r = 2` and gets worse at `r = 4`. Mitigated, the error
keeps falling through `r = 4` and lies almost on the splitting-error curve, meaning
device error has been suppressed below the Trotter error. **Mitigation buys Trotter
steps** — the optimum moves past `r = 4`, and the limit on how finely the interval
may be split becomes the algorithm's rather than the device's. This is the sharpest
result of the hardware study, and it is only visible because the same grid was
measured raw and mitigated.

### Cost of the whole campaign

| job | purpose | QPU seconds |
|---|---|---|
| `d9q4ifclp7es73b4gnn0` | main campaign, 23 circuits | 43 |
| `d9qo0t8pdb6s73e3qnsg` | ZNE pilot, global folding, 12 circuits | 32 |
| `d9qou20pdb6s73e3sj50` | ZNE grid, partial folding, 30 circuits | 79 |
| — | readout correction | 0 |
| | **total** | **154** |

Wall clock ran 94 s, 201 s and roughly 400 s respectively: QPU time is what is
billed, and it is a fraction of the time spent waiting. Queue waits were 1.8–19 s
because the backend was quiet.

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
- The main campaign used only twirling and dynamical decoupling. Readout correction
  and zero-noise extrapolation were applied afterwards and are reported separately
  above; the headline numbers in the tables at the top of this document are the
  unmitigated ones.
- The device drifted measurably between jobs. The same free-well `r = 1` circuit
  returned 0.092, 0.131 and 0.108 across three jobs on two days, and ancilla
  retention moved 0.645 → 0.583 → 0.589. Every comparison in the mitigation section
  is therefore within a single job, which is why the raw baseline is re-measured
  alongside each mitigated run rather than taken from the main campaign.
- Densities are the only observable used. Complex amplitudes are not directly
  measurable, and statevector fidelity is a simulator diagnostic, not a
  hardware-measurable quantity.

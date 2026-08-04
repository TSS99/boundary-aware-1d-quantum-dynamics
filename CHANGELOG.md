# Changelog

## 0.1.0 — unreleased

First version organised around the boundary-condition argument. Not released: see
`docs/LOCAL_RELEASE_CHECKLIST.md`.

### Added

- `src/boundary_aware_dynamics/`: reusable package replacing code that previously
  lived only inside notebooks. Typed YAML configuration is now the single source
  of truth for every parameter.
- **Validated quantum sine transform circuit.** Odd extension on a `4N`-point
  register with two unitarily uncomputed ancillas, verified against the
  analytical DST-II propagator to better than `1e-10`.
- **Direct periodic-versus-Dirichlet experiment** against a finite-difference
  reference sharing neither basis nor method with either propagator.
- **Benchmark C, the tilted infinite well** `V(x) = F(x - L/2)`: the first
  hard-wall benchmark here with `[T, V] != 0`, so Trotter convergence can be
  measured under Dirichlet boundaries.
- **Structured phase synthesis** for the harmonic, signed-momentum, folded-sine
  and linear-tilt diagonals: `O(n^2)` gates instead of `O(2^n)`.
- Exact diagonalisation of the *discrete* Hamiltonian as a reference, isolating
  Trotter error from spatial error.
- Provenance recording and provenance-aware staleness detection.
- 300+ tests covering conventions, physics invariants, circuits, convergence
  orders and resource schemas.
- `scripts/reproduce.py`, `scripts/execute_notebooks.py`,
  `scripts/verify_results.py`; six independent notebooks; a `docs/` set.

### Fixed

- The circuit previously labelled "QST" was a relabelled `QFTGate` on an extended
  register, with no odd extension, no midpoint-grid phases and no ancilla
  handling. Its disagreement with the true DST-II was as large as the transform
  itself. Replaced with a validated construction.
- Resource tables reported `n_qubits = 6` for a circuit occupying 8 wires.
  Ancillas are now counted and reported separately.
- Total resources were computed as `single_step × r`, ignoring that adjacent
  half-potential phases merge. Composition is now one half-phase, `r-1` full
  phases, one half-phase and `r` kinetic blocks.
- Barriers in counting circuits blocked transpiler optimisation and inflated the
  counts. Counting circuits now contain none.
- Fidelity axes extended above 1.0, a region no data point can occupy. Accuracy
  is now plotted as infidelity or state error on logarithmic axes.
- `00_environment_setup.ipynb` hard-coded an absolute path and raised on
  mismatch, so it could not run in this clone at all. Removed.
- Notebooks shared state through `%run`. Each notebook is now independent.
- `requirements.txt` was a 108-line full environment freeze including
  Windows-only packages. Replaced by `pyproject.toml` and `environment.yml`.
- Registering a named Jupyter kernel and setting `JUPYTER_PATH` are no longer
  required.

### Changed

- The zero-potential infinite well is now presented as a **control**, not as a
  Trotter benchmark: its splitting is exact, so its step sweep measures nothing
  about the time integrator. Its near-unity fidelity is documented as structural,
  arising because the DST-II rows are the reference eigenmodes.
- Convergence slopes are reported with fit interval, point count and `R²`, and
  exclude the pre-asymptotic first point.
- Tilted-well parameters (`F = 5`, `t_max = 2`) were chosen by measurement so the
  sweep sits in the asymptotic regime.
- Figures: sans-serif, no embedded titles or captions, validated colour-blind-safe
  palette with distinct dashes and markers.

### Removed

- Six superseded notebooks, and the `figures/` and `tables/` directories they
  wrote to. Generated output now lives under `results/<profile>/`. Earlier
  versions remain in git history.

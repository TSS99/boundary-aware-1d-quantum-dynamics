# Boundary-aware spectral propagation for 1D quantum dynamics

Companion code for a manuscript in preparation for *The European Physical Journal
Plus*.

## Scientific purpose

The spectral transform inside a split-operator step is part of the physical
model, not an implementation detail. A discrete Fourier transform diagonalises the
Laplacian on a **ring**; a discrete sine transform diagonalises it on a **box with
hard walls**. Choosing between them chooses which system is being simulated.

This repository measures how much that choice matters, validates the corresponding
quantum circuits against analytical targets, and reports their cost with the
assumptions attached.

## Contribution

Modest and specific. Split-operator propagation, the QFT and quantum sine
transforms are all established; none is introduced here. What this repository
adds is:

1. A **validated quantum sine transform circuit** — an explicit odd extension on a
   `4N`-point register with two unitarily uncomputed ancillas, agreeing with the
   analytical DST-II propagator to better than `1e-10`.
2. A **direct quantitative periodic-versus-Dirichlet comparison** of the same
   hard-wall problem against a reference sharing neither basis nor method with
   either propagator.
3. A **hard-wall benchmark with a non-zero interior potential**, so that Trotter
   convergence can actually be measured under Dirichlet boundaries.
4. **Resource accounting with its assumptions attached** — ancillas counted,
   step composition merged correctly, structured synthesis separated from a
   generic upper bound, connectivity and transpiler settings recorded per row.

See [`docs/PRIOR_WORK.md`](docs/PRIOR_WORK.md) for what is deliberately *not*
claimed.

## The key result

Propagating the *same* hard-wall wavepacket with the two transforms — identical
state, box, resolution, interval and step count — and comparing both against an
independent finite-difference hard-wall reference:

| | Dirichlet (DST-II) | Periodic (FFT) |
|---|---|---|
| Final infidelity vs reference | `~1e-5` | order unity |
| Maximum wall residual | small | several times larger |
| Divergence between the two | reaches ~1 (nearly orthogonal) | |

Increasing the grid size does not rescue the periodic model: the defect is
topological, not a resolution deficit.

## Benchmarks

| | System | Boundary | Purpose |
|---|---|---|---|
| **A** | Harmonic oscillator in a large periodic box | periodic | Validate Strang splitting and the FFT/QFT convention; measure Trotter and spatial convergence |
| **B** | Infinite square well, `V = 0` | Dirichlet | **Control.** The sine propagator is exact here, so this measures boundary topology, not integrator accuracy. Hosts the boundary comparison |
| **C** | Tilted well, `V(x) = F(x - L/2)` | Dirichlet | The genuine Trotter benchmark: `[T, V] != 0` under hard walls |

Benchmark B is deliberately *not* a Trotter benchmark. With zero interior
potential the Dirichlet splitting is exact, so a step-count sweep there is flat at
round-off and measures nothing about the time integrator.

## Classical versus quantum-circuit work

Kept explicitly separate throughout:

| Layer | Tool | Where |
|---|---|---|
| Spectral propagation | NumPy FFT, SciPy DST-II | `propagators.py` |
| Reference solutions | dense linear algebra | `references.py` |
| Circuit construction | Qiskit | `circuits/` |
| Circuit validation | Qiskit `Operator` / `Statevector` | `tests/`, notebook 04 |
| Resource estimation | Qiskit transpiler, logical counts | `circuits/resources.py` |
| Hardware execution | **none performed, none claimed** | — |

## Workflow

```
configs/*.yaml                     single source of truth for every parameter
        |
        v
src/boundary_aware_dynamics/       grids -> states -> transforms -> propagators
        |                          references -> diagnostics -> workflows
        |                          circuits/ (qft, qst, phases, resources)
        v
notebooks/00..05                   independent; each imports from src
        |
        v
scripts/reproduce.py               figures, tables, metadata, provenance
        |
        v
results/<profile>/                 figures/ tables/ metadata/ executed_notebooks/
```

## Repository layout

```
configs/        paper.yaml, smoke.yaml
src/boundary_aware_dynamics/
                config, grids, states, transforms, references,
                propagators, diagnostics, plotting, provenance, workflows
                circuits/  qft, qst, phases, state_preparation, resources
notebooks/      00 method & transform validation
                01 harmonic oscillator          (Benchmark A)
                02 infinite well & boundary      (Benchmark B)
                03 tilted infinite well          (Benchmark C)
                04 circuit validation & resources
                05 publication exports
scripts/        reproduce.py, execute_notebooks.py, verify_results.py
tests/          300+ unit, physics, circuit, convergence and resource tests
docs/           method, reproducibility, circuit assumptions, error budget,
                figure captions, manuscript alignment, prior work, release checklist
references/     references.bib  (entries pending verification)
results/        generated; not source
```

## Installation

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Python 3.11+. No Jupyter kernel registration is required and no environment
variable needs to be set; the clone may live at any path.

## Reproduction

```bash
pytest                                        # ~15 s
python scripts/reproduce.py --profile smoke   # ~30 s, pipeline check
python scripts/reproduce.py --profile paper   # ~5-10 min, manuscript numbers
python scripts/execute_notebooks.py           # fresh-kernel notebook run
python scripts/verify_results.py --full       # tests, schemas, provenance, notebooks
```

Peak memory stays under 2 GB; no GPU is used.

## Key outputs

Written to `results/<profile>/`:

- `figures/` — PDF (vector) and PNG, sans-serif, colour-blind-safe with distinct
  dashes and markers so they survive greyscale printing
- `tables/` — parameters, errors, observables, boundary comparison, convergence,
  resources, approximate-QFT trade-off
- `metadata/` — `provenance.json`, `key_results.json`, `figure_manifest.csv`,
  `config.json`, and `paper_values.tex` (LaTeX macros, so manuscript numbers are
  never typed by hand)
- `executed_notebooks/` — notebooks with outputs, tagged with profile and config
  hash

## Reproducibility model

Configuration is the single source of truth and carries a hash. Every run records
the commit, working-tree cleanliness, Python and dependency versions, the config
hash, a hash over the package source, the seeds, and hashes of every output. A
result is *stale* when any of these has moved — not merely when a file is missing.
Results from a dirty working tree are written but labelled as such.

Full detail: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Known limitations

- Circuit validation at the unitary level is limited to 2–3 data qubits, where the
  full operator can be formed.
- Resource counts are **logical**: no error correction, no fault-tolerant
  overhead, no physical-qubit estimate.
- State preparation is exact amplitude encoding, exponential in the qubit count.
  It is costed separately and **no efficient-preparation claim is made**.
- Quoted fidelities are simulator diagnostics. Measuring them on hardware needs an
  overlap protocol or tomography, at `1/eps^2` shot cost.
- No noise model, and no hardware run.
- Only the specific quadratic and linear diagonals arising here are synthesised
  structurally; arbitrary potentials are not addressed.
- Bibliography entries are **unverified** — see below.

## Citation

`CITATION.cff` contains only what could be verified from the repository itself.
Author list, affiliations, ORCID identifiers, funding and publication metadata are
**absent and must be supplied by their owners**; they have deliberately not been
guessed.

## Licence

Released under the [MIT Licence](LICENSE), © 2026 Tilock Sadhukhan.

## Data and code availability

The repository is currently **private** and has not been archived or assigned a
DOI. It is intended to be made public on acceptance of the accompanying
manuscript, at which point an archive deposit (and its DOI) should be added here
and to `CITATION.cff`.

Every outstanding manual item is tracked in
[`docs/LOCAL_RELEASE_CHECKLIST.md`](docs/LOCAL_RELEASE_CHECKLIST.md).

Bibliographic metadata in `references/references.bib` could not be verified
offline. Entries marked `VERIFY` must be checked against the published record
before submission.

## Contact

Repository author, from the local git configuration: Tilock Sadhukhan
(`tilock.2025@gmail.com`). Corresponding-author designation for the manuscript is
still to be decided.

# Reproducibility

Everything below runs from a clean clone at any path, on Windows, macOS or Linux.
Nothing contacts a network at run time.

## Clean environment

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Python 3.11 or newer is required.

No Jupyter kernel needs to be registered. `scripts/execute_notebooks.py` runs the
notebooks with the interpreter that launched it, so there is no named kernel and
no `JUPYTER_PATH` to set. Notebooks locate the project root from their own working
directory, so the clone may live at any path.

## Running

```bash
pytest                                          # ~15 s
python scripts/reproduce.py --profile smoke     # ~30 s
python scripts/reproduce.py --profile paper     # several minutes
python scripts/execute_notebooks.py             # ~60 s (smoke)
python scripts/verify_results.py --full         # tests + schemas + provenance + notebooks
```

`--profile smoke` is for checking that the pipeline works. Manuscript numbers come
from `--profile paper`.

## Runtime and memory

| Step | Smoke | Paper |
|---|---|---|
| `pytest` | ~15 s | same |
| `reproduce.py` | ~30 s | ~5-10 min |
| `execute_notebooks.py` | ~60 s | ~5 min |
| Peak memory | < 1 GB | < 2 GB |

The dominant costs are the grid sweeps and the transpilation of multi-step
circuits. No GPU is used and no calculation exceeds a few hundred megabytes:
the largest object is a `4096 x 4096` dense Hamiltonian in the finite-difference
reference.

## Configuration is the single source of truth

`configs/paper.yaml` and `configs/smoke.yaml` hold every physical and numerical
parameter. Notebooks and scripts load a profile; nothing redefines a parameter
inline, so two notebooks cannot disagree about a value.

Each config carries a 16-character `config_hash` derived from its canonical
serialisation. Changing any parameter changes the hash and marks existing results
stale.

## Provenance and staleness

Every reproduction writes `results/<profile>/metadata/provenance.json` recording:

- git commit and whether the working tree was dirty
- Python version, platform, and versions of numpy / scipy / qiskit / matplotlib /
  pandas / PyYAML
- configuration hash and a hash over every `.py` file in the package
- random seeds
- a hash of every generated figure and table
- runtime and generation timestamp, kept in a separate `volatile` section so that
  identical work produces identical deterministic fields

A result is **stale** when the configuration hash, the source hash, the dependency
versions or the seeds differ from the current state. `verify_results.py` reports
*which* of these changed, rather than a bare boolean.

Results produced from a dirty working tree are still written — that is the normal
state during development — but are labelled, so they cannot be mistaken for a
reproducible artefact.

## Determinism

Given the same profile, the same package source and the same dependency versions,
outputs are byte-identical apart from the `volatile` provenance section. The
transpiler seed is fixed and recorded on every resource row; there is no other
source of randomness in the pipeline (the seeded RNGs in the test suite do not
affect generated results).

## No hosted CI

Verification is a local script by design, so it runs from any clone with no
account, no network and no service. `verify_results.py --full` performs the checks
a CI job would: tests, result schemas, figure manifest completeness, local link
checking, provenance and staleness, and fresh-kernel notebook execution.

## Notebook output policy

Source notebooks under `notebooks/` are kept without stored outputs, so diffs stay
readable and no stale result is carried in the repository. Executed copies, with
outputs, are written to `results/<profile>/executed_notebooks/` and are tagged with
the profile and config hash they were run under. There is exactly one source per
notebook; no parallel text representation is maintained.

# Contributing

## Before changing anything

```bash
python -m pip install -e ".[dev]"
pytest
```

## The rules that matter here

**Configuration is the single source of truth.** Physical and numerical
parameters live in `configs/*.yaml`. Do not hard-code a parameter in a notebook or
a script; two copies of a number will eventually disagree.

**Notebooks import, they do not define.** Reusable code belongs in
`src/boundary_aware_dynamics/`. No notebook may run another or depend on state one
left behind; `scripts/execute_notebooks.py` runs each in a fresh kernel and will
catch it.

**Validate against closed forms, not against the library.** A transform checked
only against another call to the same library will pass even when the convention
is wrong. Every transform here is checked against an analytical expression.

**Choose the reference deliberately.** Measuring a convergence *order* requires
exact diagonalisation of the same discrete Hamiltonian, so spatial error cancels.
Measuring *total* accuracy requires a continuum reference. Conflating the two
produces a spurious floor.

**Report a slope with its window.** Fit interval, point count and `R²` accompany
every fitted slope, and pre-asymptotic points are excluded explicitly.

**Never plot fidelity.** It is bounded above by one and compresses the interesting
range; an axis drawn above one shows a region no data point can occupy. Use
infidelity or state error on a logarithmic axis.

**Resource numbers carry their assumptions.** Ancillas, synthesis model,
connectivity, basis gates, optimisation level and seed belong in the table row,
not in prose.

**Claims point at evidence.** If a statement cannot be traced to a specific test,
figure or table, it does not belong in the documentation.

## Adding a benchmark

1. Add it to both `configs/paper.yaml` and `configs/smoke.yaml`.
2. Give it a reference that shares neither basis nor method with the propagator.
3. Add physics tests: norm conservation, time reversal, an eigenstate check.
4. Add a convergence test with an expected order and a stated fit window.
5. Extend `scripts/reproduce.py` and the figure manifest.

## Before proposing a change

```bash
python scripts/verify_results.py --full
```

Do not commit generated output under `results/`, and do not commit notebooks with
stored outputs — `nbstripout` handles the latter if pre-commit is installed.

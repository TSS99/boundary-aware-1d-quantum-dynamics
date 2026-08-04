# Error budget

Every error source, how it is measured, and which experiment isolates it.

## Sources

| # | Source | Isolated by | Typical size (paper profile) |
|---|---|---|---|
| 1 | Reference truncation | basis-size sweep, tail weight reported | < 1e-9 |
| 2 | Box truncation (periodic only) | edge probability | < 1e-6 for Benchmark A |
| 3 | Spatial discretisation | grid sweep at fixed `r` | benchmark dependent |
| 4 | Transform convention | closed-form validation | machine precision |
| 5 | Trotter splitting | comparison against exact diagonalisation of the **discrete** Hamiltonian | `O(dt^2)` in state error |
| 6 | Circuit synthesis | operator comparison at small `n` | < 1e-10 |
| 7 | Approximate transform | approximation-degree sweep | swept and reported |
| 8 | State preparation | exact here; cost reported, error zero | 0 |
| 9 | Measurement sampling | shot budget, `1/eps^2` | not incurred (simulator) |
| 10 | Hardware noise | not modelled | n/a |

## The separation that matters

Sources 3 and 5 are routinely conflated. They are separated here by choosing the
reference deliberately:

- Against a **continuum** reference (Hermite, sine-Galerkin, finite-difference)
  the measured error is **total** — spatial plus temporal. This is the honest
  number for "how accurate is the simulation", and it saturates once one term
  dominates.
- Against **exact diagonalisation of the same discrete Hamiltonian** the spatial
  discretisation, basis truncation and pseudospectral aliasing all cancel
  identically, leaving only the splitting error. This is the only comparison from
  which an *order of convergence* can legitimately be read.

Reporting a convergence slope measured against a continuum reference conflates the
two and will show a spurious floor.

## Pre-asymptotic behaviour

The Strang error is `O(dt^2)` asymptotically, but there is a pre-asymptotic window
where subleading terms are not yet negligible and successive error ratios depart
from 4. For the tilted well at `F = 5`, `t_max = 2`, the transient covers roughly
`r < 80`; from `r = 80` the measured ratios are 4.05, 4.01, 4.00, 4.00.

Consequently:

- The step sweeps in `configs/paper.yaml` are chosen to sit in the asymptotic
  regime.
- The first point is excluded from every slope fit, and `fit_from_index`,
  `fit_interval_dt`, `n_points` and `r_squared` are reported alongside every slope.
- Convergence figures shade the fitted interval, so the excluded point is visible.

## Numerical floors

Two distinct floors appear, and they are labelled differently:

- **Round-off floor**, around 2e-12 in infidelity after a few thousand sequential
  transform pairs. Reached by the tilted well at very large `r`.
- **Structural exactness**, around 1e-14. The zero-potential Dirichlet propagator
  has *no* Trotter error at any step count, so its sweep is flat at the round-off
  level. No slope is fitted; the study reports `slope = nan` with an explicit note
  rather than fitting a number to numerical noise.

A floor line is drawn on a convergence figure only when the data is within two
decades of it. Otherwise the axis stretches over empty decades and a real trend
looks flat.

## Per-benchmark accounting

**A, harmonic (periodic).** Reference tail weight below 1e-9; edge probability
below 1e-6, so box truncation is negligible and the periodic model is appropriate
here. Trotter slope 2.00 in state error, near 4 in infidelity. The energy error is
a bounded `O(dt^2)` excursion, not a secular drift.

**B, infinite well (Dirichlet, `V = 0`).** The splitting error is identically
zero, so sources 3 and 5 vanish; the residual is reference and round-off only. The
comparison this benchmark supports uses a **finite-difference** reference that
shares neither basis nor method with either propagator. A sine-series reference
would be circular here, because the DST-II rows are precisely the sine eigenmodes
it would use.

**C, tilted well (Dirichlet, `V != 0`).** The only benchmark carrying both hard
walls and a genuine splitting error. The reference is converged in basis size
before use, and the tilt matrix elements are available in closed form so no
quadrature error enters. Trotter slope 2.00 with `R^2 = 1.0000` over four decades.

## What the budget does not cover

Nothing here bounds the error of a *hardware* execution, because there is none.
Sources 9 and 10 are listed for completeness and are explicitly not incurred.

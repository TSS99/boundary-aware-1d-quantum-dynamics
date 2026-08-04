# Circuit assumptions

Every resource number in this repository is conditional on the assumptions below.
A gate count quoted without them is not comparable to anything, so each is
recorded as a column in `results/<profile>/tables/resource_*.csv` rather than left
in prose.

## Gate set and transpilation

| Assumption | Value | Where recorded |
|---|---|---|
| Basis gates | `rz, sx, x, cx` | `basis_gates` column |
| Optimisation level | 3 | `optimisation_level` column |
| Transpiler seed | 20240517 | `seed` column |
| Approximation degree | 0 unless swept | `approximation_degree` column |

Counts are **logical**. There is no error correction, no magic-state distillation,
no fault-tolerant overhead and no physical-qubit estimate anywhere in this
repository.

## Endianness and transform convention

Little-endian: qubit `q` carries bit `2^q`. Qiskit's `QFTGate` implements the
`+2 pi i` convention, so the forward DFT is `QFTGate(n).inverse()`. `QFTGate`
includes the terminating swap network, so input and output share an ordering and
no manual bit reversal is inserted.

## Ancillas

| Transform | Data qubits | Ancillas | Total |
|---|---|---|---|
| QFT (periodic) | `n_q` | 0 | `n_q` |
| QST (Dirichlet) | `n_q` | 2 | `n_q + 2` |

The two QST ancillas carry the half-cell interleave and the odd reflection. They
are prepared and uncomputed by a unitary circuit (`X`, `H`, CNOTs) and its
adjoint — no measurement, no reset, and verified to return to `|0>` with leakage
below 1e-10.

Reporting `n_q` as the total qubit count for a Dirichlet circuit is wrong, and an
earlier version of this repository did exactly that.

## Connectivity

Two models are reported side by side:

- **all-to-all** — logical lower bound, no routing.
- **linear nearest-neighbour** — a line coupling map, which costs roughly 2.5x the
  all-to-all two-qubit count at these sizes.

No device-specific coupling map is used, because no device is targeted.

## Barriers

Counting circuits contain **no barriers**. A barrier blocks the transpiler from
cancelling gates across a block boundary and inflates the count. Barriers appear
only in display circuits, which are never counted.

## Step composition

`r` Strang steps are not `r` copies of the five-block single step. Adjacent
half-potential phases merge exactly, so the sequence is:

- 1 initial half-potential phase
- `r - 1` full potential phases
- 1 final half-potential phase
- `r` kinetic blocks

## Synthesis model

Two are reported:

- **structured** — the quadratic bit expansion, `O(n^2)` gates. This is the model
  the resource claims rest on.
- **generic_diagonal** — Qiskit's `DiagonalGate`, `O(2^n)`. Retained only as an
  explicitly labelled upper bound.

## State preparation

Reported as its own row, never folded into the propagation core. The
implementation is exact amplitude encoding via Qiskit's `StatePreparation`, whose
cost is exponential in the qubit count. **No efficient state-preparation claim is
made anywhere in this repository.** A structured Gaussian could in principle be
prepared more cheaply; that is not implemented and no such saving is counted.

## Measurement

Computational-basis measurement of the data register costs no gates and yields
samples of `|psi(x_j)|^2`. It does **not** yield complex amplitudes.

Every fidelity in this repository is a simulator diagnostic. Measuring it on
hardware would need a swap test, a Hadamard test or full tomography, with a shot
budget scaling as `1/epsilon^2` (about 26,500 shots for 1% precision at 99%
confidence, by Hoeffding).

## Simulator limitations

Unitary-level validation is performed at 2-3 data qubits (4-5 total for the QST),
where the full operator can be formed. Larger registers are covered by resource
counting and by statevector checks, not by operator comparison.

## What is out of scope

- Hardware execution. None performed, none claimed.
- Noise. No noise model is used; a noise study would come after all ideal
  algorithmic tests, and none is included in this version.
- Fault-tolerant resource estimation.
- Arbitrary-potential diagonal synthesis. Only the specific quadratic and linear
  diagonals used here are synthesised structurally.

# Prior work

## Position of this repository

Split-operator (Strang) propagation with Fourier transforms is a standard,
decades-old numerical method. Its quantum-circuit analogue, replacing the FFT with
a QFT, is likewise established in the quantum-simulation literature. Neither is a
contribution of this work.

Quantum sine transforms and their use for Dirichlet problems are also known; in
particular there is 2024 work applying a quantum discrete sine transform to the
infinite square well. **This repository does not claim to introduce the quantum
sine transform, a new QFT, or a new product formula.**

## What is claimed here

1. **A validated QST circuit.** The DST-II is realised through an explicit odd
   extension on a `4N`-point register with two ancillas that are uncomputed
   unitarily, and its action is verified against the analytical DST-II matrix
   (below 1e-10 at 2-3 data qubits). The mode-index fold is handled by a
   triangle-wave index map that still closes at quadratic order in the register
   bits, so the kinetic phase stays `O(n^2)`.

2. **A direct, quantitative boundary comparison.** Periodic and Dirichlet
   propagation of the *same* hard-wall problem — same state, box, resolution,
   interval and step count — measured against a reference sharing neither basis
   nor method with either. The claim that the transform choice is physical is
   demonstrated rather than asserted.

3. **A hard-wall benchmark with a non-zero interior potential.** The tilted well
   supplies genuine `[T, V] != 0` under Dirichlet boundaries, so second-order
   convergence can actually be measured — which it cannot be in a zero-potential
   well, where the splitting is exact.

4. **Resource accounting with its assumptions attached.** Ancillas counted,
   composition merged correctly, structured synthesis separated from a generic
   diagonal upper bound, and connectivity and transpiler settings recorded on
   every row.

## What is explicitly not claimed

- End-to-end quantum advantage, or exponential speedup for arbitrary potentials.
- Efficient arbitrary state preparation. The implementation here is exact
  amplitude encoding, exponential in the qubit count, and is costed separately.
- Efficient arbitrary diagonal synthesis. Only the specific quadratic and linear
  diagonals arising in these benchmarks are synthesised structurally.
- Fault-tolerant feasibility, or near-term hardware feasibility.
- Any hardware result. None was run.
- Novelty of the QFT, the QST, or the Strang splitting.

## Relation to the 2024 QDST infinite-well work

That work established the quantum discrete sine transform route for the infinite
square well. The zero-potential infinite well is, however, a case where the sine
propagator is *exact*, which limits what a step-count study there can establish.

Relative to that starting point, this repository adds: circuit-level validation of
the transform against the analytical DST-II rather than against a proxy; a direct
periodic-versus-Dirichlet measurement against an independent reference; a
non-zero-potential hard-wall benchmark where Trotter convergence is meaningful;
and resource counts qualified by ancillas, composition, synthesis model and
connectivity.

> **Verification note.** Exact bibliographic metadata for the 2024 QDST work — and
> for every entry in `../references/references.bib` — has **not** been verified
> from this offline working copy. Entries marked `VERIFY` in that file must be
> checked against the published record before submission. Nothing here should be
> taken as confirming an author list, title, venue, year or DOI.

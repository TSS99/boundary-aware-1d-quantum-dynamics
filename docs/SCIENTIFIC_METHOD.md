# Scientific method

## The organising idea

The spectral transform inside a split-operator step is not an implementation
detail. It determines which boundary condition the discretised problem carries,
and therefore which physical system is being simulated.

A discrete Fourier transform diagonalises the Laplacian on a **ring**: the grid
wraps, and amplitude leaving one end reappears at the other. A discrete sine
transform diagonalises the Laplacian on a **box with hard walls**: the
representation enforces `psi(0) = psi(L) = 0`. Choosing between them is a
modelling decision, not a numerical convenience, and it is the decision this
repository measures.

## Continuous problem

For a particle of mass `m` in one dimension,

```
i hbar d/dt psi(x, t) = [ -(hbar^2 / 2m) d^2/dx^2 + V(x) ] psi(x, t)
```

with either periodic boundary conditions on `[x_left, x_right)` or Dirichlet
conditions `psi(0, t) = psi(L, t) = 0`.

## Discretisation

Two grids, one per boundary model.

**Periodic.** `x_j = x_left + j dx`, `dx = (x_right - x_left) / N`, `j = 0..N-1`,
right endpoint excluded. This is the grid natural to an `N`-point DFT, and hence
to a QFT register of `n_q = log2 N` qubits.

**Dirichlet.** `x_j = (j + 1/2) dx`, `dx = L / N`. Both walls are excluded and
samples sit at cell midpoints. The half-cell offset is what makes the orthonormal
DST-II diagonalise the Dirichlet Laplacian exactly, with eigenvalues

```
E_nu = hbar^2 pi^2 nu^2 / (2 m L^2),   nu = 1 .. N
```

Note the index offset: DST-II output bin `k` carries mode `nu = k + 1`, because
the constant mode is not a Dirichlet eigenfunction.

## State encoding

Physical samples `psi(x_j)` are normalised in the quadrature norm
`dx * sum_j |psi(x_j)|^2 = 1` and carry units of `length^(-1/2)`.

Register amplitudes are

```
|Psi> = sqrt(dx) sum_j psi(x_j) |j>
```

normalised in the Euclidean norm and dimensionless. These are different objects;
`boundary_aware_dynamics.states` provides both directions explicitly, and a
quadrature-normalised array is never handed to a simulator directly.

Basis states are little-endian: qubit `q` carries bit `2^q` of `j`.

## Initial states

```
psi(x, 0) ~ exp[ -(x - x0)^2 / (4 sigma^2) ] exp[ i k0 (x - x0) ]
```

so that `|psi|^2 ~ exp[-(x-x0)^2 / (2 sigma^2)]` and **`sigma` is the standard
deviation of the probability density**. This convention is used in the code, the
configuration, the notebooks, the captions and the manuscript-alignment notes.

For hard-wall problems the packet is multiplied by `sin(pi x / L)` so that it
vanishes at both walls and is exactly representable in the sine basis.

## Transforms

**DFT.** Forward `F[k] = N^(-1/2) sum_j x[j] exp(-2 pi i j k / N)`, in
`numpy.fft.fftfreq` bin order: bin `k` carries signed index `k` for `k < N/2` and
`k - N` otherwise. Qiskit's `QFTGate` carries the **opposite** exponential sign,
so the forward transform is `QFTGate(n).inverse()`; it includes the terminating
swaps, so no manual bit reversal is required.

**DST-II.** The orthonormal matrix is

```
S[nu-1, j] = sqrt(2/N) sin(pi nu (j + 1/2) / N),   nu = 1 .. N-1
S[N-1,  j] = sqrt(1/N) (-1)^j
```

The extra `1/sqrt(2)` on the last row is the Nyquist normalisation. `S` is real
and orthogonal. Its rows are exactly the continuum Dirichlet eigenfunctions
sampled on the midpoint grid, which is why the zero-potential Dirichlet
propagator is exact — and why agreement with a sine-series reference in that case
is structural rather than independent evidence.

## Circuit mapping

**Periodic.** `F^dagger D_T F` on the `n_q` data qubits. No ancillas.

**Dirichlet.** The DST-II is realised exactly through an odd extension onto a
`4N`-point register. For data `x[j]`, set

```
y[2j + 1]      = +x[j]
y[4N - 2j - 1] = -x[j]
y[m]           =  0        otherwise
```

whose DFT carries the DST-II kernel. The second copy sits at bit pattern
`(bit 0 = 1, top bit = 1, middle bits = ~j)`, so the extension is `X`, `H` and
`n_q` CNOTs — a product of unitaries, hence its own uncomputation. This costs
**two ancillas**: one for the half-cell interleave, one for the odd reflection.

The transform does not leave clean DST coefficients in an `n_q`-qubit register:
mode `nu` appears at the four indices `{nu, 2N-nu, 2N+nu, 4N-nu}` (two for
`nu = N`). Extraction is unnecessary — applying the same phase to every index
carrying a given mode preserves the extended subspace, so the kinetic propagator
is `E^dagger F^dagger D_mu F E` with `mu` the triangle-wave mode map.

## Splitting

Second-order Strang:

```
U(dt) = e^{-iV dt/2hbar} T^{-1} e^{-iT_spec dt/hbar} T e^{-iV dt/2hbar}
```

with `T` the transform matched to the boundary condition. Over `r` steps adjacent
half-potential phases merge, giving one initial half-phase, `r-1` full phases,
one final half-phase and `r` kinetic blocks.

With `V = 0` under Dirichlet boundaries the splitting is **exact**, not
second-order: there is no non-commuting part left. This is why the free well is a
control benchmark and the tilted well is the Trotter benchmark.

## Structured phase synthesis

Register bits are idempotent (`b^2 = b`), so any diagonal quadratic in the
register index factorises into a global phase, `n_q` single-qubit phases and
`n_q(n_q-1)/2` controlled phases — `O(n^2)` rather than `O(2^n)`. This covers the
harmonic position phase, the signed momentum-square phase (FFT bin ordering is
two's complement), the folded sine-mode phase, and the linear tilt (which needs no
two-qubit gates at all). Derivations are in
`src/boundary_aware_dynamics/circuits/phases.py`.

## References

Three, chosen so that the comparison is never circular.

| Reference | Shares basis? | Shares method? | Use |
|---|---|---|---|
| Hermite eigenbasis | no | no | Benchmark A total error |
| Sine-Galerkin, exactly diagonalised | yes | no | Benchmark C total error |
| Finite-difference, exactly diagonalised | no | no | Benchmark B boundary comparison |
| Exact diagonalisation of the **discrete** Hamiltonian | yes | no | isolating Trotter error alone |

The last is what makes a convergence slope meaningful: it uses the same spatial
discretisation as the propagator, so grid and aliasing error cancel and only the
splitting error remains.

## Observables and measurement

Position and momentum expectations, variance, kinetic, potential and total energy
are computed spectrally in the basis matched to the boundary condition. Boundary
diagnostics report near-wall probability, wall residual, wrap-around probability
and probability current separately — near-wall probability is **not** leakage,
since the exact hard-wall solution has probability there too.

On hardware, computational-basis measurement gives samples of `|psi(x_j)|^2`, not
complex amplitudes. Every fidelity quoted in this repository is a **simulator
diagnostic**; obtaining it experimentally would require an overlap protocol or
tomography, at a shot cost that grows as `1/epsilon^2`. No hardware run is claimed
anywhere.

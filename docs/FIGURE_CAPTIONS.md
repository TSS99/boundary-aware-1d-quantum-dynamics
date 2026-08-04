# Figure captions

Manuscript-ready captions, kept apart from the images so they can be edited
without regenerating a figure. Figures carry no embedded titles or captions.

Numerical values quoted in a caption should be drawn from
`results/paper/metadata/paper_values.tex` rather than typed. The machine-readable
index of figure to source data, configuration hash and dimensions is
`results/<profile>/metadata/figure_manifest.csv`.

---

**`boundary_comparison`** — *Periodic and Dirichlet propagation of the same
hard-wall problem.* Both propagations use the identical initial state, box,
grid resolution, time interval and step count, and differ only in which spectral
transform sits inside the split-operator step. **(a)** Infidelity of each against
a finite-difference hard-wall reference that shares neither basis nor method with
either propagator; note the logarithmic axis. **(b)** Divergence between the two
propagations, `1 - F`, which reaches order unity: the two become nearly
orthogonal. **(c)** Residual wall amplitude, extrapolated to `x = 0` and `x = L`
from the two nearest midpoints. The periodic representation departs from the
hard-wall solution as soon as the packet reaches a wall, because a ring has no
walls to reflect from.

**`boundary_density_snapshots`** — *Probability density under both boundary
topologies.* The reference is drawn as a thick pale line beneath the numerical
curves. The Dirichlet propagation tracks it throughout; the periodic propagation
develops amplitude at the opposite end of the box, the wrap-around that a ring
topology permits and a box does not.

**`harmonic_density_snapshots`** — *Harmonic oscillator, Benchmark A.*
Probability density at four times over one classical period, periodic split
operator against the analytical Hermite eigenbasis reference.

**`harmonic_infidelity_vs_time`** — *Accuracy of the harmonic benchmark against
time.* Infidelity on a logarithmic axis; fidelity itself is not plotted because
it is bounded above by one and compresses precisely the range of interest.

**`harmonic_trotter_convergence`** — *Second-order Trotter convergence,
Benchmark A.* `L^2` state error against the time step, measured against exact
diagonalisation of the same discrete Hamiltonian so that spatial discretisation
cancels and the fitted slope is the order of the time integrator alone. The
shaded band marks the interval the slope was fitted over; the first point is
excluded because it lies in the pre-asymptotic transient. The dashed line is the
theoretical `O(dt^2)` guide.

**`infinite_well_density_snapshots`** — *Infinite square well, Benchmark B.*
Zero interior potential, so the Dirichlet split-operator propagator is exact and
this benchmark serves as a control rather than as a test of the time integrator.

**`infinite_well_infidelity_vs_time`** — *Residual error of the free-well
control.* The residual sits at the round-off floor at all times. This near-perfect
agreement is **structural**: the DST-II rows are the analytical sine eigenmodes
sampled on the midpoint grid, and the propagator uses the same eigenvalues as the
reference, so the agreement is built in and is not independent evidence of
continuum accuracy.

**`tilted_well_density_snapshots`** — *Tilted infinite well, Benchmark C.*
`V(x) = F(x - L/2)` with hard walls, the only benchmark here carrying both
Dirichlet boundaries and a non-zero interior potential.

**`tilted_well_infidelity_vs_time`** — *Accuracy of the tilted-well benchmark
against time,* measured against a converged sine-Galerkin reference.

**`tilted_well_trotter_convergence`** — *Second-order Trotter convergence under
Dirichlet boundaries, Benchmark C.* Because `[T, V] != 0` here, this is the
repository's only genuine measurement of the splitting order under hard walls.
Fitted slope and `R^2` are shown; the shaded band is the fit interval and the
dashed line the `O(dt^2)` guide.

**`resource_scaling`** — *Two-qubit gate count of one propagation step against
total register size.* Structured phase synthesis, all-to-all connectivity,
optimisation level 3, `rz/sx/x/cx` basis, fixed transpiler seed. The Dirichlet
register includes the two QST ancillas, so its total exceeds the data-qubit count
by two. Counts cover the propagation core only: state preparation and measurement
are excluded and reported separately.

**`graphical_abstract`** — *Boundary condition determines grid, transform and
circuit structure.* A Dirichlet condition leads to a midpoint grid, the DST-II
and a two-ancilla quantum sine transform; a periodic condition leads to a
uniform grid, the DFT and an ancilla-free QFT. Both feed the same Strang
splitting and are validated against independent references. The two paths are not
interchangeable: they represent different physical systems.

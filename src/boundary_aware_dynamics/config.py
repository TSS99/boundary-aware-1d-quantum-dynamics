"""Typed configuration loaded from YAML.

The YAML files under ``configs/`` are the single source of truth for every
physical and numerical parameter used in the paper.  Notebooks load a profile
rather than defining parameters inline, so a value cannot silently disagree
between two notebooks.

Each config carries a ``config_hash`` derived from its canonical serialisation.
The hash is recorded alongside every generated figure and table, which is what
makes stale results detectable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Physics:
    """Fundamental constants and the particle mass, in simulation units."""

    hbar: float = 1.0
    mass: float = 1.0


@dataclass(frozen=True)
class InitialState:
    """Gaussian wavepacket parameters.

    ``sigma`` is the standard deviation of the probability density ``|psi|^2``,
    i.e. the envelope is ``exp[-(x - centre)^2 / (4 sigma^2)]``.
    """

    centre: float
    momentum: float
    sigma: float
    sine_window: bool = False


@dataclass(frozen=True)
class Domain:
    """Spatial domain and its boundary model."""

    boundary: str
    n_grid: int
    x_left: float = 0.0
    x_right: float = 1.0

    def __post_init__(self) -> None:
        if self.boundary not in ("periodic", "dirichlet"):
            raise ValueError(f"boundary must be 'periodic' or 'dirichlet', got {self.boundary!r}.")

    @property
    def length(self) -> float:
        return self.x_right - self.x_left


@dataclass(frozen=True)
class TimeGrid:
    """Propagation interval and the Trotter step counts to sweep."""

    t_max: float
    n_steps: int
    step_sweep: tuple[int, ...] = ()

    def time_step(self, n_steps: int | None = None) -> float:
        return self.t_max / float(n_steps if n_steps is not None else self.n_steps)


@dataclass(frozen=True)
class Reference:
    """Settings for the independent reference solution."""

    dense_grid_size: int = 4097
    basis_cap: int = 256
    tail_tolerance: float = 1e-10


@dataclass(frozen=True)
class Circuits:
    """Transpilation and synthesis settings for resource accounting."""

    basis_gates: tuple[str, ...] = ("rz", "sx", "x", "cx")
    optimisation_level: int = 3
    coupling: str = "all_to_all"
    seed: int = 20240517
    approximation_degree: float = 0.0
    validation_qubits: tuple[int, ...] = (2, 3)
    validation_tolerance: float = 1e-10


@dataclass(frozen=True)
class Plotting:
    """Figure export settings."""

    mode: str = "publication"
    dpi: int = 600
    single_column_mm: float = 85.0
    double_column_mm: float = 170.0
    formats: tuple[str, ...] = ("pdf", "png")


@dataclass(frozen=True)
class Benchmark:
    """One physical experiment: domain, state, time grid and reference."""

    name: str
    domain: Domain
    initial_state: InitialState
    time_grid: TimeGrid
    reference: Reference = field(default_factory=Reference)
    omega: float | None = None
    tilt_force: float | None = None
    grid_sweep: tuple[int, ...] = ()


@dataclass(frozen=True)
class RunConfig:
    """A complete reproduction profile."""

    profile: str
    physics: Physics
    benchmarks: dict[str, Benchmark]
    circuits: Circuits = field(default_factory=Circuits)
    plotting: Plotting = field(default_factory=Plotting)
    seed: int = 20240517
    output_root: str = "results"

    @property
    def config_hash(self) -> str:
        return _hash_payload(_to_plain(self))

    def benchmark(self, name: str) -> Benchmark:
        try:
            return self.benchmarks[name]
        except KeyError:
            available = ", ".join(sorted(self.benchmarks))
            raise KeyError(f"Unknown benchmark {name!r}. Available: {available}.") from None


def _to_plain(value: Any) -> Any:
    """Recursively convert dataclasses/tuples to JSON-serialisable primitives."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _hash_payload(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _build(cls: type, payload: dict[str, Any]) -> Any:
    """Instantiate a config dataclass, coercing lists to tuples."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(f"Unknown key(s) for {cls.__name__}: {sorted(unknown)}.")
    kwargs: dict[str, Any] = {}
    for name, value in payload.items():
        kwargs[name] = tuple(value) if isinstance(value, list) else value
    return cls(**kwargs)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML reproduction profile."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    benchmarks: dict[str, Benchmark] = {}
    for name, spec in raw.get("benchmarks", {}).items():
        spec = dict(spec)
        benchmarks[name] = Benchmark(
            name=name,
            domain=_build(Domain, spec.pop("domain")),
            initial_state=_build(InitialState, spec.pop("initial_state")),
            time_grid=_build(TimeGrid, spec.pop("time_grid")),
            reference=_build(Reference, spec.pop("reference", {})),
            omega=spec.pop("omega", None),
            tilt_force=spec.pop("tilt_force", None),
            grid_sweep=tuple(spec.pop("grid_sweep", ())),
        )
        if spec:
            raise ValueError(f"Unknown key(s) in benchmark {name!r}: {sorted(spec)}.")

    return RunConfig(
        profile=raw["profile"],
        physics=_build(Physics, raw.get("physics", {})),
        benchmarks=benchmarks,
        circuits=_build(Circuits, raw.get("circuits", {})),
        plotting=_build(Plotting, raw.get("plotting", {})),
        seed=raw.get("seed", 20240517),
        output_root=raw.get("output_root", "results"),
    )


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Return a plain-dict view suitable for JSON/YAML metadata export."""
    return _to_plain(config)


__all__ = [
    "Benchmark",
    "Circuits",
    "Domain",
    "InitialState",
    "Physics",
    "Plotting",
    "Reference",
    "RunConfig",
    "TimeGrid",
    "config_to_dict",
    "load_config",
]

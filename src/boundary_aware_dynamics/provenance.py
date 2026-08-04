"""Provenance recording and provenance-aware staleness detection.

A result file existing is not evidence that it is current.  Every reproduction
records the commit, the working-tree cleanliness, the dependency versions, the
configuration hash, a hash of the package source and hashes of the outputs
themselves.  :func:`check_staleness` then compares a stored record against the
present state, so a figure generated before a change to ``transforms.py`` is
reported as stale rather than silently reused.

Results produced from a dirty working tree are labelled as such.  They are still
written -- that is the normal state during development -- but the label travels
with them so they cannot be mistaken for a reproducible artefact.

Volatile fields (wall-clock timestamps, runtimes) are kept in a separate section
from the deterministic ones, so that re-running identical work produces identical
deterministic fields and the diff stays readable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "Provenance",
    "check_staleness",
    "collect_provenance",
    "hash_directory",
    "hash_file",
    "hash_source_tree",
    "read_provenance",
    "write_provenance",
]

TRACKED_PACKAGES = ("numpy", "scipy", "qiskit", "matplotlib", "pandas", "PyYAML")


def _run_git(*arguments: str) -> str | None:
    """Run a read-only git command, returning ``None`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            capture_output=True, text=True, check=False,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def hash_file(path: Path) -> str:
    """Return a short SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def hash_directory(directory: Path, patterns: tuple[str, ...] = ("*",)) -> dict[str, str]:
    """Return ``{relative path: hash}`` for matching files, sorted for stability."""
    directory = Path(directory)
    if not directory.exists():
        return {}
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(p for p in directory.rglob(pattern) if p.is_file())
    return {
        str(path.relative_to(directory)).replace("\\", "/"): hash_file(path)
        for path in sorted(paths)
    }


def hash_source_tree() -> str:
    """Return one hash covering every ``.py`` file in the package.

    This is what makes a result detectably stale after the code that produced it
    changes, even when the configuration is untouched.
    """
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@dataclass
class Provenance:
    """Deterministic and volatile provenance for one reproduction run."""

    profile: str
    config_hash: str
    source_hash: str
    git_commit: str | None
    git_dirty: bool
    python_version: str
    platform_name: str
    dependencies: dict[str, str]
    seeds: dict[str, int]
    schema_version: str = "1"
    output_hashes: dict[str, str] = field(default_factory=dict)
    volatile: dict[str, Any] = field(default_factory=dict)

    @property
    def is_reproducible_artefact(self) -> bool:
        """True only when the tree is clean and the commit is known."""
        return self.git_commit is not None and not self.git_dirty

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_provenance(config, runtime_seconds: float | None = None) -> Provenance:
    """Gather provenance for the current environment and configuration."""
    import importlib.metadata as metadata

    dependencies = {}
    for package in TRACKED_PACKAGES:
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            dependencies[package] = "not installed"

    status = _run_git("status", "--porcelain")
    volatile: dict[str, Any] = {"generated_at_utc": datetime.now(timezone.utc).isoformat()}
    if runtime_seconds is not None:
        volatile["runtime_seconds"] = round(runtime_seconds, 3)

    return Provenance(
        profile=config.profile,
        config_hash=config.config_hash,
        source_hash=hash_source_tree(),
        git_commit=_run_git("rev-parse", "HEAD"),
        git_dirty=bool(status),
        python_version=sys.version.split()[0],
        platform_name=platform.platform(),
        dependencies=dependencies,
        seeds={"global": config.seed, "transpiler": config.circuits.seed},
        volatile=volatile,
    )


def write_provenance(provenance: Provenance, directory: Path) -> Path:
    """Write ``provenance.json`` and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "provenance.json"
    path.write_text(json.dumps(provenance.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_provenance(directory: Path) -> Provenance | None:
    """Read a stored provenance record, or ``None`` if there is not one."""
    path = Path(directory) / "provenance.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("schema_version", None)
    return Provenance(schema_version="1", **payload)


def check_staleness(stored: Provenance | None, current: Provenance) -> dict[str, Any]:
    """Compare a stored record against the present state.

    Returns the reasons a result is considered stale rather than a bare boolean,
    so a verification failure says *what* changed.
    """
    if stored is None:
        return {"stale": True, "reasons": ["no stored provenance record"]}

    reasons = []
    if stored.schema_version != current.schema_version:
        reasons.append(
            f"schema version changed: {stored.schema_version} -> {current.schema_version}"
        )
    if stored.config_hash != current.config_hash:
        reasons.append(f"configuration changed: {stored.config_hash} -> {current.config_hash}")
    if stored.source_hash != current.source_hash:
        reasons.append(f"package source changed: {stored.source_hash} -> {current.source_hash}")

    changed = {
        name: (stored.dependencies.get(name), version)
        for name, version in current.dependencies.items()
        if stored.dependencies.get(name) != version
    }
    if changed:
        reasons.append(f"dependency versions changed: {changed}")
    if stored.seeds != current.seeds:
        reasons.append(f"seeds changed: {stored.seeds} -> {current.seeds}")

    return {
        "stale": bool(reasons),
        "reasons": reasons,
        "stored_was_dirty": stored.git_dirty,
        "current_is_dirty": current.git_dirty,
    }


def verify_output_hashes(provenance: Provenance, directory: Path) -> dict[str, Any]:
    """Recompute output hashes and report any that no longer match."""
    directory = Path(directory)
    mismatched, missing = {}, []
    for relative, expected in provenance.output_hashes.items():
        path = directory / relative
        if not path.exists():
            missing.append(relative)
        else:
            actual = hash_file(path)
            if actual != expected:
                mismatched[relative] = {"expected": expected, "actual": actual}
    return {
        "ok": not mismatched and not missing,
        "missing": missing,
        "mismatched": mismatched,
        "n_checked": len(provenance.output_hashes),
    }

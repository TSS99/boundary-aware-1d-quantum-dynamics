"""Configuration loading, validation and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from boundary_aware_dynamics.config import Domain, config_to_dict, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PROFILES = ["paper.yaml", "smoke.yaml"]


@pytest.mark.parametrize("filename", PROFILES)
def test_shipped_profiles_load(filename):
    config = load_config(CONFIG_DIR / filename)
    assert config.profile == Path(filename).stem
    assert set(config.benchmarks) == {"harmonic", "infinite_well", "tilted_well"}


@pytest.mark.parametrize("filename", PROFILES)
def test_every_benchmark_has_a_power_of_two_grid(filename):
    config = load_config(CONFIG_DIR / filename)
    for benchmark in config.benchmarks.values():
        n_grid = benchmark.domain.n_grid
        assert n_grid & (n_grid - 1) == 0, f"{benchmark.name} grid {n_grid} is not a power of two"
        for swept in benchmark.grid_sweep:
            assert swept & (swept - 1) == 0


@pytest.mark.parametrize("filename", PROFILES)
def test_boundary_models_match_the_intended_physics(filename):
    config = load_config(CONFIG_DIR / filename)
    assert config.benchmark("harmonic").domain.boundary == "periodic"
    assert config.benchmark("infinite_well").domain.boundary == "dirichlet"
    assert config.benchmark("tilted_well").domain.boundary == "dirichlet"


@pytest.mark.parametrize("filename", PROFILES)
def test_only_the_tilted_well_carries_an_interior_potential(filename):
    # Benchmark B must stay a zero-potential control; the Trotter benchmark is
    # Benchmark C.  Swapping these is the confusion this test guards against.
    config = load_config(CONFIG_DIR / filename)
    assert config.benchmark("infinite_well").tilt_force is None
    assert config.benchmark("tilted_well").tilt_force is not None
    assert config.benchmark("tilted_well").tilt_force != 0.0


@pytest.mark.parametrize("filename", PROFILES)
def test_step_sweeps_are_geometric_and_long_enough_to_fit_a_slope(filename):
    config = load_config(CONFIG_DIR / filename)
    for benchmark in config.benchmarks.values():
        sweep = benchmark.time_grid.step_sweep
        assert len(sweep) >= 4
        ratios = [b / a for a, b in zip(sweep, sweep[1:])]
        assert all(ratio == pytest.approx(2.0) for ratio in ratios)


def test_time_step_follows_t_max_over_r():
    config = load_config(CONFIG_DIR / "paper.yaml")
    time_grid = config.benchmark("harmonic").time_grid
    assert time_grid.time_step() == pytest.approx(time_grid.t_max / time_grid.n_steps)
    assert time_grid.time_step(40) == pytest.approx(time_grid.t_max / 40)


def test_unknown_boundary_is_rejected():
    with pytest.raises(ValueError, match="boundary must be"):
        Domain(boundary="reflecting", n_grid=8)


def test_unknown_config_key_is_rejected(tmp_path):
    payload = yaml.safe_load((CONFIG_DIR / "smoke.yaml").read_text(encoding="utf-8"))
    payload["benchmarks"]["harmonic"]["typo_key"] = 1
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(path)


def test_requesting_a_missing_benchmark_names_the_available_ones():
    config = load_config(CONFIG_DIR / "paper.yaml")
    with pytest.raises(KeyError, match="harmonic"):
        config.benchmark("does_not_exist")


def test_config_hash_is_deterministic_and_profile_sensitive():
    paper = load_config(CONFIG_DIR / "paper.yaml")
    smoke = load_config(CONFIG_DIR / "smoke.yaml")
    assert paper.config_hash == load_config(CONFIG_DIR / "paper.yaml").config_hash
    assert paper.config_hash != smoke.config_hash
    assert len(paper.config_hash) == 16


def test_config_serialises_to_plain_types():
    payload = config_to_dict(load_config(CONFIG_DIR / "smoke.yaml"))
    assert payload["profile"] == "smoke"
    assert payload["benchmarks"]["harmonic"]["domain"]["boundary"] == "periodic"
    assert isinstance(payload["circuits"]["basis_gates"], list)

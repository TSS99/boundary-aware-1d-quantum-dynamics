"""Publication figures.

Two modes.  ``preview`` is for looking at results in a notebook; ``publication``
applies the journal styling and writes vector output.  Both draw the same data
through the same functions, so a figure cannot look different in the paper from
the way it looked when it was checked.

Rules enforced here rather than left to the caller
--------------------------------------------------
* **Fidelity is never plotted.**  It is bounded above by one, so near-perfect
  results compress into an uninformative flat line, and an axis drawn above one
  shows a region no data point can occupy.  Accuracy is plotted as *infidelity*
  or *state error* on a logarithmic axis instead.
* **No titles and no captions inside the image.**  Captions live in
  ``docs/FIGURE_CAPTIONS.md`` so they can be edited without regenerating a
  figure, and journals set titles from the caption.
* **Numerical floors are annotated**, so a curve that has stopped converging is
  visibly at the floor rather than looking like a failure of the method.
* **Convergence fits carry their window.**  The fitted interval is shaded and the
  expected theoretical slope is drawn as a guide, so a reported slope is never
  detached from the range it was fitted over.

Colour
------
The categorical palette is validated for colour-vision deficiency (worst adjacent
pair ΔE 9.2 deutan, 27.6 normal).  Colour is never the only channel: every series
also carries a distinct dash pattern and a distinct marker, so the figures survive
greyscale printing.  Reference curves are drawn in neutral ink because a reference
is a baseline rather than one category among several.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "FigureManifest",
    "PALETTE",
    "SERIES",
    "apply_style",
    "millimetres",
    "plot_boundary_comparison",
    "plot_convergence",
    "plot_density_snapshots",
    "plot_error_vs_time",
    "plot_graphical_abstract",
    "plot_resource_scaling",
    "save_figure",
]

# Validated categorical palette (see module docstring). Reference is neutral ink.
PALETTE = {
    "reference": "#3d3d3a",
    "dirichlet": "#2a78d6",
    "periodic": "#eb6834",
    "third": "#1baf7a",
    "guide": "#4a3aa7",
    "floor": "#8a8a84",
    "grid": "#d6d6d0",
}

# Colour + dash + marker together, so identity survives greyscale.
SERIES = {
    "reference": {"color": PALETTE["reference"], "linestyle": "-", "marker": None},
    "dirichlet": {"color": PALETTE["dirichlet"], "linestyle": "-", "marker": "o"},
    "periodic": {"color": PALETTE["periodic"], "linestyle": "--", "marker": "s"},
    "third": {"color": PALETTE["third"], "linestyle": "-.", "marker": "^"},
}


def millimetres(value: float) -> float:
    """Convert millimetres to inches, the unit matplotlib wants."""
    return value / 25.4


def apply_style(mode: str = "publication") -> None:
    """Apply the figure style for ``preview`` or ``publication``."""
    if mode not in ("preview", "publication"):
        raise ValueError(f"mode must be 'preview' or 'publication', got {mode!r}.")

    base = 8.0 if mode == "publication" else 10.0
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "mathtext.fontset": "dejavusans",
            "font.size": base,
            "axes.labelsize": base,
            "axes.titlesize": base,
            "legend.fontsize": base - 0.5,
            "xtick.labelsize": base - 0.5,
            "ytick.labelsize": base - 0.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _recessive_grid(axis: plt.Axes) -> None:
    axis.grid(True, color=PALETTE["grid"], linewidth=0.4, linestyle="-", alpha=0.7)
    axis.set_axisbelow(True)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.16, 1.04, label, transform=axis.transAxes,
        fontsize=mpl.rcParams["font.size"] + 1, fontweight="bold", va="bottom", ha="left",
    )


# ------------------------------------------------------------- manifest ----


@dataclass
class FigureRecord:
    """Everything needed to regenerate and caption one figure."""

    figure_id: str
    filename: str
    formats: list[str]
    source_notebook: str
    source_data: str
    config_hash: str
    width_mm: float
    height_mm: float
    caption: str
    generation_command: str
    key_parameters: dict[str, Any] = field(default_factory=dict)


class FigureManifest:
    """Collects figure records and writes them next to the figures."""

    def __init__(self) -> None:
        self.records: list[FigureRecord] = []

    def add(self, record: FigureRecord) -> None:
        self.records.append(record)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([asdict(record) for record in self.records])

    def write(self, directory: Path) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / "figure_manifest.csv"
        json_path = directory / "figure_manifest.json"
        self.to_dataframe().to_csv(csv_path, index=False)
        json_path.write_text(
            json.dumps([asdict(record) for record in self.records], indent=2), encoding="utf-8"
        )
        return csv_path, json_path


def save_figure(
    figure: plt.Figure,
    stem: str,
    directory: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
) -> list[Path]:
    """Write a figure in each requested format and confirm the files exist."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in formats:
        path = directory / f"{stem}.{suffix}"
        figure.savefig(path, dpi=dpi)
        if not path.exists():
            raise FileNotFoundError(f"Failed to write {path}.")
        paths.append(path)
    return paths


# ---------------------------------------------------------------- plots ----


def plot_density_snapshots(
    positions: np.ndarray,
    times: np.ndarray,
    series: dict[str, np.ndarray],
    snapshot_indices: np.ndarray,
    width_mm: float | None = None,
) -> plt.Figure:
    """Stacked density panels at selected times.

    ``series`` maps a style key from :data:`SERIES` to an array of states.  The
    legend is placed below the panels rather than inside the top one: an
    in-axes legend for three series is wide enough at single-column width to
    collapse the axes entirely.
    """
    if width_mm is None:
        width_mm = 85.0 if len(series) <= 2 else 170.0

    n_panels = len(snapshot_indices)
    figure, axes = plt.subplots(
        n_panels, 1, figsize=(millimetres(width_mm), 0.8 * n_panels + 0.65),
        sharex=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    labels = {
        "reference": "Reference",
        "dirichlet": "Dirichlet (DST-II)",
        "periodic": "Periodic (FFT)",
        "third": "Third",
    }
    for axis, index in zip(axes, snapshot_indices):
        for key, states in series.items():
            style = SERIES[key]
            # The reference is drawn as a thick pale underlay so that a numerical
            # curve lying on top of it stays visible rather than hiding it.
            is_reference = key == "reference"
            axis.plot(
                positions, np.abs(states[index]) ** 2,
                color=style["color"], linestyle=style["linestyle"],
                linewidth=2.6 if is_reference else 1.2,
                alpha=0.35 if is_reference else 1.0,
                solid_capstyle="round", label=labels.get(key, key),
            )
        axis.set_ylabel(r"$|\psi|^2$")
        axis.text(
            0.015, 0.9, rf"$t={times[index]:.2f}$", transform=axis.transAxes,
            ha="left", va="top", fontsize=mpl.rcParams["font.size"] - 0.5,
            bbox={"boxstyle": "square,pad=0.15", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.85},
        )
        _recessive_grid(axis)

    axes[-1].set_xlabel(r"$x$")
    handles, labels_used = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels_used, loc="outside lower center",
        ncol=len(series), columnspacing=1.4, handlelength=2.0,
    )
    return figure


def plot_error_vs_time(
    times: np.ndarray,
    series: dict[str, np.ndarray],
    ylabel: str = "Infidelity",
    floor: float | None = None,
    width_mm: float = 85.0,
    height_mm: float = 60.0,
) -> plt.Figure:
    """Error against time on a logarithmic axis.

    Fidelity itself is deliberately not offered: see the module docstring.
    """
    figure, axis = plt.subplots(
        figsize=(millimetres(width_mm), millimetres(height_mm)), constrained_layout=True
    )
    labels = {
        "dirichlet": "Dirichlet (DST-II)",
        "periodic": "Periodic (FFT)",
        "third": "Third",
        "reference": "Reference",
    }
    for key, values in series.items():
        style = SERIES[key]
        safe = np.maximum(np.asarray(values, dtype=float), 1e-18)
        axis.semilogy(
            times, safe, color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markevery=max(1, len(times) // 12),
            markerfacecolor="white", markeredgewidth=0.9, label=labels.get(key, key),
        )

    if floor is not None:
        axis.axhline(floor, color=PALETTE["floor"], linewidth=0.8, linestyle=":")
        axis.text(
            times[-1], floor * 1.6, "numerical floor", ha="right", va="bottom",
            fontsize=mpl.rcParams["font.size"] - 1.5, color=PALETTE["floor"],
        )

    axis.set_xlabel(r"$t$")
    axis.set_ylabel(ylabel)
    _recessive_grid(axis)
    if len(series) > 1:
        axis.legend(loc="best")
    return figure


def plot_convergence(
    step_sizes: np.ndarray,
    errors: np.ndarray,
    fit: dict[str, Any] | None = None,
    expected_slope: float | None = 2.0,
    xlabel: str = r"$\Delta t$",
    ylabel: str = r"$L^2$ state error",
    floor: float | None = None,
    width_mm: float = 85.0,
    height_mm: float = 65.0,
) -> plt.Figure:
    """Log-log convergence plot carrying its fit, fit window and expected slope."""
    figure, axis = plt.subplots(
        figsize=(millimetres(width_mm), millimetres(height_mm)), constrained_layout=True
    )
    step_sizes = np.asarray(step_sizes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    axis.loglog(
        step_sizes, errors, color=PALETTE["dirichlet"], linestyle="none",
        marker="o", markerfacecolor="white", markeredgewidth=1.1, label="measured",
    )

    if fit is not None and np.isfinite(fit.get("slope", np.nan)):
        start = fit.get("fit_from_index", 0)
        fitted_x = step_sizes[start:]
        fitted_y = np.exp(fit["intercept"]) * fitted_x ** fit["slope"]
        axis.loglog(
            fitted_x, fitted_y, color=PALETTE["dirichlet"], linestyle="-", linewidth=1.2,
            label=rf"fit: slope $={fit['slope']:.2f}$, $R^2={fit['r_squared']:.4f}$",
        )
        # Shade the interval the slope was actually fitted over.
        axis.axvspan(fitted_x.min(), fitted_x.max(), color=PALETTE["dirichlet"], alpha=0.07, lw=0)

    if expected_slope is not None:
        anchor_x, anchor_y = step_sizes[-1], errors[-1]
        guide_y = anchor_y * (step_sizes / anchor_x) ** expected_slope
        axis.loglog(
            step_sizes, guide_y, color=PALETTE["guide"], linestyle="--", linewidth=1.0,
            label=rf"$\mathcal{{O}}(\Delta t^{{{expected_slope:g}}})$ guide",
        )

    # Draw the numerical floor only when the data is actually near it. Otherwise
    # the axis stretches over empty decades and the trend looks flat.
    if floor is not None and errors.min() < 100.0 * floor:
        axis.axhline(floor, color=PALETTE["floor"], linewidth=0.8, linestyle=":")
        axis.text(
            step_sizes.max(), floor * 1.5, "numerical floor", ha="right", va="bottom",
            fontsize=mpl.rcParams["font.size"] - 1.5, color=PALETTE["floor"],
        )
    else:
        axis.set_ylim(errors.min() / 4.0, errors.max() * 4.0)

    # Label only the sampled step sizes: the default log minor labels collide
    # when the range spans well under a decade per tick.
    axis.set_xticks(step_sizes)
    axis.set_xticklabels([f"{value:.3g}" for value in step_sizes])
    axis.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    _recessive_grid(axis)
    axis.legend(loc="best")
    return figure


def plot_boundary_comparison(
    times: np.ndarray,
    dirichlet_infidelity: np.ndarray,
    periodic_infidelity: np.ndarray,
    cross_fidelity: np.ndarray,
    dirichlet_wall: np.ndarray,
    periodic_wall: np.ndarray,
    width_mm: float = 170.0,
) -> plt.Figure:
    """The central experiment, as three panels sharing a time axis.

    (a) error of each propagator against an independent hard-wall reference;
    (b) how far the two propagations have drifted from one another;
    (c) how well each respects the wall condition it is supposed to model.
    """
    figure, axes = plt.subplots(
        1, 3, figsize=(millimetres(width_mm), millimetres(52.0)), constrained_layout=True
    )

    # Labels are kept short: at 170 mm across three panels there is roughly
    # 55 mm per panel, and a long y-label is silently clipped.
    floor = 1e-16
    axes[0].semilogy(
        times, np.maximum(dirichlet_infidelity, floor), **_line("dirichlet", times),
        label="Dirichlet (DST-II)",
    )
    axes[0].semilogy(
        times, np.maximum(periodic_infidelity, floor), **_line("periodic", times),
        label="Periodic (FFT)",
    )
    axes[0].set_ylim(bottom=0.2 * floor)
    axes[0].set_ylabel("Infidelity vs reference")
    axes[0].legend(loc="lower right")
    _panel_label(axes[0], "(a)")

    axes[1].plot(times, 1.0 - cross_fidelity, color=PALETTE["third"], linewidth=1.4)
    axes[1].set_ylabel(r"Divergence, $1-\mathcal{F}$")
    axes[1].set_ylim(-0.02, 1.02)
    _panel_label(axes[1], "(b)")

    axes[2].plot(times, dirichlet_wall, **_line("dirichlet", times), label="Dirichlet (DST-II)")
    axes[2].plot(times, periodic_wall, **_line("periodic", times), label="Periodic (FFT)")
    axes[2].set_ylabel("Wall residual")
    axes[2].legend(loc="upper left")
    _panel_label(axes[2], "(c)")

    for axis in axes:
        axis.set_xlabel(r"$t$")
        _recessive_grid(axis)
    return figure


def _line(key: str, times: np.ndarray) -> dict[str, Any]:
    style = SERIES[key]
    return {
        "color": style["color"],
        "linestyle": style["linestyle"],
        "marker": style["marker"],
        "markevery": max(1, len(times) // 10),
        "markerfacecolor": "white",
        "markeredgewidth": 0.9,
        "linewidth": 1.4,
    }


def plot_resource_scaling(
    qubit_counts: np.ndarray,
    series: dict[str, np.ndarray],
    ylabel: str = "Two-qubit gates",
    width_mm: float = 85.0,
    height_mm: float = 62.0,
) -> plt.Figure:
    """Gate cost against register size, with one line per synthesis model."""
    figure, axis = plt.subplots(
        figsize=(millimetres(width_mm), millimetres(height_mm)), constrained_layout=True
    )
    keys = ["dirichlet", "periodic", "third"]
    for (label, values), key in zip(series.items(), keys):
        style = SERIES[key]
        axis.plot(
            qubit_counts, values, color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markerfacecolor="white", markeredgewidth=0.9, label=label,
        )
    axis.set_xlabel("Total qubits (data + ancilla)")
    axis.set_ylabel(ylabel)
    _recessive_grid(axis)
    axis.legend(loc="upper left")
    return figure


def plot_graphical_abstract(width_mm: float = 170.0) -> plt.Figure:
    """Flow diagram: boundary condition -> grid -> transform -> propagation -> validation."""
    figure, axis = plt.subplots(
        figsize=(millimetres(width_mm), millimetres(46.0)), constrained_layout=True
    )
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    rows = [
        (0.72, "Dirichlet\n(hard wall)", "midpoint grid\n$x_j=(j+\\frac{1}{2})\\Delta x$",
         "DST-II / QST\n(+2 ancillas)", PALETTE["dirichlet"]),
        (0.28, "Periodic\n(ring)", "periodic grid\n$x_j=x_0+j\\Delta x$",
         "DFT / QFT\n(no ancillas)", PALETTE["periodic"]),
    ]
    # Positions are chosen so the rightmost box stays inside the axes; a box
    # centred much beyond 0.85 is clipped at this figure width.
    x_positions = [0.07, 0.25, 0.43]
    for y, boundary, grid_text, transform, colour in rows:
        for x, text in zip(x_positions, (boundary, grid_text, transform)):
            axis.text(
                x, y, text, ha="center", va="center", fontsize=7.0, color=colour,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "edgecolor": colour, "linewidth": 1.0},
            )
        for start, end in zip(x_positions, x_positions[1:]):
            axis.annotate(
                "", xy=(end - 0.068, y), xytext=(start + 0.068, y),
                arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": 1.1},
            )
        axis.annotate(
            "", xy=(0.565, 0.5), xytext=(x_positions[-1] + 0.068, y),
            arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": 1.1},
        )

    axis.text(
        0.64, 0.5, "Strang split\npropagation", ha="center", va="center", fontsize=7.0,
        color=PALETTE["reference"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": PALETTE["reference"], "linewidth": 1.0},
    )
    axis.annotate(
        "", xy=(0.775, 0.5), xytext=(0.715, 0.5),
        arrowprops={"arrowstyle": "-|>", "color": PALETTE["reference"], "linewidth": 1.1},
    )
    axis.text(
        0.87, 0.5, "validation vs\nindependent\nreference", ha="center", va="center",
        fontsize=7.0, color=PALETTE["reference"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": PALETTE["reference"], "linewidth": 1.0},
    )
    return figure

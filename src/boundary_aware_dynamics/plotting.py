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
  visibly at the floor rather than looking like a failure of the method.  The
  annotation is placed where the data is furthest from the floor, so it never
  lands on a curve.
* **Nothing is written on top of data.**  Every panel reserves empty space above
  its data before a legend or an in-axes label is drawn there, and legends carry
  an opaque frame so that a curve passing behind one cannot make it unreadable.
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
from matplotlib.patches import FancyBboxPatch

__all__ = [
    "FigureManifest",
    "PALETTE",
    "SERIES",
    "apply_style",
    "millimetres",
    "plot_boundary_comparison",
    "plot_circuit_diagram",
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


def _headroom(axis: plt.Axes, fraction: float = 0.30) -> None:
    """Reserve empty space above the data.

    A legend or an in-axes label is only safe once the space it will occupy is
    known to be empty, so every panel that carries one calls this first.
    ``fraction`` is measured in axis span: decades on a logarithmic axis, data
    units on a linear one.
    """
    bottom, top = axis.get_ylim()
    if axis.get_yscale() == "log":
        if bottom <= 0.0 or top <= 0.0:
            return
        decades = np.log10(top / bottom)
        axis.set_ylim(bottom, bottom * 10.0 ** (decades * (1.0 + fraction)))
    else:
        axis.set_ylim(bottom, bottom + (top - bottom) * (1.0 + fraction))


def _framed_legend(axis: plt.Axes, **kwargs: Any) -> plt.Legend:
    """Legend on an opaque panel, so a curve behind it cannot obscure the text."""
    legend = axis.legend(**kwargs)
    legend.set_frame_on(True)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(PALETTE["grid"])
    frame.set_linewidth(0.5)
    frame.set_alpha(0.92)
    return legend


def _clearest_fraction(
    x_values: np.ndarray, curves: list[np.ndarray], log_x: bool
) -> float:
    """Fraction along the x axis where the data sits furthest above the floor.

    Used to place the floor annotation: writing it at a fixed end of the axis
    puts it on top of a curve whenever that curve happens to end near the floor.
    """
    positions = np.asarray(x_values, dtype=float)
    if log_x:
        positions = np.log10(np.maximum(positions, 1e-300))
    low, high = positions.min(), positions.max()
    if high <= low:
        return 0.5
    best_fraction, best_clearance = 0.5, -np.inf
    for fraction in (0.15, 0.5, 0.85):
        centre = low + fraction * (high - low)
        window = np.abs(positions - centre) <= 0.18 * (high - low)
        if not window.any():
            continue
        clearance = min(np.min(np.asarray(curve, dtype=float)[window]) for curve in curves)
        if clearance > best_clearance:
            best_fraction, best_clearance = fraction, clearance
    return best_fraction


def _annotate_floor(
    axis: plt.Axes, x_values: np.ndarray, curves: list[np.ndarray], floor: float
) -> None:
    """Draw the numerical floor and label it clear of the data."""
    axis.axhline(floor, color=PALETTE["floor"], linewidth=0.8, linestyle=":")
    fraction = _clearest_fraction(x_values, curves, axis.get_xscale() == "log")
    axis.text(
        fraction, floor, "numerical floor",
        # x in axis fractions, y in data units, so the label rides the floor line.
        transform=axis.get_yaxis_transform(), ha="center", va="bottom",
        fontsize=mpl.rcParams["font.size"] - 1.5, color=PALETTE["floor"],
        bbox={"boxstyle": "square,pad=0.2", "facecolor": "white",
              "edgecolor": "none", "alpha": 0.85},
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

    Each panel is scaled to its own peak with a reserved band above it, and the
    time stamp is written in that band.  Fixing the label position without
    reserving the space puts it on top of the density whenever the packet is
    tall near the left wall, which is exactly the interesting case.
    """
    if width_mm is None:
        width_mm = 85.0 if len(series) <= 2 else 170.0

    n_panels = len(snapshot_indices)
    figure, axes = plt.subplots(
        n_panels, 1, figsize=(millimetres(width_mm), 0.85 * n_panels + 0.7),
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
        peak = max(float(np.max(np.abs(states[index]) ** 2)) for states in series.values())
        if peak > 0.0:
            axis.set_ylim(0.0, peak * 1.32)
        axis.set_xlim(float(positions.min()), float(positions.max()))
        axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        axis.set_ylabel(r"$|\psi|^2$")
        axis.text(
            0.012, 0.97, rf"$t={times[index]:.2f}$", transform=axis.transAxes,
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

    curves = [np.maximum(np.asarray(values, dtype=float), 1e-18) for values in series.values()]
    if floor is not None:
        _annotate_floor(axis, times, curves, floor)

    axis.set_xlabel(r"$t$")
    axis.set_ylabel(ylabel)
    _recessive_grid(axis)
    if len(series) > 1:
        # Reserve the band first: these curves rise to the right, so an
        # unreserved "best" legend lands on the end of the data.
        _headroom(axis, 0.22)
        _framed_legend(axis, loc="upper left")
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
        _annotate_floor(axis, step_sizes, [errors], floor)
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
    # The data runs corner to corner, so the upper left is the only empty region
    # and it is only large enough for a three-entry legend once it is reserved.
    _headroom(axis, 0.34)
    _framed_legend(axis, loc="upper left")
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

    Panels (a) and (c) show the same two series, so they share one legend below
    the row.  Repeating it inside each panel costs the space the curves need:
    the wall residual in (c) peaks in the upper half at both ends, leaving no
    in-axes corner that a legend can occupy without covering data.
    """
    figure, axes = plt.subplots(
        1, 3, figsize=(millimetres(width_mm), millimetres(58.0)), constrained_layout=True
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
    _panel_label(axes[0], "(a)")

    axes[1].plot(times, 1.0 - cross_fidelity, color=PALETTE["third"], linewidth=1.4)
    axes[1].set_ylabel(r"Divergence, $1-\mathcal{F}$")
    axes[1].set_ylim(-0.02, 1.02)
    _panel_label(axes[1], "(b)")

    axes[2].plot(times, dirichlet_wall, **_line("dirichlet", times), label="Dirichlet (DST-II)")
    axes[2].plot(times, periodic_wall, **_line("periodic", times), label="Periodic (FFT)")
    axes[2].set_ylabel("Wall residual")
    axes[2].set_ylim(bottom=0.0)
    _panel_label(axes[2], "(c)")

    for axis in axes:
        axis.set_xlabel(r"$t$")
        axis.set_xlim(float(times[0]), float(times[-1]))
        _recessive_grid(axis)

    handles, labels = axes[2].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncol=2,
        columnspacing=1.8, handlelength=2.4,
    )
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
    # A register cannot hold half a qubit, so only integer ticks are meaningful.
    axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    _recessive_grid(axis)
    _headroom(axis, 0.26)
    _framed_legend(axis, loc="upper left")
    return figure


def _tint(colour: str, weight: float = 0.10) -> tuple[float, float, float]:
    """Blend a palette colour towards white for a fill behind dark text."""
    return tuple(1.0 - weight * (1.0 - channel) for channel in mpl.colors.to_rgb(colour))


def plot_circuit_diagram(
    circuit, fold: int = 26, scale: float = 0.6, style: str = "iqp"
) -> plt.Figure:
    """Draw an already-transpiled circuit as single- and two-qubit gates.

    The circuit is drawn exactly as handed over: this function neither
    transpiles nor decomposes, so a diagram cannot show a different gate set
    from the one that was counted.  Qiskit sizes the canvas from the gate
    count, and the result is wider than a journal column -- a two-qubit gate
    drawn small enough to fit an 85 mm column is a gate nobody can read.  These
    are vector appendix figures and are meant to be zoomed.

    ``style`` is a Qiskit drawer style name; the default ``iqp`` is IBM's own
    scheme, which is what a reader will recognise from Quantum Composer and
    from every other Qiskit circuit they have seen.  The repository palette is
    deliberately not used here: it is built for distinguishing data series, and
    a circuit diagram has no data series to distinguish.

    ``fold`` is the number of gate columns per drawn row.
    """
    from qiskit.visualization import circuit_drawer

    return circuit_drawer(
        circuit, output="mpl", fold=fold, scale=scale, style=style,
        idle_wires=False, initial_state=False, plot_barriers=False,
    )


def plot_graphical_abstract(width_mm: float = 170.0, height_mm: float = 58.0) -> plt.Figure:
    """The method as a comparison table: what changes, and what does not.

    Every stage carries the equation that distinguishes the two boundary
    topologies, so the diagram states the argument rather than naming its
    steps.  Mathematics is set in Computer Modern through matplotlib's own
    mathtext (``math_fontfamily="cm"``), which gives LaTeX typesetting without
    making figure generation depend on a TeX installation being present.
    Setting ``usetex=True`` on these texts would hand the typesetting to a real
    LaTeX run instead, at the cost of that dependency.

    Layout
    ------
    Stages are rows and the two boundary conditions are columns, so the reader
    compares along a row rather than tracing a path.  The last two rows span
    both columns because propagation and validation are shared: the table shape
    itself carries the claim that the two runs differ in one column of inputs
    and nothing else.  A flow chart cannot say that without arrows, and the
    arrows were what made earlier versions of this figure noisy.

    Chrome is deliberately thin -- tinted bands, hairline rules between rows and
    one vertical divider, no boxes -- because in a table the alignment already
    does the grouping that borders would otherwise have to do.
    """
    figure, axis = plt.subplots(
        figsize=(millimetres(width_mm), millimetres(height_mm)), constrained_layout=True
    )
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    label_x = 0.128
    body_x0 = 0.150
    divider_x = 0.575
    columns = {"dirichlet": (body_x0, divider_x - 0.010), "periodic": (divider_x + 0.010, 1.0)}
    split_block = (0.360, 0.880)
    rows = {
        "boundary": (0.710, 0.880),
        "grid": (0.560, 0.710),
        "transform": (0.360, 0.560),
        "propagate": (0.190, 0.340),
        "validate": (0.020, 0.190),
    }
    ink = PALETTE["reference"]

    def band(x0: float, x1: float, y0: float, y1: float, colour: str, weight: float) -> None:
        axis.add_patch(
            mpl.patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, facecolor=_tint(colour, weight),
                edgecolor="none", zorder=0,
            )
        )

    def rule(x0: float, x1: float, y: float, colour: str, width: float, alpha: float = 1.0) -> None:
        axis.plot([x0, x1], [y, y], color=colour, linewidth=width, alpha=alpha,
                  solid_capstyle="butt", zorder=1)

    def cell(x0: float, x1: float, y: float, text: str, **kwargs: Any) -> None:
        axis.text(0.5 * (x0 + x1), y, text, ha="center", va="center", zorder=2, **kwargs)

    def row_label(name: str, y: float) -> None:
        axis.text(
            label_x, y, name, ha="right", va="center", fontsize=7.0,
            color=PALETTE["floor"], fontstyle="italic", zorder=2,
        )

    # Column bands run the height of the three rows that actually differ; the
    # shared rows take neutral ink so the eye reads them as common ground.
    for key, (x0, x1) in columns.items():
        band(x0, x1, *split_block, PALETTE[key], 0.09)
    band(body_x0, 1.0, rows["validate"][0], rows["propagate"][1], ink, 0.07)

    headers = {
        "dirichlet": ("DIRICHLET", "hard wall"),
        "periodic": ("PERIODIC", "ring"),
    }
    for key, (name, gloss) in headers.items():
        x0, x1 = columns[key]
        cell(x0, x1, 0.945, f"{name}  ·  {gloss}", fontsize=8.0,
             color=PALETTE[key], fontweight="bold")
        rule(x0, x1, 0.897, PALETTE[key], 1.2)

    content = {
        "boundary": (r"$\psi(0,t)=\psi(L,t)=0$", r"$\psi(x+L,t)=\psi(x,t)$"),
        "grid": (r"$x_j=(j+\frac{1}{2})\,\Delta x$", r"$x_j=x_0+j\,\Delta x$"),
    }
    for name, (dirichlet_text, periodic_text) in content.items():
        y0, y1 = rows[name]
        centre = 0.5 * (y0 + y1)
        row_label(name, centre)
        for key, text in (("dirichlet", dirichlet_text), ("periodic", periodic_text)):
            cell(*columns[key], centre, text, fontsize=9.5, color=ink, math_fontfamily="cm")

    # The transform row carries the resource cost as well as the name, since the
    # ancilla count is the practical price of the hard-wall boundary.
    y0, y1 = rows["transform"]
    row_label("transform", 0.5 * (y0 + y1))
    for key, name, cost in (
        ("dirichlet", r"$\mathcal{T}$ = DST-II / QST", "+2 ancillas"),
        ("periodic", r"$\mathcal{T}$ = DFT / QFT", "no ancillas"),
    ):
        cell(*columns[key], y0 + 0.62 * (y1 - y0), name, fontsize=9.0,
             color=ink, math_fontfamily="cm")
        cell(*columns[key], y0 + 0.26 * (y1 - y0), cost, fontsize=7.0, color=PALETTE[key],
             fontweight="semibold")

    y0, y1 = rows["propagate"]
    row_label("propagate", 0.5 * (y0 + y1))
    cell(
        body_x0, 1.0, 0.5 * (y0 + y1),
        r"$U(\Delta t)=e^{-\frac{i\Delta t}{2\hbar}V}\;\mathcal{T}^{\dagger}\,"
        r"e^{-\frac{i\hbar\Delta t}{2m}k^{2}}\,\mathcal{T}\;e^{-\frac{i\Delta t}{2\hbar}V}$",
        fontsize=10.0, color=ink, math_fontfamily="cm",
    )

    y0, y1 = rows["validate"]
    row_label("validate", 0.5 * (y0 + y1))
    cell(body_x0, 1.0, y0 + 0.66 * (y1 - y0),
         r"$1-\left|\langle\psi_{\mathrm{ref}}|\psi\rangle\right|^{2}$",
         fontsize=9.5, color=ink, math_fontfamily="cm")
    cell(body_x0, 1.0, y0 + 0.22 * (y1 - y0), "against an independent reference solution",
         fontsize=7.0, color=PALETTE["floor"])

    # Hairlines between rows; the heavier rule marks where the columns merge.
    for y in (rows["boundary"][0], rows["grid"][0], rows["validate"][1]):
        rule(body_x0, 1.0, y, PALETTE["grid"], 0.7)
    rule(body_x0, 1.0, split_block[0], ink, 1.0, alpha=0.55)
    axis.plot(
        [divider_x, divider_x], [split_block[0], split_block[1]],
        color=PALETTE["grid"], linewidth=0.7, solid_capstyle="butt", zorder=1,
    )
    return figure

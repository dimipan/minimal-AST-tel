"""Static plots for the shared AST trajectory format.

Design rules, all of which exist because the earlier plots were misleading:

* Conditions from the same substrate share axis limits. A per-condition
  autoscale makes an efficient control's SI noise look like a stall.
* Time is encoded explicitly (colour gradient, arrows, step labels), because a
  bare line cannot show dwell -- and dwell is the phenomenon.
* The evidence-geometry view fits ONE PCA basis across the union of the compared
  conditions. Separately fitted projections are not comparable. Recurrence links
  are computed in the FULL embedding space and only drawn in the projection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

TIME_CMAP = "viridis"
GAIN_EPS = 1e-9


# ---------------------------------------------------------------------------
# Shared axis limits
# ---------------------------------------------------------------------------
def substrate_limits(
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    si_threshold: float = 0.20,
) -> dict[str, float]:
    """Axis limits shared by every condition of one substrate.

    Called once per substrate and threaded into every plot, so that traces from
    different conditions are read against the same ruler.
    """
    si_values = [float(r["si"]) for records in runs.values() for r in records]
    pe_values = [float(r["pe_total"]) for records in runs.values() for r in records]
    gains = [_mean_gain(r) for records in runs.values() for r in records]
    si_max = max([*si_values, si_threshold * 1.5])
    return {
        "si_max": float(np.ceil(si_max * 10.0) / 10.0),
        "pe_max": float(max(pe_values) * 1.08) if pe_values else 1.0,
        "gain_max": float(max([*gains, 1e-3]) * 1.08),
        "si_threshold": float(si_threshold),
    }


def _mean_gain(record: Mapping[str, Any]) -> float:
    gains = record["gains"]
    return float(sum(gains.values())) / max(1, len(gains))


def _time_colours(n: int):
    return plt.get_cmap(TIME_CMAP)(np.linspace(0.0, 1.0, max(n, 2)))


def _add_time_colourbar(fig, ax, n: int, label: str = "Exchange t"):
    mappable = ScalarMappable(norm=Normalize(vmin=1, vmax=max(n, 2)), cmap=TIME_CMAP)
    bar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.10)
    bar.set_label(label)


def _dwell_sizes(records: Sequence[Mapping[str, Any]], base: float = 26.0) -> np.ndarray:
    """Marker area grows with consecutive no-gain occupancy.

    A stall is not one big point; it is many points that refuse to move. Sizing by
    the length of the current no-gain run makes that legible.
    """
    sizes, run = [], 0
    for record in records:
        run = run + 1 if _mean_gain(record) <= GAIN_EPS else 0
        sizes.append(base * (1.0 + 0.55 * run))
    return np.asarray(sizes)


def _arrows(ax, xs, ys, zs=None, colours=None, every: int = 1):
    """Direction of travel. 3-D quiver where zs is given, 2-D annotate otherwise."""
    for i in range(0, len(xs) - 1, every):
        colour = colours[i] if colours is not None else "0.4"
        if zs is None:
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.0, alpha=0.85),
            )
        else:
            ax.quiver(
                xs[i], ys[i], zs[i],
                xs[i + 1] - xs[i], ys[i + 1] - ys[i], zs[i + 1] - zs[i],
                color=colour, arrow_length_ratio=0.28, linewidth=0.9, alpha=0.85,
            )


def _si_plane(ax, xlim, ylim, threshold: float):
    """Translucent SI = threshold plane in a 3-D view whose Z axis is SI."""
    xs = np.linspace(*xlim, 2)
    ys = np.linspace(*ylim, 2)
    grid_x, grid_y = np.meshgrid(xs, ys)
    ax.plot_surface(
        grid_x, grid_y, np.full_like(grid_x, threshold),
        alpha=0.13, color="crimson", linewidth=0, shade=False, zorder=0,
    )


# ---------------------------------------------------------------------------
# 1. Per-condition time series, shared axes
# ---------------------------------------------------------------------------
def plot_trajectory(
    records: Sequence[Mapping[str, Any]],
    schema: Sequence[str],
    output_path: str | Path,
    *,
    title: str,
    limits: Mapping[str, float],
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    turns = [int(r["t"]) for r in records]
    threshold = limits["si_threshold"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    for dim in schema:
        axes[0].plot(turns, [float(r["upsilon"][dim]) for r in records], label=dim, lw=1.4)
    axes[0].set_ylabel(r"Completeness $\upsilon_i$")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(ncol=min(4, len(schema)), fontsize=8, loc="upper left")

    axes[1].plot(turns, [float(r["pe_total"]) for r in records], color="#1f77b4", lw=1.4)
    axes[1].set_ylabel(r"Total $PE^{B}$")
    axes[1].set_ylim(0.0, limits["pe_max"])

    si = [float(r["si"]) for r in records]
    axes[2].plot(turns, si, color="#333333", lw=1.5)
    axes[2].axhline(threshold, ls="--", lw=1.2, color="crimson",
                    label=rf"$\theta$ = {threshold:.2f}")
    axes[2].fill_between(turns, threshold, limits["si_max"], color="crimson", alpha=0.06)
    stalls = [int(r["t"]) for r in records if r["ground_truth"].get("stall", False)]
    if stalls:
        axes[2].scatter(stalls, [si[t - 1] for t in stalls], marker="x", s=45,
                        color="crimson", zorder=5, label="constructed stall")
    axes[2].set_ylabel(r"$SI^{\perp}$")
    axes[2].set_ylim(0.0, limits["si_max"])
    axes[2].legend(loc="upper left", fontsize=8)

    axes[3].bar(turns, [_mean_gain(r) for r in records], color="#2ca02c", width=0.7)
    axes[3].set_ylabel(r"Mean gain $\overline{\Delta\upsilon}$")
    axes[3].set_ylim(0.0, limits["gain_max"])
    axes[3].set_xlabel(r"Exchange $t$")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_scenario_comparison(
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: str | Path,
    *,
    title: str,
    limits: Mapping[str, float],
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    threshold = limits["si_threshold"]
    longest = max(len(records) for records in runs.values())
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for name, records in runs.items():
        turns = [int(r["t"]) for r in records]
        axes[0].plot(turns, [float(r["si"]) for r in records], label=name, lw=1.5)
        axes[1].plot(turns, [float(r["knowledge_mean"]) for r in records], label=name, lw=1.5)
    axes[0].axhline(threshold, ls="--", lw=1.2, color="crimson",
                    label=rf"$\theta$ = {threshold:.2f}")
    axes[0].fill_between([1, longest], threshold, limits["si_max"], color="crimson", alpha=0.06)
    axes[0].set_ylabel(r"$SI^{\perp}$")
    axes[0].set_ylim(0.0, limits["si_max"])
    axes[0].set_xlim(1, longest)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Mean completeness")
    axes[1].set_xlabel(r"Exchange $t$")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlim(1, longest)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# 2. Global AST telemetry trajectory  (was: "phase space")
# ---------------------------------------------------------------------------
def plot_telemetry_trajectory(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    title: str,
    limits: Mapping[str, float],
) -> Path:
    """Completeness x SI x PE, with time made explicit.

    Deliberately NOT called a phase portrait: the axes are telemetry channels,
    not a state and its derivative. See plot_regime_portrait for the regime view.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    knowledge = [float(r["knowledge_mean"]) for r in records]
    si = [float(r["si"]) for r in records]
    pe = [float(r["pe_total"]) for r in records]
    colours = _time_colours(len(records))
    sizes = _dwell_sizes(records)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _si_plane(ax, (0.0, 1.0), (0.0, limits["pe_max"]), limits["si_threshold"])
    ax.plot(knowledge, pe, si, color="0.6", lw=0.9, alpha=0.7, zorder=2)
    ax.scatter(knowledge, pe, si, c=colours, s=sizes, depthshade=False,
               edgecolors="0.25", linewidths=0.4, zorder=3)
    _arrows(ax, knowledge, pe, si, colours=colours, every=max(1, len(records) // 14))

    step = max(1, len(records) // 8)
    for i in range(0, len(records), step):
        ax.text(knowledge[i], pe[i], si[i], f" {records[i]['t']}", fontsize=7, color="0.25")
    ax.text(knowledge[-1], pe[-1], si[-1], f" {records[-1]['t']}", fontsize=7, color="0.25")

    ax.set_xlabel(r"Mean completeness $\overline{\upsilon}$")
    ax.set_ylabel(r"Total $PE^{B}$")
    ax.set_zlabel(r"$SI^{\perp}$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, limits["pe_max"])
    ax.set_zlim(0.0, limits["si_max"])
    ax.set_title(f"Global AST telemetry trajectory\n{title}", fontsize=11)
    ax.view_init(elev=22, azim=-125)
    _add_time_colourbar(fig, ax, len(records))
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# 3. Regime portrait: yield x friction x residual pressure
# ---------------------------------------------------------------------------
REGIME_TABLE = """Productive        gain > 0     SI low    PE varies
Unresolved stall  gain ~ 0     SI high   PE high
Saturated hold    gain ~ 0     SI low    PE low"""


def plot_regime_portrait(
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: str | Path,
    *,
    title: str,
    limits: Mapping[str, float],
) -> Path:
    """x = mean gain, y = SI, z = PE. The three regimes should occupy three corners.

    This is the view that actually discriminates: a saturated hold and an
    unresolved stall are indistinguishable on gain alone, and are separated here
    by SI and PE together.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 7.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.0], wspace=0.02)
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    markers = ["o", "^", "s", "D", "v"]

    for index, (name, records) in enumerate(runs.items()):
        gain = [_mean_gain(r) for r in records]
        si = [float(r["si"]) for r in records]
        pe = [float(r["pe_total"]) for r in records]
        colours = _time_colours(len(records))
        ax.plot(gain, si, pe, color="0.65", lw=0.7, alpha=0.5)
        ax.scatter(
            gain, si, pe, c=colours, s=_dwell_sizes(records, base=22.0),
            marker=markers[index % len(markers)], depthshade=False,
            edgecolors="0.2", linewidths=0.4, label=name,
        )

    ax.set_xlabel(r"Mean gain $\overline{\Delta\upsilon}_t$", labelpad=10)
    ax.set_ylabel(r"$SI^{\perp}_t$", labelpad=10)
    ax.set_zlabel(r"$PE^{B}_t$", labelpad=8)
    ax.set_xlim(0.0, limits["gain_max"])
    ax.set_ylim(0.0, limits["si_max"])
    ax.set_zlim(0.0, limits["pe_max"])
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.25, 1.0, 0.85))
    ax.set_title("Regime portrait", fontsize=12, pad=2)

    # Corner labels: name the three regions the equations predict.
    ax.text(0.0, limits["si_max"] * 0.92, limits["pe_max"] * 0.92,
            "unresolved\nstall", fontsize=8, color="crimson", ha="left")
    ax.text(0.0, limits["si_max"] * 0.05, limits["pe_max"] * 0.06,
            "saturated\nhold", fontsize=8, color="#1a7f37", ha="left")
    ax.text(limits["gain_max"] * 0.80, limits["si_max"] * 0.05, limits["pe_max"] * 0.55,
            "productive", fontsize=8, color="#1f77b4", ha="left")

    _add_time_colourbar(fig, ax, max(len(r) for r in runs.values()))

    side = fig.add_subplot(grid[0, 1])
    side.axis("off")
    side.text(0.0, 0.98, title, fontsize=11, va="top", weight="bold")
    side.text(
        0.0, 0.88,
        "Marker area grows with consecutive no-gain occupancy.\n"
        "Colour encodes exchange index (dark = early, bright = late).",
        fontsize=9, va="top",
    )
    side.text(0.0, 0.72, "Expected occupancy", fontsize=10, va="top", weight="bold")
    side.text(0.0, 0.65, REGIME_TABLE, fontsize=9, va="top", family="monospace")
    side.text(
        0.0, 0.40,
        "A saturated hold and an unresolved stall both sit at\n"
        "zero gain. They are separated only by SI and PE\n"
        "together -- which is the case for the telemetry pair.",
        fontsize=9, va="top",
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.06)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# 4. Evidence geometry: one shared PCA basis across the compared conditions
# ---------------------------------------------------------------------------
def plot_evidence_geometry(
    embeddings: Mapping[str, np.ndarray],
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: str | Path,
    *,
    title: str,
    recurrence_cosine: float = 0.90,
    min_lag: int = 3,
) -> tuple[Path, float]:
    """Project every evidence vector into ONE basis fitted on the union.

    Two things this gets right that a naive version does not:

    1. Recurrence is computed by cosine similarity in the FULL embedding space and
       only DRAWN in the projection. A nearest neighbour found in 2-D would be an
       artefact of the projection, not a property of the geometry.

    2. Candidates within `min_lag` exchanges are excluded. Adjacency is not
       recurrence: a smooth kinematic trajectory has cosine > 0.99 with the frame
       it just left, and counting that would make every continuous motion look
       recurrent. Recurrence means RETURNING to a region already departed.

    All panels share one basis AND one set of axis limits, so positions are
    comparable across conditions.

    Returns (path, explained_variance_ratio_of_2_components).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    names = list(embeddings)
    stacked = np.vstack([embeddings[name] for name in names])
    centre = stacked.mean(axis=0, keepdims=True)
    centred = stacked - centre
    _u, singular, vt = np.linalg.svd(centred, full_matrices=False)
    basis = vt[:2]
    variance = singular**2
    explained = float(variance[:2].sum() / variance.sum()) if variance.sum() > 0 else 0.0

    fig, axes = plt.subplots(1, len(names), figsize=(6.0 * len(names), 5.8), squeeze=False)
    for column, name in enumerate(names):
        ax = axes[0][column]
        vectors = embeddings[name]
        records = runs[name]
        projected = (vectors - centre) @ basis.T
        colours = _time_colours(len(records))
        gains = np.array([_mean_gain(r) for r in records])
        productive = gains > GAIN_EPS

        normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        n_links = 0
        for i in range(min_lag, len(vectors)):
            horizon = i - min_lag + 1          # exclude the last (min_lag - 1) exchanges
            similarity = normed[:horizon] @ normed[i]
            j = int(np.argmax(similarity))
            if similarity[j] >= recurrence_cosine:
                n_links += 1
                ax.plot(
                    [projected[i, 0], projected[j, 0]],
                    [projected[i, 1], projected[j, 1]],
                    color="crimson", lw=0.9, alpha=0.45, zorder=1,
                )

        ax.plot(projected[:, 0], projected[:, 1], color="0.7", lw=0.9, alpha=0.8, zorder=2)
        _arrows(ax, projected[:, 0], projected[:, 1], colours=colours)
        if productive.any():
            ax.scatter(
                projected[productive, 0], projected[productive, 1],
                c=colours[productive], s=70, marker="o", edgecolors="0.2",
                linewidths=0.6, zorder=3, label="gain",
            )
        if (~productive).any():
            ax.scatter(
                projected[~productive, 0], projected[~productive, 1],
                c=colours[~productive], s=115, marker="X", edgecolors="crimson",
                linewidths=1.1, zorder=3, label="no gain",
            )
        for i in range(len(projected)):
            ax.annotate(str(records[i]["t"]), projected[i], textcoords="offset points",
                        xytext=(6, 5), fontsize=7, color="0.3")
        ax.set_title(f"{name}  —  {n_links} recurrent returns", fontsize=10)
        ax.set_xlabel("PC 1")
        if column == 0:
            ax.set_ylabel("PC 2")
        ax.legend(fontsize=8, loc="best")

    # One basis is not enough: the panels must also share one viewport.
    projected_all = (stacked - centre) @ basis.T
    pad = 0.08 * max(np.ptp(projected_all[:, 0]), np.ptp(projected_all[:, 1]), 1e-6)
    xlim = (projected_all[:, 0].min() - pad, projected_all[:, 0].max() + pad)
    ylim = (projected_all[:, 1].min() - pad, projected_all[:, 1].max() + pad)
    for ax in axes[0]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Evidence geometry — {title}\n"
        f"one shared PCA basis fitted on the union of all conditions; "
        f"2 components explain {explained:.1%} of variance; all panels share one basis and one viewport.\n"
        f"Red links: cosine >= {recurrence_cosine:.2f} to an exchange at least {min_lag} steps earlier "
        f"(adjacency is not recurrence), computed in the full space, drawn in the projection.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output, explained

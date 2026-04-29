"""
poster_plots.py — per-panel selective-accuracy figures for the ICLR workshop poster.

Four independent wrappers (plot_gsm8k_phi3, plot_gsm8k_qwen, plot_olympiad_phi3,
plot_olympiad_qwen) each load their panel's data and call the core renderer.

Run as a script to write all four panels to poster/figs/ as 300-dpi PDFs:
    python poster_plots.py
"""

from __future__ import annotations

import json
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from utils import (
    add_tiebreak_noise,
    ci95,
    compute_seed_metrics,
    discover_trajectory_files,
)

# ── Constants ──────────────────────────────────────────────────────────────────

ABSTENTION_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Poster-locked palette. Fixed-position (_cs) variants share the same hex as
# their solid counterpart; linestyle="--" provides the visual distinction.
DEFAULT_COLORS: dict[str, str] = {
    "full":               "#3D2E7A",  # deep purple — THE METHOD
    "baseline":           "#378ADD",  # blue — strongest baseline
    "baseline_cs":        "#378ADD",  # same blue, dashed
    "lora_abstention":    "#2E8B57",  # sea green — distinct from threshold red
    "lora_abstention_cs": "#2E8B57",  # same green, dashed
    "self_assessment":    "#EF9F27",  # amber
    "self_assessment_cs": "#EF9F27",  # same amber, dashed
    "no_abstention":      "#888780",  # gray reference line
}

# Per-method line/marker/alpha defaults.
# fill_alpha controls the ±1 std confidence band opacity.
DEFAULT_LINE_STYLES: dict[str, dict] = {
    "full":               {"linewidth": 5, "linestyle": "-",  "marker": "o", "markersize": 8,  "alpha": 1.0, "fill_alpha": 0.15},
    "baseline":           {"linewidth": 4, "linestyle": "-",  "marker": None, "markersize": 6, "alpha": 1.0, "fill_alpha": 0.15},
    "baseline_cs":        {"linewidth": 4, "linestyle": "--", "marker": None, "markersize": 6, "alpha": 1.0, "fill_alpha": 0.15},
    "lora_abstention":    {"linewidth": 4, "linestyle": "-",  "marker": None, "markersize": 6, "alpha": 1.0, "fill_alpha": 0.15},
    "lora_abstention_cs": {"linewidth": 4, "linestyle": "--", "marker": None, "markersize": 6, "alpha": 1.0, "fill_alpha": 0.15},
    "self_assessment":    {"linewidth": 4, "linestyle": "-",  "marker": None, "markersize": 6, "alpha": 1.0, "fill_alpha": 0.15},
    "self_assessment_cs": {"linewidth": 4, "linestyle": "--", "marker": None, "markersize": 6, "alpha": 1.0, "fill_alpha": 0.15},
    "no_abstention":      {"linewidth": 1.0, "linestyle": "--", "marker": None, "markersize": 6, "alpha": 0.6, "fill_alpha": 0.08},
}

LABEL_MAP: dict[str, str] = {
    "full":               "Dynamic (Ours)",
    "baseline":           "Constant Step Probe",
    "baseline_cs":        "Constant Step Probe (k={k})",  # {k} filled at render time
    "lora_abstention":    "LoRA Abstention",
    "lora_abstention_cs": "LoRA Abstention (k={k})",
    "self_assessment":    "Self-Assessment",
    "self_assessment_cs": "Self-Assessment (k={k})",
    "no_abstention":      "No Abstention",
    "best_baseline":      "Best Baseline",
    "transfer":           "Transfer",
}


def _label(model: str, k_step: int | None) -> str:
    """Resolve LABEL_MAP entry, substituting k_step into {k} placeholders."""
    raw = LABEL_MAP.get(model, model)
    if "{k}" in raw:
        return raw.replace("{k}", str(k_step) if k_step is not None else "?")
    return raw

DATASET_TITLE_MAP: dict[str, str] = {
    "gsm8k_qwen":        "GSM8K (Qwen)",
    "gsm8k_phi3":        "GSM8K (Phi-3)",
    "olympiadMath_qwen": "OlympiadBench (Qwen)",
    "olympiadMath_phi3": "OlympiadBench (Phi-3)",
}

DEFAULT_MODEL_ORDER = [
    "full", "baseline", "lora_abstention", "self_assessment",
    "baseline_cs", "lora_abstention_cs", "self_assessment_cs",
]

# Transfer plot: teal best-baseline, terracotta transfer, same deep purple for Dynamic
TRANSFER_COLORS: dict[str, str] = {
    **DEFAULT_COLORS,
    "best_baseline": "#2CA090",  # teal
    "transfer":      "#C0552A",  # terracotta
}

TRANSFER_LINE_STYLES: dict[str, dict] = {
    "full":          {"linewidth": 5, "linestyle": "-",  "marker": "o", "markersize": 8,  "alpha": 1.0, "fill_alpha": 0.15},
    "best_baseline": {"linewidth": 4, "linestyle": "-",  "marker": None, "markersize": 6, "alpha": 1.0, "fill_alpha": 0.15},
    "transfer":      {"linewidth": 4, "linestyle": "--", "marker": None, "markersize": 6, "alpha": 1.0, "fill_alpha": 0.15},
    "no_abstention": {"linewidth": 1.0, "linestyle": "--", "marker": None, "markersize": 6, "alpha": 0.6, "fill_alpha": 0.08},
}

TRANSFER_MODEL_ORDER = ["full", "best_baseline", "transfer"]

_YTICKS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_seed_file(fpath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(fpath, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or "trajectory_value" not in df.columns:
        return pd.DataFrame()
    df["trajectory_value"] = pd.to_numeric(df["trajectory_value"], errors="coerce")
    df["output_length"] = pd.to_numeric(df["output_length"], errors="coerce")
    df = df.dropna(subset=["trajectory_value", "output_length", "should_abstain_label"])
    df = add_tiebreak_noise(df)
    return df


def _aggregate_with_std(
    seed_files: list[str],
    baselines: list[str] | None = None,
    rename_cs: bool = False,
) -> pd.DataFrame:
    """Aggregate seed files → mean ± CI ± std per (model, abstention_ratio).

    Mirrors main_results._aggregate_group but also computes acc_std / rew_std
    so the poster can render ±1 std bands (distinct from the paper's 95%-CI bands).
    """
    seed_records: list[list[dict]] = []
    for fpath in seed_files:
        df_raw = _load_seed_file(fpath)
        if df_raw.empty:
            continue
        if rename_cs:
            df_raw["model"] = df_raw["model"].map(
                lambda m: m + "_cs" if m in ["baseline", "lora_abstention", "self_assessment"] else m
            )
        seed_records.append(
            compute_seed_metrics(df_raw, ABSTENTION_RATIOS, baselines=baselines)
        )
    if not seed_records:
        return pd.DataFrame()

    all_df = pd.concat([pd.DataFrame(r) for r in seed_records], ignore_index=True)
    agg = (
        all_df.groupby(["model", "abstention_ratio"])
        .agg(
            accuracy=("accuracy", "mean"),
            acc_ci=("accuracy", ci95),
            acc_std=("accuracy", "std"),
            reward=("reward", "mean"),
            rew_ci=("reward", ci95),
            rew_std=("reward", "std"),
            saved_tokens=("saved_tokens", "mean"),
            saved_tokens_ci=("saved_tokens", ci95),
        )
        .reset_index()
    )
    agg["n_seeds"] = len(seed_records)
    agg["acc_std"] = agg["acc_std"].fillna(0.0)
    agg["rew_std"] = agg["rew_std"].fillna(0.0)
    return agg


def load_panel_data(dataset_key: str) -> pd.DataFrame:
    """Return aggregated (main + constant-step) data for one dataset/model key.

    Mirrors the combined_df construction in main_results.py but includes std columns.
    """
    main_dir = os.path.join(BASE_DIR, "traj_csvs", "main")
    main_groups = discover_trajectory_files(main_dir)
    main_files = main_groups.get(dataset_key, [])
    if not main_files:
        raise FileNotFoundError(
            f"No main trajectory files found for '{dataset_key}' in {main_dir}"
        )
    main_agg = _aggregate_with_std(main_files)
    main_agg["dataset_model_string"] = dataset_key

    cs_dir = os.path.join(BASE_DIR, "traj_csvs", "constant_step")
    if not os.path.isdir(cs_dir):
        return main_agg

    cs_groups = discover_trajectory_files(cs_dir)
    cs_files = cs_groups.get(dataset_key, [])
    if not cs_files:
        return main_agg

    cs_baselines = ["baseline_cs", "lora_abstention_cs", "self_assessment_cs"]
    cs_agg = _aggregate_with_std(cs_files, baselines=cs_baselines, rename_cs=True)
    cs_only = cs_agg[
        cs_agg["model"].isin(["baseline_cs", "lora_abstention_cs", "self_assessment_cs"])
    ].copy()
    cs_only["dataset_model_string"] = dataset_key
    return pd.concat([main_agg, cs_only], ignore_index=True)


def _with_best_baseline(data: pd.DataFrame, y_col: str = "accuracy") -> pd.DataFrame:
    """Replace all six baseline variants with a single pointwise-max 'best_baseline' row."""
    bl_models = ["baseline", "lora_abstention", "self_assessment",
                 "baseline_cs", "lora_abstention_cs", "self_assessment_cs"]
    non_bl = data[~data["model"].isin(bl_models)].copy()
    bl = data[data["model"].isin(bl_models)]
    if bl.empty:
        return non_bl
    best = bl.loc[bl.groupby("abstention_ratio")[y_col].idxmax()].copy()
    best["model"] = "best_baseline"
    return pd.concat([non_bl, best], ignore_index=True)


def load_rtp_data() -> pd.DataFrame:
    """Return aggregated data for the RTP (rtp_qwen) dataset."""
    main_dir = os.path.join(BASE_DIR, "traj_csvs", "main")
    main_groups = discover_trajectory_files(main_dir)
    rtp_files = main_groups.get("rtp_qwen", [])
    if not rtp_files:
        raise FileNotFoundError(f"No main trajectory files found for 'rtp_qwen' in {main_dir}")
    agg = _aggregate_with_std(rtp_files)
    agg["dataset_model_string"] = "rtp_qwen"
    return agg


def load_transfer_data(target_key: str) -> pd.DataFrame:
    """Return data for a transfer panel: Dynamic + Best Baseline + Transfer + no_abstention.

    Loads main + constant-step data for `target_key` to compute the best baseline,
    then loads all transfer trajectories whose target matches `target_key`.
    """
    # Main data for target
    main_dir = os.path.join(BASE_DIR, "traj_csvs", "main")
    main_groups = discover_trajectory_files(main_dir)
    main_files = main_groups.get(target_key, [])
    if not main_files:
        raise FileNotFoundError(f"No main trajectory files for '{target_key}' in {main_dir}")
    main_agg = _aggregate_with_std(main_files)

    # Constant-step data for target (optional)
    cs_agg = pd.DataFrame()
    cs_dir = os.path.join(BASE_DIR, "traj_csvs", "constant_step")
    if os.path.isdir(cs_dir):
        cs_groups = discover_trajectory_files(cs_dir)
        cs_files = cs_groups.get(target_key, [])
        if cs_files:
            _cs_baselines = ["baseline_cs", "lora_abstention_cs", "self_assessment_cs"]
            cs_agg = _aggregate_with_std(cs_files, baselines=_cs_baselines, rename_cs=True)
            cs_agg = cs_agg[cs_agg["model"].isin(_cs_baselines)]

    # Best baseline: pointwise max over all six baseline variants
    all_bl = pd.concat([main_agg, cs_agg], ignore_index=True) if not cs_agg.empty else main_agg
    combined = _with_best_baseline(all_bl)
    # Keep only the three models needed for the transfer panel
    keep = combined[combined["model"].isin(["full", "no_abstention", "best_baseline"])].copy()

    # Transfer trajectories: any source _to_ target_key
    tr_dir = os.path.join(BASE_DIR, "traj_csvs", "transfer")
    if os.path.isdir(tr_dir):
        tr_groups = discover_trajectory_files(tr_dir)
        for tr_key, tr_files in sorted(tr_groups.items()):
            if not tr_key.endswith(f"_to_{target_key}"):
                continue
            tr_agg = _aggregate_with_std(tr_files)
            tr_full = tr_agg[tr_agg["model"] == "full"].copy()
            tr_full["model"] = "transfer"
            keep = pd.concat([keep, tr_full], ignore_index=True)

    keep["dataset_model_string"] = target_key
    return keep


# ── Core renderer ──────────────────────────────────────────────────────────────

def plot_selective_accuracy(
    data: pd.DataFrame,
    ax: plt.Axes | None = None,
    *,
    colors: dict[str, str] | None = None,
    line_styles: dict[str, dict] | None = None,
    legend_mode: str = "inline",
    legend_loc: str = "upper center",
    legend_bbox_to_anchor: tuple[float, float] | None = (0.5, -0.18),
    legend_ncol: int = 2,
    show_token_savings: bool = True,
    savings_offset: float = 12.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (8, 6),
    fontsize: float | None = None,
    title: str | None = None,
    show_baseline: bool = True,
    model_order: list[str] | None = None,
    k_step: int | None = None,
    ylabel: str = "Selective Accuracy",
) -> plt.Axes:
    """Render one selective-accuracy panel.

    Parameters
    ----------
    data        : DataFrame from load_panel_data() — must have columns
                  (model, abstention_ratio, accuracy, acc_std, saved_tokens).
    ax          : Existing axes to draw on; a new figure is created when None.
    colors      : Per-method hex overrides (merged on top of DEFAULT_COLORS).
    line_styles : Per-method style overrides (merged on top of DEFAULT_LINE_STYLES).
                  Recognised keys: linewidth, linestyle, marker, markersize, alpha,
                  fill_alpha (controls the ±1 std band transparency).
    legend_mode : "inline" — annotate each line's rightmost point (poster default).
                  "box"    — compact legend box; legend_loc/bbox_to_anchor control
                             placement.
                  "none"   — suppress entirely (when panel shares a poster-level legend).
    legend_loc, legend_bbox_to_anchor, legend_ncol : used only when legend_mode="box".
    show_token_savings : Annotate the Dynamic line with per-α token-savings %.
    savings_offset     : Vertical offset (pt) for those annotations.
    xlim, ylim  : Axis limits; auto-scaled when None.
    figsize     : Figure size used when ax is None.
    fontsize    : Base font size for labels/title; inherits from rcParams when None.
    title       : Axes title text; None suppresses it (poster uses section headers).
    show_baseline : False omits baseline / baseline_cs lines.
    model_order : Rendering order (back-to-front); defaults to DEFAULT_MODEL_ORDER.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_alpha(0.0)  # transparent background for poster card surface
    ax.patch.set_alpha(0.0)

    _colors = {**DEFAULT_COLORS, **(colors or {})}

    # Deep-merge per-method style overrides
    _styles: dict[str, dict] = {m: dict(v) for m, v in DEFAULT_LINE_STYLES.items()}
    for model, overrides in (line_styles or {}).items():
        _styles.setdefault(model, {}).update(overrides)

    _model_order = list(model_order or DEFAULT_MODEL_ORDER)
    if not show_baseline:
        _model_order = [m for m in _model_order if "baseline" not in m]

    # ── no_abstention: full-width horizontal reference line ───────────────────
    _no_abs_y: float | None = None
    _no_abs_handle: plt.Line2D | None = None
    no_abs_rows = data[data["model"] == "no_abstention"]
    if len(no_abs_rows) > 0:
        _no_abs_y = float(no_abs_rows["accuracy"].mean())
        color = _colors["no_abstention"]
        _no_abs_handle = ax.axhline(
            _no_abs_y,
            linestyle="--",
            color=color,
            linewidth=1.0,
            alpha=0.6,
        )

    # ── Method lines ──────────────────────────────────────────────────────────
    line_handles: dict[str, plt.Line2D] = {}
    rightmost: dict[str, tuple[float, float]] = {}  # model → (x_last, y_last)

    for model in _model_order:
        mdf = data[data["model"] == model].sort_values("abstention_ratio")
        if len(mdf) == 0:
            continue
        sty = _styles.get(model, {"linewidth": 4, "linestyle": "-", "marker": None,
                                   "markersize": 6, "alpha": 1.0, "fill_alpha": 0.15})
        color = _colors.get(model, "black")

        (l,) = ax.plot(
            mdf["abstention_ratio"], mdf["accuracy"],
            color=color,
            linewidth=sty.get("linewidth", 4),
            linestyle=sty.get("linestyle", "-"),
            marker=sty.get("marker"),
            markersize=sty.get("markersize", 6),
            alpha=sty.get("alpha", 1.0),
        )
        # ±1 std confidence band — non-negotiable
        band = mdf["acc_std"] if "acc_std" in mdf.columns else mdf.get("acc_ci", pd.Series(0.0, index=mdf.index))
        ax.fill_between(
            mdf["abstention_ratio"],
            mdf["accuracy"] - band,
            mdf["accuracy"] + band,
            color=color,
            alpha=sty.get("fill_alpha", 0.15),
        )
        line_handles[model] = l
        rightmost[model] = (float(mdf["abstention_ratio"].iloc[-1]),
                            float(mdf["accuracy"].iloc[-1]))

        # Token-savings annotation on the Dynamic line — color matches the method
        if model == "full" and show_token_savings:
            probe = data[data["model"] == "baseline"]
            for _, row in mdf.iterrows():
                match = probe[probe["abstention_ratio"] == row["abstention_ratio"]]
                if len(match) == 0 or match["saved_tokens"].values[0] <= 0:
                    continue
                ratio_val = row["saved_tokens"] / match["saved_tokens"].values[0]
                ax.annotate(
                    f"{ratio_val:.0%}",
                    xy=(row["abstention_ratio"], row["accuracy"]),
                    xytext=(-10, savings_offset),
                    textcoords="offset points",
                    fontsize="small",
                    color=color,
                    ha="center",
                    va="bottom",
                )

    # ── Axes formatting ────────────────────────────────────────────────────────
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(min(ABSTENTION_RATIOS) - 0.1, max(ABSTENTION_RATIOS) + 0.06)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.07 * (ymax - ymin))

    # Y-axis: explicit ticks with one-decimal formatting
    visible_yticks = [y for y in _YTICKS if ax.get_ylim()[0] <= y <= ax.get_ylim()[1]]
    if visible_yticks:
        ax.set_yticks(visible_yticks)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}"))

    ax.set_xlabel("Abstention Rate", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(True, linestyle="--", alpha=0.5)

    if title is not None:
        ax.set_title(title, fontweight="bold", pad=10, fontsize=fontsize)

    # ── Legend ─────────────────────────────────────────────────────────────────
    if legend_mode == "inline" and (line_handles or _no_abs_y is not None):
        ax.figure.canvas.draw()  # finalise xlim before annotating
        x_max = ax.get_xlim()[1]
        small_fs = (fontsize * 0.8) if fontsize else "small"
        for model, (x_last, y_last) in rightmost.items():
            ax.annotate(
                _label(model, k_step),
                xy=(x_last, y_last),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=small_fs,
                color=_colors.get(model, "black"),
                ha="left",
                va="center",
                clip_on=False,
            )
        if _no_abs_y is not None:
            x_mid = (ax.get_xlim()[0] + x_max) / 2
            ax.annotate(
                "No abstention",
                xy=(x_mid, _no_abs_y),
                xytext=(0, -6),
                textcoords="offset points",
                fontsize=small_fs,
                color=_colors["no_abstention"],
                ha="center",
                va="top",
                clip_on=False,
            )

    elif legend_mode == "box" and (line_handles or _no_abs_handle is not None):
        legend_kwargs: dict = dict(
            loc=legend_loc,
            ncol=legend_ncol,
            frameon=False,
            fontsize=fontsize,
        )
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        handles = list(line_handles.values())
        labels = [_label(m, k_step) for m in line_handles]
        if _no_abs_handle is not None:
            handles.append(_no_abs_handle)
            labels.append("No abstention")
        ax.legend(handles, labels, **legend_kwargs)
    # legend_mode == "none": nothing to do

    return ax


# ── Plot description (for widget / non-matplotlib rendering) ──────────────────

def describe_selective_accuracy(
    data: pd.DataFrame,
    *,
    colors: dict[str, str] | None = None,
    line_styles: dict[str, dict] | None = None,
    legend_mode: str = "inline",
    legend_loc: str = "upper center",
    legend_bbox_to_anchor: tuple[float, float] | None = (0.5, -0.18),
    legend_ncol: int = 2,
    show_token_savings: bool = True,
    savings_offset: float = 12.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (8, 6),
    fontsize: float | None = None,
    title: str | None = None,
    show_baseline: bool = True,
    model_order: list[str] | None = None,
    k_step: int | None = None,
    ylabel: str = "Selective Accuracy",
) -> dict:
    """Return a complete JSON-serialisable description of what plot_selective_accuracy
    would draw, including all x/y data, styles, annotations, and layout parameters.
    Intended for widget renderers that cannot use matplotlib."""

    _colors = {**DEFAULT_COLORS, **(colors or {})}
    _styles: dict[str, dict] = {m: dict(v) for m, v in DEFAULT_LINE_STYLES.items()}
    for model, overrides in (line_styles or {}).items():
        _styles.setdefault(model, {}).update(overrides)

    _model_order = list(model_order or DEFAULT_MODEL_ORDER)
    if not show_baseline:
        _model_order = [m for m in _model_order if "baseline" not in m]

    # No-abstention reference line
    no_abs_y: float | None = None
    no_abs_rows = data[data["model"] == "no_abstention"]
    if len(no_abs_rows) > 0:
        no_abs_y = float(no_abs_rows["accuracy"].mean())

    no_abstention_line = None
    if no_abs_y is not None:
        no_abs_color = _colors["no_abstention"]
        no_abs_style = _styles.get("no_abstention", {})
        no_abstention_line = {
            "y": no_abs_y,
            "color": no_abs_color,
            "linewidth": no_abs_style.get("linewidth", 1.0),
            "linestyle": no_abs_style.get("linestyle", "--"),
            "alpha": no_abs_style.get("alpha", 0.6),
            "label": "No abstention",
        }

    # Axis limits
    _xlim = list(xlim) if xlim is not None else [
        min(ABSTENTION_RATIOS) - 0.1,
        max(ABSTENTION_RATIOS) + 0.06,
    ]
    # ylim requires data; compute approximate bounds and apply the +7% head-room rule
    if ylim is not None:
        _ylim = list(ylim)
    else:
        all_acc = data[data["model"] != "no_abstention"]["accuracy"]
        all_std = data[data["model"] != "no_abstention"].get("acc_std", pd.Series(dtype=float))
        y_lo = float((all_acc - all_std.fillna(0)).min()) if len(all_acc) else 0.0
        y_hi = float((all_acc + all_std.fillna(0)).max()) if len(all_acc) else 1.0
        if no_abs_y is not None:
            y_lo = min(y_lo, no_abs_y)
            y_hi = max(y_hi, no_abs_y)
        _ylim = [y_lo, y_hi + 0.07 * (y_hi - y_lo)]

    visible_yticks = [y for y in _YTICKS if _ylim[0] <= y <= _ylim[1]]

    # Series
    series = []
    for model in _model_order:
        mdf = data[data["model"] == model].sort_values("abstention_ratio")
        if len(mdf) == 0:
            continue
        sty = _styles.get(model, {"linewidth": 4, "linestyle": "-", "marker": None,
                                   "markersize": 6, "alpha": 1.0, "fill_alpha": 0.15})
        color = _colors.get(model, "black")
        has_std = "acc_std" in mdf.columns
        band_vals = mdf["acc_std"].tolist() if has_std else [0.0] * len(mdf)

        entry: dict = {
            "model": model,
            "label": _label(model, k_step),
            "color": color,
            "linewidth": sty.get("linewidth", 4),
            "linestyle": sty.get("linestyle", "-"),
            "marker": sty.get("marker"),
            "markersize": sty.get("markersize", 6),
            "alpha": sty.get("alpha", 1.0),
            "confidence_band": {
                "type": "±1 std",
                "fill_alpha": sty.get("fill_alpha", 0.15),
                "values": band_vals,
            },
            "x": mdf["abstention_ratio"].tolist(),
            "y": mdf["accuracy"].tolist(),
        }

        # Token-savings annotations on the Dynamic line
        if model == "full" and show_token_savings:
            probe = data[data["model"] == "baseline"]
            annotations = []
            for _, row in mdf.iterrows():
                match = probe[probe["abstention_ratio"] == row["abstention_ratio"]]
                if len(match) == 0 or match["saved_tokens"].values[0] <= 0:
                    continue
                ratio_val = row["saved_tokens"] / match["saved_tokens"].values[0]
                small_fs = (fontsize * 0.8) if fontsize else "small"
                annotations.append({
                    "text": f"{ratio_val:.0%}",
                    "x": float(row["abstention_ratio"]),
                    "y": float(row["accuracy"]),
                    "xytext": [-10, savings_offset],
                    "textcoords": "offset points",
                    "fontsize": small_fs,
                    "color": color,
                    "ha": "center",
                    "va": "bottom",
                })
            entry["token_savings_annotations"] = annotations

        # Inline label annotation (rightmost point)
        if legend_mode == "inline":
            x_last = float(mdf["abstention_ratio"].iloc[-1])
            y_last = float(mdf["accuracy"].iloc[-1])
            small_fs = (fontsize * 0.8) if fontsize else "small"
            entry["inline_label"] = {
                "text": _label(model, k_step),
                "x": x_last,
                "y": y_last,
                "xytext": [4, 0],
                "textcoords": "offset points",
                "fontsize": small_fs,
                "color": color,
                "ha": "left",
                "va": "center",
            }

        series.append(entry)

    # No-abstention inline annotation
    no_abs_annotation = None
    if legend_mode == "inline" and no_abs_y is not None:
        x_mid = (_xlim[0] + _xlim[1]) / 2
        small_fs = (fontsize * 0.8) if fontsize else "small"
        no_abs_annotation = {
            "text": "No abstention",
            "x": x_mid,
            "y": no_abs_y,
            "xytext": [0, -6],
            "textcoords": "offset points",
            "fontsize": small_fs,
            "color": _colors["no_abstention"],
            "ha": "center",
            "va": "top",
        }

    return {
        "title": title,
        "k_step": k_step,
        "figure": {
            "figsize": list(figsize),
            "transparent_bg": True,
        },
        "font": {
            "fontsize": fontsize,
        },
        "axes": {
            "xlabel": "Abstention Rate",
            "ylabel": ylabel,
            "xlim": _xlim,
            "ylim": _ylim,
            "yticks": visible_yticks,
            "ytick_format": "{y:.1f}",
            "grid": {"linestyle": "--", "alpha": 0.5},
        },
        "legend": {
            "mode": legend_mode,
            "loc": legend_loc,
            "bbox_to_anchor": list(legend_bbox_to_anchor) if legend_bbox_to_anchor else None,
            "ncol": legend_ncol,
        },
        "no_abstention_line": no_abstention_line,
        "no_abstention_annotation": no_abs_annotation,
        "series": series,
    }


# ── Panel wrappers ─────────────────────────────────────────────────────────────

def plot_gsm8k_phi3(**kwargs) -> plt.Axes:
    """Load GSM8K (Phi-3) data and render one panel."""
    kwargs.setdefault("k_step", 20)
    return plot_selective_accuracy(load_panel_data("gsm8k_phi3"), **kwargs)


def plot_gsm8k_qwen(**kwargs) -> plt.Axes:
    """Load GSM8K (Qwen) data and render one panel."""
    kwargs.setdefault("k_step", 20)
    return plot_selective_accuracy(load_panel_data("gsm8k_qwen"), **kwargs)


def plot_olympiad_phi3(**kwargs) -> plt.Axes:
    """Load OlympiadBench (Phi-3) data and render one panel."""
    kwargs.setdefault("k_step", 100)
    return plot_selective_accuracy(load_panel_data("olympiadMath_phi3"), **kwargs)


def plot_olympiad_qwen(**kwargs) -> plt.Axes:
    """Load OlympiadBench (Qwen) data and render one panel."""
    kwargs.setdefault("k_step", 100)
    return plot_selective_accuracy(load_panel_data("olympiadMath_qwen"), **kwargs)


# ── Transfer wrappers ──────────────────────────────────────────────────────────

def _plot_transfer(target_key: str, **kwargs) -> plt.Axes:
    kwargs.setdefault("colors", TRANSFER_COLORS)
    kwargs.setdefault("line_styles", TRANSFER_LINE_STYLES)
    kwargs.setdefault("model_order", TRANSFER_MODEL_ORDER)
    kwargs.setdefault("show_token_savings", False)
    return plot_selective_accuracy(load_transfer_data(target_key), **kwargs)


def plot_transfer_gsm8k_phi3(**kwargs) -> plt.Axes:
    return _plot_transfer("gsm8k_phi3", **kwargs)


def plot_transfer_gsm8k_qwen(**kwargs) -> plt.Axes:
    return _plot_transfer("gsm8k_qwen", **kwargs)


def plot_transfer_olympiad_phi3(**kwargs) -> plt.Axes:
    return _plot_transfer("olympiadMath_phi3", **kwargs)


def plot_transfer_olympiad_qwen(**kwargs) -> plt.Axes:
    return _plot_transfer("olympiadMath_qwen", **kwargs)


# ── RTP wrapper ────────────────────────────────────────────────────────────────

def plot_rtp_qwen(**kwargs) -> plt.Axes:
    """Load RTP (Qwen) data and render one panel."""
    kwargs.setdefault("show_token_savings", False)
    kwargs.setdefault("ylabel", "Non-Toxic Response Rate")
    return plot_selective_accuracy(load_rtp_data(), **kwargs)


# ── Cross-split calibration plot (target α vs achieved α) ─────────────────────
# Reads from figures/rebuttal/cross_split_calibration.csv (pre-computed by rebuttal_experiments.py)

_CAL_LINE_COLOR   = "#3D2E7A"  # deep purple — matches Dynamic (Ours)
_CAL_DIAG_COLOR   = "#888780"  # gray


def load_calibration_data(dataset_key: str) -> dict:
    """Return {'x': alphas, 'mean': mean_achieved, 'std': std_achieved} for one panel."""
    csv_path = os.path.join(BASE_DIR, "figures", "rebuttal", "cross_split_calibration.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"cross_split_calibration.csv not found at {csv_path}. "
            "Run rebuttal_experiments.py first."
        )
    df = pd.read_csv(csv_path)
    df = df[df["dataset_key"] == dataset_key].sort_values("alpha")
    if df.empty:
        raise ValueError(f"No calibration data for dataset '{dataset_key}'")
    return {
        "x":    df["alpha"].tolist(),
        "mean": df["mean_achieved"].tolist(),
        "std":  df["std_achieved"].tolist(),
        "mae":  float(df["mean_achieved"].sub(df["alpha"]).abs().mean()),
    }


def plot_calibration_panel(
    dataset_key: str,
    ax: plt.Axes | None = None,
    *,
    line_color: str = _CAL_LINE_COLOR,
    diag_color: str = _CAL_DIAG_COLOR,
    figsize: tuple[float, float] = (6, 6),
    fontsize: float | None = None,
    title: str | None = None,
    fill_alpha: float = 0.2,
    show_mae: bool = True,
) -> plt.Axes:
    """Render one cross-split calibration panel (target α vs achieved α)."""
    cal = load_calibration_data(dataset_key)
    x    = np.array(cal["x"])
    mean = np.array(cal["mean"])
    std  = np.array(cal["std"])

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    ax.plot([0, 1], [0, 1], linestyle="--", color=diag_color,
            linewidth=1.5, alpha=0.7, label="Perfect calibration", zorder=0)
    ax.plot(x, mean, color=line_color, linewidth=3, marker="o",
            markersize=5, label="Achieved α (mean)")
    ax.fill_between(x, mean - std, mean + std, color=line_color, alpha=fill_alpha,
                    label=f"±1 std")

    if show_mae:
        ax.text(0.05, 0.92, f"MAE = {cal['mae']:.4f}", transform=ax.transAxes,
                fontsize=fontsize or "small", color=line_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Target α", fontsize=fontsize)
    ax.set_ylabel("Achieved α on held-out half", fontsize=fontsize)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=fontsize, frameon=False)
    if title is not None:
        ax.set_title(title, fontweight="bold", pad=10, fontsize=fontsize)
    return ax


def describe_calibration_panel(
    dataset_key: str,
    *,
    line_color: str = _CAL_LINE_COLOR,
    diag_color: str = _CAL_DIAG_COLOR,
    figsize: tuple[float, float] = (6, 6),
    fontsize: float | None = None,
    title: str | None = None,
    fill_alpha: float = 0.2,
    show_mae: bool = True,
) -> dict:
    cal = load_calibration_data(dataset_key)
    return {
        "title": title,
        "figure": {"figsize": list(figsize), "transparent_bg": True},
        "font": {"fontsize": fontsize},
        "axes": {
            "xlabel": "Target α",
            "ylabel": "Achieved α on held-out half",
            "xlim": [0, 1],
            "ylim": [0, 1],
            "aspect": "equal",
            "grid": {"linestyle": "--", "alpha": 0.4},
        },
        "diagonal": {
            "label": "Perfect calibration",
            "x": [0, 1], "y": [0, 1],
            "color": diag_color,
            "linewidth": 1.5, "linestyle": "--", "alpha": 0.7,
        },
        "series": [{
            "label": "Achieved α (mean)",
            "color": line_color,
            "linewidth": 3,
            "linestyle": "-",
            "marker": "o",
            "markersize": 5,
            "confidence_band": {"type": "±1 std", "fill_alpha": fill_alpha, "values": cal["std"]},
            "x": cal["x"],
            "y": cal["mean"],
        }],
        "mae_annotation": {"value": cal["mae"], "show": show_mae},
    }


def plot_calibration_gsm8k_phi3(**kwargs) -> plt.Axes:
    return plot_calibration_panel("gsm8k_phi3", **kwargs)

def plot_calibration_gsm8k_qwen(**kwargs) -> plt.Axes:
    return plot_calibration_panel("gsm8k_qwen", **kwargs)

def plot_calibration_olympiad_phi3(**kwargs) -> plt.Axes:
    return plot_calibration_panel("olympiadMath_phi3", **kwargs)

def plot_calibration_olympiad_qwen(**kwargs) -> plt.Axes:
    return plot_calibration_panel("olympiadMath_qwen", **kwargs)


# ── Degradation plot (additive noise robustness) ───────────────────────────────
# Reads from figures/rebuttal/additive_noise.csv (pre-computed by rebuttal_experiments.py)

_DEGRADATION_NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

# Purple → gray: σ=0 matches Dynamic purple; higher σ fades to neutral gray
_DEGRADATION_COLORS = [
    "#3D2E7A",  # σ=0   — deep purple (no noise = Dynamic's true behaviour)
    "#6650A4",  # σ=0.1
    "#9577C0",  # σ=0.25
    "#B8A0CF",  # σ=0.5
    "#C8BBDA",  # σ=1.0
    "#9E9E9E",  # σ=2.0 — neutral gray (fully degraded)
]


def load_degradation_data(dataset_key: str) -> dict:
    """Return {sigma: {'x': alphas, 'mean': accs, 'std': stds}} from additive_noise.csv."""
    csv_path = os.path.join(BASE_DIR, "figures", "rebuttal", "additive_noise.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"additive_noise.csv not found at {csv_path}. "
            "Run rebuttal_experiments.py first."
        )
    df = pd.read_csv(csv_path)
    df = df[df["dataset_key"] == dataset_key].sort_values(["sigma", "alpha"])
    if df.empty:
        raise ValueError(f"No degradation data for dataset '{dataset_key}'")
    result = {}
    for sigma, grp in df.groupby("sigma"):
        result[sigma] = {
            "x":    grp["alpha"].tolist(),
            "mean": grp["accuracy_mean"].tolist(),
            "std":  grp["accuracy_std"].tolist(),
        }
    return result


def plot_degradation_panel(
    dataset_key: str,
    ax: plt.Axes | None = None,
    *,
    noise_levels: list | None = None,
    colors: list | None = None,
    figsize: tuple[float, float] = (8, 6),
    fontsize: float | None = None,
    title: str | None = None,
    fill_alpha: float = 0.15,
    legend_mode: str = "box",
    legend_loc: str = "lower left",
) -> plt.Axes:
    """Render one additive-noise degradation panel."""
    _noise_levels = noise_levels or _DEGRADATION_NOISE_LEVELS
    _colors       = colors or _DEGRADATION_COLORS
    deg_data = load_degradation_data(dataset_key)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    for j, sigma in enumerate(_noise_levels):
        if sigma not in deg_data:
            continue
        d = deg_data[sigma]
        x    = d["x"]
        mean = np.array(d["mean"])
        std  = np.array(d["std"])
        color = _colors[j] if j < len(_colors) else "gray"
        lw    = 5 if sigma == 0.0 else 3
        label = "σ=0 (original)" if sigma == 0.0 else f"σ={sigma}×std"
        ax.plot(x, mean, color=color, linewidth=lw, marker="o",
                markersize=5, label=label, alpha=1.0)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=fill_alpha)

    ax.set_xlabel("Abstention Rate", fontsize=fontsize)
    ax.set_ylabel("Selective Accuracy", fontsize=fontsize)
    ax.set_xlim(min(ABSTENTION_RATIOS) - 0.1, max(ABSTENTION_RATIOS) + 0.06)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.07 * (ymax - ymin))
    ax.grid(True, linestyle="--", alpha=0.5)
    if legend_mode == "box":
        ax.legend(loc=legend_loc, fontsize=fontsize, frameon=False)
    if title is not None:
        ax.set_title(title, fontweight="bold", pad=10, fontsize=fontsize)
    return ax


def describe_degradation_panel(
    dataset_key: str,
    *,
    noise_levels: list | None = None,
    colors: list | None = None,
    figsize: tuple[float, float] = (8, 6),
    fontsize: float | None = None,
    title: str | None = None,
    fill_alpha: float = 0.15,
    legend_mode: str = "box",
    legend_loc: str = "lower left",
) -> dict:
    _noise_levels = noise_levels or _DEGRADATION_NOISE_LEVELS
    _colors       = colors or _DEGRADATION_COLORS
    deg_data = load_degradation_data(dataset_key)

    series = []
    for j, sigma in enumerate(_noise_levels):
        if sigma not in deg_data:
            continue
        d = deg_data[sigma]
        series.append({
            "sigma": sigma,
            "label": "σ=0 (original)" if sigma == 0.0 else f"σ={sigma}×std",
            "color": _colors[j] if j < len(_colors) else "gray",
            "linewidth": 5 if sigma == 0.0 else 3,
            "linestyle": "-",
            "marker": "o",
            "markersize": 5,
            "alpha": 1.0,
            "confidence_band": {"type": "±1 std", "fill_alpha": fill_alpha, "values": d["std"]},
            "x": d["x"],
            "y": d["mean"],
        })

    return {
        "title": title,
        "figure": {"figsize": list(figsize), "transparent_bg": True},
        "font": {"fontsize": fontsize},
        "axes": {
            "xlabel": "Abstention Rate",
            "ylabel": "Selective Accuracy",
            "xlim": [min(ABSTENTION_RATIOS) - 0.1, max(ABSTENTION_RATIOS) + 0.06],
            "grid": {"linestyle": "--", "alpha": 0.5},
        },
        "legend": {"mode": legend_mode, "loc": legend_loc},
        "series": series,
    }


def plot_degradation_gsm8k_phi3(**kwargs) -> plt.Axes:
    return plot_degradation_panel("gsm8k_phi3", **kwargs)

def plot_degradation_gsm8k_qwen(**kwargs) -> plt.Axes:
    return plot_degradation_panel("gsm8k_qwen", **kwargs)

def plot_degradation_olympiad_phi3(**kwargs) -> plt.Axes:
    return plot_degradation_panel("olympiadMath_phi3", **kwargs)

def plot_degradation_olympiad_qwen(**kwargs) -> plt.Axes:
    return plot_degradation_panel("olympiadMath_qwen", **kwargs)


# ── __main__: write all four panels as PDFs ───────────────────────────────────

if __name__ == "__main__":
    import seaborn as sns

    sns.set_context("notebook", font_scale=2.5)
    sns.set_style("whitegrid")

    out_dir = os.path.join(BASE_DIR, "poster", "figs")
    os.makedirs(out_dir, exist_ok=True)

    render_kwargs = dict(figsize=(8, 6), legend_mode="inline", title=None)

    # ── Selective accuracy panels ─────────────────────────────────────────────
    acc_panels = [
        ("gsm8k_phi3",        plot_gsm8k_phi3,        20),
        ("gsm8k_qwen",        plot_gsm8k_qwen,        20),
        ("olympiadMath_phi3", plot_olympiad_phi3,     100),
        ("olympiadMath_qwen", plot_olympiad_qwen,     100),
    ]

    for key, fn, k_step in acc_panels:
        ax = fn(**render_kwargs)
        out_path = os.path.join(out_dir, f"{key}_selective_accuracy.pdf")
        ax.figure.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved: {out_path}")
        plt.close(ax.figure)

        desc = describe_selective_accuracy(
            load_panel_data(key), k_step=k_step, **render_kwargs
        )
        json_path = os.path.join(out_dir, f"{key}_selective_accuracy.json")
        with open(json_path, "w") as f:
            json.dump(desc, f, indent=2)
        print(f"Saved: {json_path}")

    # ── Transfer panels ───────────────────────────────────────────────────────
    transfer_panels = [
        ("gsm8k_phi3",        plot_transfer_gsm8k_phi3),
        ("gsm8k_qwen",        plot_transfer_gsm8k_qwen),
        ("olympiadMath_phi3", plot_transfer_olympiad_phi3),
        ("olympiadMath_qwen", plot_transfer_olympiad_qwen),
    ]

    for key, fn in transfer_panels:
        ax = fn(**render_kwargs)
        out_path = os.path.join(out_dir, f"{key}_transfer.pdf")
        ax.figure.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved: {out_path}")
        plt.close(ax.figure)

        desc = describe_selective_accuracy(
            load_transfer_data(key),
            colors=TRANSFER_COLORS, line_styles=TRANSFER_LINE_STYLES,
            model_order=TRANSFER_MODEL_ORDER, show_token_savings=False,
            **render_kwargs,
        )
        json_path = os.path.join(out_dir, f"{key}_transfer.json")
        with open(json_path, "w") as f:
            json.dump(desc, f, indent=2)
        print(f"Saved: {json_path}")

    # ── RTP panel ─────────────────────────────────────────────────────────────
    ax = plot_rtp_qwen(**render_kwargs)
    out_path = os.path.join(out_dir, "rtp_qwen_panel.pdf")
    ax.figure.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Saved: {out_path}")
    plt.close(ax.figure)

    desc = describe_selective_accuracy(
        load_rtp_data(), show_token_savings=False,
        ylabel="Non-Toxic Response Rate", **render_kwargs,
    )
    json_path = os.path.join(out_dir, "rtp_qwen_panel.json")
    with open(json_path, "w") as f:
        json.dump(desc, f, indent=2)
    print(f"Saved: {json_path}")

    # ── Calibration panels (V_0 vs V_tau) ─────────────────────────────────────
    cal_panels = [
        ("gsm8k_phi3",        plot_calibration_gsm8k_phi3),
        ("gsm8k_qwen",        plot_calibration_gsm8k_qwen),
        ("olympiadMath_phi3", plot_calibration_olympiad_phi3),
        ("olympiadMath_qwen", plot_calibration_olympiad_qwen),
    ]
    cal_kwargs = dict(figsize=(6, 6), title=None, show_mae=True)
    for key, fn in cal_panels:
        ax = fn(**cal_kwargs)
        out_path = os.path.join(out_dir, f"{key}_calibration.pdf")
        ax.figure.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved: {out_path}")
        plt.close(ax.figure)

        desc = describe_calibration_panel(key, **cal_kwargs)
        json_path = os.path.join(out_dir, f"{key}_calibration.json")
        with open(json_path, "w") as f:
            json.dump(desc, f, indent=2)
        print(f"Saved: {json_path}")

    # ── Degradation panels (additive noise robustness) ─────────────────────────
    deg_panels = [
        ("gsm8k_phi3",        plot_degradation_gsm8k_phi3),
        ("gsm8k_qwen",        plot_degradation_gsm8k_qwen),
        ("olympiadMath_phi3", plot_degradation_olympiad_phi3),
        ("olympiadMath_qwen", plot_degradation_olympiad_qwen),
    ]
    deg_kwargs = dict(figsize=(8, 6), title=None, legend_mode="box", legend_loc="lower left")
    for key, fn in deg_panels:
        ax = fn(**deg_kwargs)
        out_path = os.path.join(out_dir, f"{key}_degradation.pdf")
        ax.figure.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved: {out_path}")
        plt.close(ax.figure)

        desc = describe_degradation_panel(key, **deg_kwargs)
        json_path = os.path.join(out_dir, f"{key}_degradation.json")
        with open(json_path, "w") as f:
            json.dump(desc, f, indent=2)
        print(f"Saved: {json_path}")

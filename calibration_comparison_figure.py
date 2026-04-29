"""
Generate calibration comparison figure: Baseline (V_0) vs Dynamic (V_tau)

Compares calibration of:
1. Baseline method: value at t=0 (V_0), from prompt hidden states
2. Dynamic abstention: value at abstention time (V_tau)

For dynamic abstention, V_tau is the value when the trajectory first drops below
the threshold. Since V_tau ~ T by construction, we collect (V_tau, correctness)
pairs across 50 evenly-spaced thresholds.

Multi-seed: calibration curves are computed per seed and averaged.
Shaded regions show +/- 1 standard deviation across seeds.
"""

import os
import sys
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 12})
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import discover_trajectory_files

DATASET_TITLE_MAP = {
    "gsm8k_qwen":        "GSM8K (Qwen)",
    "gsm8k_phi3":        "GSM8K (Phi-3)",
    "olympiadMath_qwen": "OlympiadBench (Qwen)",
    "olympiadMath_phi3": "OlympiadBench (Phi-3)",
}

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR  = os.path.join(BASE_DIR, "traj_csvs", "main")
OUTPUT_PATH = os.path.join(BASE_DIR, "figures", "calibration_comparison_matrix.png")

N_BINS       = 10
N_THRESHOLDS = 50

COLORS = {
    "baseline": "#2ca02c",
    "dynamic":  "#d62728",
}


# ── Per-seed calibration helpers ──────────────────────────────────────────────

def load_seed_file(fpath):
    df = pd.read_csv(fpath, low_memory=False)
    df["trajectory_value"]   = pd.to_numeric(df["trajectory_value"],   errors="coerce")
    df["should_abstain_label"] = pd.to_numeric(df["should_abstain_label"], errors="coerce")
    df["output_length"]      = pd.to_numeric(df["output_length"],      errors="coerce")
    df = df.dropna(subset=["trajectory_value", "should_abstain_label", "output_length"])
    return df


def compute_calibration_curve(values, correctness, n_bins=N_BINS):
    """
    Returns arrays (pred, actual) of length <= n_bins, one point per occupied bin.
    Bins are spaced over the actual data range so that compressed value ranges
    (e.g. all values < 0.1 for OlympiadBench Phi-3 dynamic) still produce a
    multi-point curve suitable for interpolation.
    """
    values      = np.asarray(values,      dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    lo, hi = values.min(), values.max()
    if lo == hi:
        return np.array([lo]), np.array([correctness.mean()])
    bins = np.linspace(lo, hi, n_bins + 1)
    idx  = np.clip(np.digitize(values, bins) - 1, 0, n_bins - 1)

    pred, actual = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() > 0:
            pred.append(values[mask].mean())
            actual.append(correctness[mask].mean())
    return np.array(pred), np.array(actual)


def baseline_cal_for_seed(df):
    """Calibration of V_0 (prompt probe) for one seed file."""
    sub = df[df["model"] == "baseline"]
    if len(sub) == 0:
        return None
    values      = sub["trajectory_value"].values
    correctness = (1 - sub["should_abstain_label"]).values
    return compute_calibration_curve(values, correctness)


def dynamic_cal_for_seed(df):
    """
    Calibration of V_tau (value at first-crossing abstention time) for one seed file.
    Pools (V_tau, correctness) across N_THRESHOLDS evenly-spaced thresholds.
    """
    sub = df[df["model"] == "full"]
    if len(sub) == 0:
        return None

    # Build per-sample trajectory arrays (sorted by output_length)
    trajs       = {}
    correctness = {}
    for sid, grp in sub.groupby("sample_index"):
        grp_s = grp.sort_values("output_length")
        trajs[sid]       = grp_s["trajectory_value"].values
        correctness[sid] = 1 - int(grp_s["should_abstain_label"].values[0])

    min_vals   = np.array([t.min() for t in trajs.values()])
    thresholds = np.linspace(
        np.percentile(min_vals, 2),
        np.percentile(min_vals, 98),
        N_THRESHOLDS,
    )

    abs_values, abs_correct = [], []
    for T in thresholds:
        for sid, traj in trajs.items():
            # first position where value drops below T
            below = np.where(traj < T)[0]
            if len(below) > 0:
                abs_values.append(traj[below[0]])
                abs_correct.append(correctness[sid])

    if len(abs_values) == 0:
        return None
    return compute_calibration_curve(abs_values, abs_correct)


# ── Aggregation across seeds ───────────────────────────────────────────────────

def interpolate_to_grid(pred, actual, grid):
    """
    Linear interpolation of an (pred, actual) calibration curve onto a fixed grid.
    Points outside the pred range are NaN.
    """
    if len(pred) < 2:
        return np.full_like(grid, np.nan)
    return np.interp(grid, pred, actual, left=np.nan, right=np.nan)


def aggregate_curves(curves, n_grid=200):
    """
    Given a list of (pred, actual) pairs from multiple seeds, interpolate each
    onto a common grid and return (grid, mean, std).
    """
    # Choose grid spanning the union of all pred ranges
    all_pred = np.concatenate([p for p, _ in curves])
    grid     = np.linspace(all_pred.min(), all_pred.max(), n_grid)

    interp = np.array([
        interpolate_to_grid(p, a, grid) for p, a in curves
    ])  # shape: (n_seeds, n_grid)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(interp, axis=0)
        std  = np.nanstd(interp,  axis=0)

    # Only keep grid points where at least half the seeds have data
    valid = np.sum(~np.isnan(interp), axis=0) >= max(1, len(curves) // 2)
    return grid[valid], mean[valid], std[valid]


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_calibration_comparison(file_groups, output_path):
    datasets = sorted(file_groups.keys())
    n        = len(datasets)
    ncols    = 2
    nrows    = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 6 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for i, dataset in enumerate(datasets):
        ax       = axes[i]
        fpaths   = file_groups[dataset]
        n_seeds  = len(fpaths)

        base_curves, dyn_curves = [], []
        for fpath in fpaths:
            df = load_seed_file(fpath)
            bc = baseline_cal_for_seed(df)
            dc = dynamic_cal_for_seed(df)
            if bc is not None:
                base_curves.append(bc)
            if dc is not None:
                dyn_curves.append(dc)

        # Baseline
        if base_curves:
            gx, gm, gs = aggregate_curves(base_curves)
            ax.plot(gx, gm, "-o", color=COLORS["baseline"],
                    linewidth=2, markersize=5, label=r"Baseline ($V_0$)")
            ax.fill_between(gx, gm - gs, gm + gs,
                            color=COLORS["baseline"], alpha=0.15)

        # Dynamic
        if dyn_curves:
            gx, gm, gs = aggregate_curves(dyn_curves)
            ax.plot(gx, gm, "-s", color=COLORS["dynamic"],
                    linewidth=2, markersize=5, label=r"Dynamic ($V_\tau$)")
            ax.fill_between(gx, gm - gs, gm + gs,
                            color=COLORS["dynamic"], alpha=0.15)

        # Perfect calibration
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7, label="Perfect")

        title    = DATASET_TITLE_MAP.get(dataset, dataset)
        ci_note  = f"  (n={n_seeds})" if n_seeds > 1 else ""
        ax.set_title(title + ci_note, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Accuracy")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="lower right")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    file_groups = discover_trajectory_files(TRAJ_DIR)
    print(f"Found {len(file_groups)} dataset/model group(s):")
    for k, v in sorted(file_groups.items()):
        print(f"  {k}: {len(v)} seed(s)")
    print()

    plot_calibration_comparison(file_groups, OUTPUT_PATH)

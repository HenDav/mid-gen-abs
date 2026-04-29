"""
Rebuttal experiments for "Knowing When to Quit".

TODOs:
  1. P(incorrect | abstained) vs. abstention rate
  2. Per-question base model success rate among abstained questions
  3. Mean tokens before abstention vs. abstention rate
  4. Synthetic miscalibration (monotone bias + additive noise)
  5. Threshold calibration: cross-split transfer of T for a target abstention rate

Run:
    /Users/jesu4970/miniconda3/envs/abstention/bin/python3 rebuttal_experiments.py

All outputs saved to figures/rebuttal/.
"""

import os
import sys
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import add_tiebreak_noise

# ── Style (match main_results.py) ─────────────────────────────────────────────

sns.set_context("notebook", font_scale=2.0)
sns.set_style("whitegrid")

matplotlib.rcParams["pdf.fonttype"] = 42  # editable text in PDFs

COLORS = {
    "full": "#d62728",
    "baseline": "#2ca02c",
    "lora_abstention": "#1f77b4",
    "self_assessment": "#ff7f0e",
    "no_abstention": "gray",
}

DATASET_TITLE_MAP = {
    "gsm8k_qwen": "GSM8K (Qwen)",
    "gsm8k_phi3": "GSM8K (Phi-3)",
    "olympiadMath_qwen": "OlympiadBench (Qwen)",
    "olympiadMath_phi3": "OlympiadBench (Phi-3)",
    "rtp_qwen": "RealToxicityPrompts (Qwen)",
}

RTP_KEYS = {"rtp_qwen"}

ABSTENTION_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "rebuttal")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data loading ───────────────────────────────────────────────────────────────

def discover_trajectory_files(traj_dir):
    """Group seed CSVs by dataset_model key (same as utils.py)."""
    files = sorted(glob.glob(os.path.join(traj_dir, "*.csv")))
    groups = {}
    for f in files:
        stem = os.path.basename(f).replace("trajectory_values_", "").replace(".csv", "")
        key = re.sub(r"_seed\d+$", "", stem)
        groups.setdefault(key, []).append(f)
    return groups


def load_seed_file(fpath):
    df = pd.read_csv(fpath, low_memory=False)
    df["trajectory_value"] = pd.to_numeric(df["trajectory_value"], errors="coerce")
    df["output_length"] = pd.to_numeric(df["output_length"], errors="coerce")
    df = df.dropna(subset=["trajectory_value", "output_length", "should_abstain_label"])
    df = add_tiebreak_noise(df)
    return df


def load_full_model_data(seed_files):
    """
    Returns dict: seed_index -> DataFrame of 'full' model rows (per-token).
    """
    seed_dfs = {}
    for i, fpath in enumerate(seed_files):
        df = load_seed_file(fpath)
        full = df[df["model"] == "full"].copy().reset_index(drop=True)
        if len(full) > 0:
            seed_dfs[i] = full
    return seed_dfs


def load_all_model_data(seed_files):
    """
    Returns dict: seed_index -> full DataFrame (all models).
    """
    seed_dfs = {}
    for i, fpath in enumerate(seed_files):
        df = load_seed_file(fpath)
        if len(df) > 0:
            seed_dfs[i] = df
    return seed_dfs


# ── Core abstention utilities ──────────────────────────────────────────────────

def get_min_traj_per_sample(full_df):
    """For each sample, return the row with the minimum trajectory_value."""
    return full_df.loc[full_df.groupby("sample_index")["trajectory_value"].idxmin()].copy()


def get_threshold(min_traj, alpha):
    """Threshold T such that fraction alpha of samples are abstained (V_hat < T)."""
    # Use quantile: alpha-th quantile of min trajectory values
    return float(min_traj["trajectory_value"].quantile(alpha))


def get_abstained_mask(min_traj, T, alpha):
    """
    Boolean mask over min_traj rows indicating abstention.
    Uses exact-percentile logic to match utils.py.
    """
    target_abstain = int(round(len(min_traj) * alpha))
    # Sort ascending; bottom target_abstain are abstained
    sorted_idx = min_traj["trajectory_value"].argsort()
    mask = pd.Series(False, index=min_traj.index)
    mask.iloc[sorted_idx.iloc[:target_abstain]] = True
    return mask


def get_abstention_token(sample_df, T):
    """
    For a single sample's per-token DataFrame, return the first token_index
    where trajectory_value < T, or None if the trace is never abstained.
    """
    below = sample_df[sample_df["trajectory_value"] < T]
    if below.empty:
        return None
    return int(below.sort_values("token_index")["token_index"].iloc[0])


# ── TODO 1: P(incorrect | abstained) ──────────────────────────────────────────

BASELINE_MODELS = ["baseline", "lora_abstention", "self_assessment"]
LABEL_MAP = {
    "full": "Dynamic (Ours)",
    "baseline": "Prompt Probe",
    "lora_abstention": "LoRA Abstention",
    "self_assessment": "Self-Assessment",
}


def compute_p_incorrect_given_abstained_for_model(df, model, alphas=ABSTENTION_RATIOS):
    """
    Compute P(incorrect | abstained) per alpha for one model.
    For 'full': uses min trajectory value per sample.
    For baselines: single row per sample, uses trajectory_value directly.
    Returns DataFrame: alpha, p_incorrect, n_abstained.
    """
    if model == "full":
        rows = get_min_traj_per_sample(df[df["model"] == "full"])
    else:
        rows = df[df["model"] == model].drop_duplicates("sample_index").copy()

    if len(rows) == 0:
        return pd.DataFrame(columns=["alpha", "p_incorrect", "n_abstained"])

    records = []
    for alpha in alphas:
        T = get_threshold(rows, alpha)
        mask = get_abstained_mask(rows, T, alpha)
        abstained = rows[mask]
        p_incorrect = float(abstained["should_abstain_label"].mean()) if len(abstained) > 0 else np.nan
        records.append({"alpha": alpha, "p_incorrect": p_incorrect, "n_abstained": int(mask.sum())})
    return pd.DataFrame(records)


def run_todo1(all_seed_dfs):
    """
    Run TODO 1 across seeds for all models.
    Returns: (per_model_seed_lists, base_error_rate)
      per_model_seed_lists: dict model -> list of per-seed DataFrames
    """
    per_model = {m: [] for m in ["full"] + BASELINE_MODELS}
    base_rates = []

    for seed_idx, df in all_seed_dfs.items():
        for model in per_model:
            res = compute_p_incorrect_given_abstained_for_model(df, model)
            if len(res) > 0:
                res = res.copy()
                res["seed"] = seed_idx
                per_model[model].append(res)

        # Base rate from full model
        full_rows = get_min_traj_per_sample(df[df["model"] == "full"])
        if len(full_rows) > 0:
            base_rates.append(float(full_rows["should_abstain_label"].mean()))

    # Drop models with no data
    per_model = {m: v for m, v in per_model.items() if v}
    return per_model, float(np.mean(base_rates)) if base_rates else np.nan


# ── TODO 3: Mean tokens before abstention ─────────────────────────────────────

def compute_abstention_time_stats(full_df, alphas=ABSTENTION_RATIOS):
    """
    For each alpha, find first token_index where V_hat < T (per abstained trace).
    Returns DataFrame with: alpha, mean_tau, median_tau, std_tau,
                             mean_tau_frac, median_tau_frac, n_abstained.
    """
    min_traj = get_min_traj_per_sample(full_df)
    # Build per-sample lookup of token-sorted trajectories
    sample_groups = {
        sid: grp.sort_values("token_index")
        for sid, grp in full_df.groupby("sample_index")
    }

    records = []
    for alpha in alphas:
        T = get_threshold(min_traj, alpha)
        mask = get_abstained_mask(min_traj, T, alpha)
        abstained_sample_ids = min_traj.loc[mask, "sample_index"].values

        taus, tau_fracs = [], []
        for sid in abstained_sample_ids:
            grp = sample_groups[sid]
            tau = get_abstention_token(grp, T)
            if tau is None:
                # Should not happen for abstained samples, but guard
                continue
            total_len = int(grp["output_length"].iloc[0])
            taus.append(tau)
            tau_fracs.append(tau / max(total_len, 1))

        if len(taus) == 0:
            records.append({"alpha": alpha, "mean_tau": np.nan, "median_tau": np.nan,
                            "std_tau": np.nan, "mean_tau_frac": np.nan,
                            "median_tau_frac": np.nan, "n_abstained": 0})
            continue

        records.append({
            "alpha": alpha,
            "mean_tau": float(np.mean(taus)),
            "median_tau": float(np.median(taus)),
            "std_tau": float(np.std(taus)),
            "mean_tau_frac": float(np.mean(tau_fracs)),
            "median_tau_frac": float(np.median(tau_fracs)),
            "p25_tau_frac": float(np.percentile(tau_fracs, 25)),
            "p75_tau_frac": float(np.percentile(tau_fracs, 75)),
            "n_abstained": len(taus),
        })
    return pd.DataFrame(records)


def run_todo3(seed_dfs):
    """Run TODO 3 across seeds; return list of per-seed DataFrames."""
    results = []
    for seed_idx, full_df in seed_dfs.items():
        res = compute_abstention_time_stats(full_df)
        res["seed"] = seed_idx
        results.append(res)
    return results


# ── TODO 4: Synthetic miscalibration ──────────────────────────────────────────

def compute_selective_accuracy(full_df, alphas=ABSTENTION_RATIOS,
                                value_col="trajectory_value"):
    """
    Compute selective accuracy at each alpha using the given value column.
    Returns list of (alpha, accuracy) tuples.
    """
    min_traj = full_df.loc[
        full_df.groupby("sample_index")[value_col].idxmin()
    ][["sample_index", value_col, "should_abstain_label"]].copy()
    min_traj = min_traj.rename(columns={value_col: "min_val"})

    records = []
    for alpha in alphas:
        T = float(min_traj["min_val"].quantile(alpha))
        # exact percentile selection
        target_n = int(len(min_traj) * (1 - alpha))
        above_T = min_traj[min_traj["min_val"] > T]
        equal_T = min_traj[min_traj["min_val"] == T]
        needed = target_n - len(above_T)
        if needed > 0:
            sampled = equal_T.sample(n=min(needed, len(equal_T)), random_state=42)
            surviving = pd.concat([above_T, sampled], ignore_index=True)
        else:
            surviving = above_T.head(target_n)
        acc = float(1 - surviving["should_abstain_label"].mean()) if len(surviving) > 0 else np.nan
        records.append({"alpha": alpha, "accuracy": acc})
    return pd.DataFrame(records)


# ── Part A: Monotone transformations ──────────────────────────────────────────

MONOTONE_TRANSFORMS = {
    "$g(v)=v^2$": lambda v: v ** 2,
    "$g(v)=\\sqrt{v}$": lambda v: np.sqrt(v),
    "$g(v)=\\sigma(5(v-0.5))$": lambda v: 1 / (1 + np.exp(-5 * (v - 0.5))),
}


def apply_transform_to_full_df(full_df, transform_fn):
    """Apply a scalar function elementwise to trajectory_value, return modified df."""
    df2 = full_df.copy()
    df2["trajectory_value_transformed"] = transform_fn(df2["trajectory_value"].values)
    return df2


def run_todo4a(full_df, alphas=ABSTENTION_RATIOS):
    """Returns dict: transform_name -> accuracy DataFrame."""
    results = {}
    for name, fn in MONOTONE_TRANSFORMS.items():
        df2 = full_df.copy()
        df2["traj_transformed"] = fn(df2["trajectory_value"].values)
        # Replace trajectory_value with transformed version, recompute
        df2["trajectory_value"] = df2["traj_transformed"]
        acc_df = compute_selective_accuracy(df2, alphas)
        results[name] = acc_df
    return results


# ── Part B: Additive Gaussian noise ───────────────────────────────────────────

NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]  # multiples of std(min V̂ per sample)
N_NOISE_SEEDS = 8


def run_todo4b(full_df, alphas=ABSTENTION_RATIOS,
               noise_levels=NOISE_LEVELS, n_seeds=N_NOISE_SEEDS):
    """
    Noise levels are given as multiples of the standard deviation of
    per-sample minimum trajectory values, making them comparable across
    datasets with different value spreads.

    Returns dict: sigma_relative -> {mean: acc_df, std: acc_df}.
    Averages over n_seeds random noise instantiations.
    """
    value_std = float(full_df.groupby("sample_index")["trajectory_value"].min().std())
    rng = np.random.default_rng(42)
    results = {}
    for sigma_rel in noise_levels:
        sigma_abs = sigma_rel * value_std
        seed_accs = []
        for _ in range(n_seeds):
            df2 = full_df.copy()
            if sigma_abs > 0:
                noise = rng.normal(0, sigma_abs, size=len(df2))
                df2["trajectory_value"] = np.clip(df2["trajectory_value"] + noise, 0.0, 1.0)
            acc_df = compute_selective_accuracy(df2, alphas)
            seed_accs.append(acc_df["accuracy"].values)
        arr = np.array(seed_accs)  # shape (n_seeds, len(alphas))
        mean_acc = pd.DataFrame({"alpha": alphas, "accuracy": arr.mean(axis=0)})
        std_acc = pd.DataFrame({"alpha": alphas, "accuracy": arr.std(axis=0)})
        results[sigma_rel] = {"mean": mean_acc, "std": std_acc}
    return results


# ── Aggregation helpers ────────────────────────────────────────────────────────

def agg_seed_dfs(seed_result_list, value_col="p_incorrect"):
    """
    Given a list of per-seed DataFrames (each with 'alpha' and value_col),
    return a summary DataFrame with mean and std across seeds.
    """
    combined = pd.concat(seed_result_list, ignore_index=True)
    agg = (combined.groupby("alpha")[value_col]
           .agg(mean="mean", std="std")
           .reset_index())
    return agg


# ── Plotting helpers ───────────────────────────────────────────────────────────

def save_fig(fig, name):
    """Save figure as both PDF and PNG."""
    for ext in ["pdf", "png"]:
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def make_2x2_axes(figsize=(14, 10)):
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    return fig, axes.flatten()


# ── TODO 5: Threshold calibration reliability ─────────────────────────────────
#
# Shows that finding T for a target α is easy and reliable:
# A threshold calibrated on a random cal-split achieves the target α on a
# held-out eval-split — tight correspondence across all target values.

CALIBRATION_ALPHAS = np.round(np.arange(0.05, 1.0, 0.05), 2).tolist()  # finer grid than main


def _min_traj_values(full_df):
    """Return Series of per-sample minimum trajectory_value, indexed by sample_index."""
    return full_df.groupby("sample_index")["trajectory_value"].min()


def run_todo5_cross_split(full_df, alphas=CALIBRATION_ALPHAS, n_splits=20,
                          cal_frac=0.1, random_state=42):
    """
    Repeated random cal/eval split experiment.

    cal_frac of samples are used to find T; the remaining (1-cal_frac) are the
    eval set on which we measure the achieved abstention rate.

    For each split and each target alpha:
      - Find T = alpha-quantile of min-V̂ on cal set
      - Measure achieved_alpha = fraction of eval set with min-V̂ < T

    Returns DataFrame: alpha, mean_achieved, std_achieved, mean_error
    """
    rng = np.random.default_rng(random_state)
    min_vals = _min_traj_values(full_df)
    sample_ids = min_vals.index.values
    n_cal = max(1, int(len(sample_ids) * cal_frac))

    split_records = []  # list of dicts: {alpha, achieved_alpha, split_id}
    for split_id in range(n_splits):
        shuffled = rng.permutation(len(sample_ids))
        cal_ids = sample_ids[shuffled[:n_cal]]
        eval_ids = sample_ids[shuffled[n_cal:]]
        cal_vals = min_vals.loc[cal_ids]
        eval_vals = min_vals.loc[eval_ids]
        for alpha in alphas:
            T = float(cal_vals.quantile(alpha))
            achieved = float((eval_vals < T).mean())
            split_records.append({"alpha": alpha, "achieved_alpha": achieved, "split_id": split_id})

    df = pd.DataFrame(split_records)
    agg = df.groupby("alpha")["achieved_alpha"].agg(
        mean_achieved="mean",
        std_achieved="std",
    ).reset_index()
    agg["mean_error"] = agg["mean_achieved"] - agg["alpha"]
    return agg


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    traj_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traj_csvs", "main")
    file_groups = {}
    files = sorted(glob.glob(os.path.join(traj_dir, "*.csv")))
    for f in files:
        stem = os.path.basename(f).replace("trajectory_values_", "").replace(".csv", "")
        key = re.sub(r"_seed\d+$", "", stem)
        file_groups.setdefault(key, []).append(f)

    print(f"Found {len(file_groups)} dataset/model groups:")
    for k, v in file_groups.items():
        print(f"  {k}: {len(v)} seed(s)")

    # Ordered keys for consistent 2x2 layout (prefer Phi-3 then Qwen, GSM8K then Olympiad)
    preferred_order = [
        "gsm8k_phi3", "gsm8k_qwen",
        "olympiadMath_phi3", "olympiadMath_qwen",
    ]
    ordered_keys = list(dict.fromkeys(k for k in preferred_order if k in file_groups))
    for k in file_groups:
        if k not in ordered_keys and k not in RTP_KEYS:
            ordered_keys.append(k)
    ordered_keys_rtp = [k for k in file_groups if k in RTP_KEYS]

    # ── Collect results per dataset_key ───────────────────────────────────────

    # TODO 1
    todo1_results = {}   # key -> list of per-seed DataFrames

    # TODO 3
    todo3_results = {}   # key -> list of per-seed DataFrames
    # TODO 4A
    todo4a_results = {}  # key -> dict(transform_name -> acc_df)  (first seed only)
    # TODO 4B
    todo4b_results = {}  # key -> dict(sigma -> {mean, std})      (first seed only)
    # TODO 5
    todo5_results = {}   # key -> cross_split DataFrame

    for dataset_key in ordered_keys:
        print(f"\n=== {dataset_key} ===")
        seed_files = file_groups[dataset_key]
        all_seed_dfs = load_all_model_data(seed_files)
        seed_dfs = {i: df[df["model"] == "full"].copy().reset_index(drop=True)
                    for i, df in all_seed_dfs.items()
                    if len(df[df["model"] == "full"]) > 0}
        if not seed_dfs:
            print("  No 'full' model data found, skipping.")
            continue

        print(f"  Loaded {len(seed_dfs)} seed(s) with 'full' model data.")

        # --- TODO 1 ---
        print("  TODO 1: P(incorrect | abstained)...")
        todo1_results[dataset_key] = run_todo1(all_seed_dfs)  # (per_model_dict, base_rate)

        # --- TODO 3 ---
        print("  TODO 3: Abstention time stats...")
        todo3_results[dataset_key] = run_todo3(seed_dfs)

        # --- TODO 4A: Monotone transforms (all seeds) ---
        print("  TODO 4A: Monotone transforms...")
        todo4a_results[dataset_key] = {}
        for seed_full_df in seed_dfs.values():
            for tname, acc_df in run_todo4a(seed_full_df).items():
                todo4a_results[dataset_key].setdefault(tname, []).append(acc_df["accuracy"].values)

        # --- TODO 4B: Additive noise (all seeds) ---
        print("  TODO 4B: Additive noise...")
        todo4b_results[dataset_key] = {}
        for seed_full_df in seed_dfs.values():
            for sigma, d in run_todo4b(seed_full_df).items():
                todo4b_results[dataset_key].setdefault(sigma, []).append(d["mean"]["accuracy"].values)

        # --- TODO 5 (all seeds) ---
        print("  TODO 5: Threshold calibration reliability...")
        todo5_results[dataset_key] = [
            run_todo5_cross_split(seed_full_df) for seed_full_df in seed_dfs.values()
        ]

    # ── RTP: TODO 1 only ──────────────────────────────────────────────────────
    todo1_results_rtp = {}
    for dataset_key in ordered_keys_rtp:
        print(f"\n=== {dataset_key} (RTP) ===")
        seed_files = file_groups[dataset_key]
        all_seed_dfs = load_all_model_data(seed_files)
        print(f"  TODO 1: P(incorrect | abstained)...")
        todo1_results_rtp[dataset_key] = run_todo1(all_seed_dfs)

    print("\n=== Generating figures and CSVs ===")

    # ── TODO 1 FIGURE ─────────────────────────────────────────────────────────
    n_plots = len(ordered_keys)
    ncols = 2
    nrows = (n_plots + 1) // 2
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(9 * ncols, 6 * nrows), constrained_layout=True)
    axes1 = np.array(axes1).flatten()
    all_todo1_csv = []

    for i, dataset_key in enumerate(ordered_keys):
        if dataset_key not in todo1_results:
            axes1[i].axis("off")
            continue
        ax = axes1[i]
        per_model, base_rate = todo1_results[dataset_key]
        model_order = ["full"] + BASELINE_MODELS
        for model in model_order:
            if model not in per_model:
                continue
            seed_list = per_model[model]
            agg = agg_seed_dfs(seed_list, value_col="p_incorrect")
            lw = 2.5 if model == "full" else 2.0
            ax.plot(agg["alpha"], agg["mean"], marker="o", color=COLORS.get(model, "black"),
                    linewidth=lw, label=LABEL_MAP.get(model, model))
            if len(seed_list) > 1:
                ax.fill_between(agg["alpha"],
                                agg["mean"] - agg["std"],
                                agg["mean"] + agg["std"],
                                alpha=0.15, color=COLORS.get(model, "black"))
            # Accumulate CSV
            for seed_df in seed_list:
                seed_df_copy = seed_df.copy()
                seed_df_copy["model"] = model
                seed_df_copy["dataset_key"] = dataset_key
                all_todo1_csv.append(seed_df_copy)
        ax.axhline(base_rate, linestyle="--", color="gray", alpha=0.6,
                   label=f"Random ({base_rate:.2f})")
        ax.set_title(DATASET_TITLE_MAP.get(dataset_key, dataset_key), fontweight="bold")
        ax.set_xlabel("Abstention Rate α")
        ax.set_ylabel("P(incorrect | abstained)")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0.05, 0.95)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.legend(fontsize="x-small")
        ax.grid(True, linestyle="--", alpha=0.5)

    for j in range(len(ordered_keys), len(axes1)):
        axes1[j].axis("off")
    save_fig(fig1, "p_incorrect_given_abstained")

    if all_todo1_csv:
        pd.concat(all_todo1_csv, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "p_incorrect_given_abstained.csv"), index=False)
        print("  Saved: p_incorrect_given_abstained.csv")

    # ── RTP TODO 1 FIGURE (separate) ──────────────────────────────────────────
    if todo1_results_rtp:
        n_rtp = len(todo1_results_rtp)
        ncols_rtp = min(n_rtp, 2)
        nrows_rtp = (n_rtp + 1) // 2
        fig_rtp, axes_rtp = plt.subplots(nrows_rtp, ncols_rtp,
                                         figsize=(9 * ncols_rtp, 6 * nrows_rtp),
                                         constrained_layout=True)
        axes_rtp = np.array(axes_rtp).flatten()
        for i, dataset_key in enumerate(ordered_keys_rtp):
            if dataset_key not in todo1_results_rtp:
                axes_rtp[i].axis("off")
                continue
            ax = axes_rtp[i]
            per_model, base_rate = todo1_results_rtp[dataset_key]
            model_order = ["full"] + BASELINE_MODELS
            for model in model_order:
                if model not in per_model:
                    continue
                seed_list = per_model[model]
                agg = agg_seed_dfs(seed_list, value_col="p_incorrect")
                lw = 2.5 if model == "full" else 2.0
                ax.plot(agg["alpha"], agg["mean"], marker="o", color=COLORS.get(model, "black"),
                        linewidth=lw, label=LABEL_MAP.get(model, model))
                if len(seed_list) > 1:
                    ax.fill_between(agg["alpha"],
                                    agg["mean"] - agg["std"],
                                    agg["mean"] + agg["std"],
                                    alpha=0.15, color=COLORS.get(model, "black"))
            ax.axhline(base_rate, linestyle="--", color="gray", alpha=0.6,
                       label=f"Base rate ({base_rate:.3f})")
            ax.set_title(DATASET_TITLE_MAP.get(dataset_key, dataset_key), fontweight="bold")
            ax.set_xlabel("Abstention Rate α")
            ax.set_ylabel("P(toxic | abstained)")
            _, _yhi = ax.get_ylim()
            _yhi_data = max(_yhi, base_rate)
            ax.set_ylim(0, _yhi_data * 1.4)
            ax.set_xlim(0.05, 0.95)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.legend(fontsize="x-small")
            ax.grid(True, linestyle="--", alpha=0.5)
        for j in range(n_rtp, len(axes_rtp)):
            axes_rtp[j].axis("off")
        save_fig(fig_rtp, "p_incorrect_given_abstained_rtp")

    # ── TODO 3 FIGURE ─────────────────────────────────────────────────────────
    fig3a, axes3a = plt.subplots(nrows, ncols, figsize=(9 * ncols, 6 * nrows), constrained_layout=True)
    axes3a = np.array(axes3a).flatten()
    fig3b, axes3b = plt.subplots(nrows, ncols, figsize=(9 * ncols, 6 * nrows), constrained_layout=True)
    axes3b = np.array(axes3b).flatten()
    all_todo3_csv = []

    for i, dataset_key in enumerate(ordered_keys):
        if dataset_key not in todo3_results:
            axes3a[i].axis("off")
            axes3b[i].axis("off")
            continue
        seed_list = todo3_results[dataset_key]
        agg_tau = agg_seed_dfs(seed_list, value_col="mean_tau")
        agg_frac = agg_seed_dfs(seed_list, value_col="mean_tau_frac")
        title = DATASET_TITLE_MAP.get(dataset_key, dataset_key)

        # Panel A: mean tokens
        ax = axes3a[i]
        ax.plot(agg_tau["alpha"], agg_tau["mean"], marker="o", color=COLORS["full"], linewidth=2.5)
        if len(seed_list) > 1:
            ax.fill_between(agg_tau["alpha"],
                            agg_tau["mean"] - agg_tau["std"],
                            agg_tau["mean"] + agg_tau["std"],
                            alpha=0.2, color=COLORS["full"])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Abstention Rate α")
        ax.set_ylabel("Mean τ (tokens before abstention)")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Panel B: mean fraction
        ax = axes3b[i]
        ax.plot(agg_frac["alpha"], agg_frac["mean"], marker="o", color=COLORS["full"], linewidth=2.5)
        if len(seed_list) > 1:
            ax.fill_between(agg_frac["alpha"],
                            agg_frac["mean"] - agg_frac["std"],
                            agg_frac["mean"] + agg_frac["std"],
                            alpha=0.2, color=COLORS["full"])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Abstention Rate α")
        ax.set_ylabel("Mean τ/c (fraction of trace length)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Accumulate CSV
        for seed_df in seed_list:
            sd_copy = seed_df.copy()
            sd_copy["dataset_key"] = dataset_key
            all_todo3_csv.append(sd_copy)

    for j in range(len(ordered_keys), len(axes3a)):
        axes3a[j].axis("off")
        axes3b[j].axis("off")
    save_fig(fig3a, "mean_tokens_before_abstention")
    save_fig(fig3b, "mean_tau_fraction")

    if all_todo3_csv:
        pd.concat(all_todo3_csv, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "abstention_time_stats.csv"), index=False)
        print("  Saved: abstention_time_stats.csv")

    # ── TODO 4A FIGURE ────────────────────────────────────────────────────────
    # One illustrative dataset (phi3_olympiadMath if available, else first)
    illustrative_key = next(
        (k for k in ["olympiadMath_phi3"] if k in todo4a_results),
        ordered_keys[0])

    todo4a_csv_rows = []

    # Also make 2x2 across all datasets
    fig4a, axes4a = plt.subplots(nrows, ncols, figsize=(9 * ncols, 6 * nrows), constrained_layout=True)
    axes4a = np.array(axes4a).flatten()

    for i, dataset_key in enumerate(ordered_keys):
        if dataset_key not in todo4a_results:
            axes4a[i].axis("off")
            continue
        ax = axes4a[i]
        transform_dict = todo4a_results[dataset_key]
        colors_4a = plt.cm.tab10.colors
        for j, (tname, seed_accs) in enumerate(transform_dict.items()):
            mean_acc = np.mean(seed_accs, axis=0)
            std_acc = np.std(seed_accs, axis=0)
            lw = 2.0
            ls = "-"
            ax.plot(ABSTENTION_RATIOS, mean_acc,
                    label=tname, color=colors_4a[j], linewidth=lw, linestyle=ls, marker="o")
            if len(seed_accs) > 1:
                ax.fill_between(ABSTENTION_RATIOS, mean_acc - std_acc, mean_acc + std_acc,
                                alpha=0.15, color=colors_4a[j])
            for alpha_val, acc_val in zip(ABSTENTION_RATIOS, mean_acc):
                todo4a_csv_rows.append({
                    "dataset_key": dataset_key, "transform": tname,
                    "alpha": alpha_val, "accuracy": acc_val})
        ax.set_title(DATASET_TITLE_MAP.get(dataset_key, dataset_key), fontweight="bold")
        ax.set_xlabel("Abstention Rate α")
        ax.set_ylabel("Selective Accuracy")
        ax.legend(fontsize="x-small", loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    for j in range(len(ordered_keys), len(axes4a)):
        axes4a[j].axis("off")
    save_fig(fig4a, "monotone_bias")
    pd.DataFrame(todo4a_csv_rows).to_csv(
        os.path.join(OUT_DIR, "monotone_bias.csv"), index=False)
    print("  Saved: monotone_bias.csv")

    # ── TODO 4B FIGURE ────────────────────────────────────────────────────────
    todo4b_csv_rows = []
    fig4b, axes4b = plt.subplots(nrows, ncols, figsize=(9 * ncols, 6 * nrows), constrained_layout=True)
    axes4b = np.array(axes4b).flatten()

    noise_cmap = plt.cm.Blues
    noise_colors = [noise_cmap(0.3 + 0.7 * k / max(len(NOISE_LEVELS) - 1, 1))
                    for k in range(len(NOISE_LEVELS))]

    for i, dataset_key in enumerate(ordered_keys):
        if dataset_key not in todo4b_results:
            axes4b[i].axis("off")
            continue
        ax = axes4b[i]
        noise_dict = todo4b_results[dataset_key]
        for j, sigma in enumerate(NOISE_LEVELS):
            seed_accs = noise_dict[sigma]
            mean_acc = np.mean(seed_accs, axis=0)
            std_acc = np.std(seed_accs, axis=0)
            label = "σ=0" if sigma == 0.0 else f"σ={sigma}×std"
            lw = 3.5 if sigma == 0.0 else 2.0
            ax.plot(ABSTENTION_RATIOS, mean_acc,
                    label=label, color=noise_colors[j], linewidth=lw, marker="o")
            ax.fill_between(ABSTENTION_RATIOS,
                            mean_acc - std_acc, mean_acc + std_acc,
                            alpha=0.15, color=noise_colors[j])
            for alpha_val, acc_val, std_val in zip(ABSTENTION_RATIOS, mean_acc, std_acc):
                todo4b_csv_rows.append({
                    "dataset_key": dataset_key, "sigma": sigma,
                    "alpha": alpha_val,
                    "accuracy_mean": acc_val,
                    "accuracy_std": std_val})
        ax.set_title(DATASET_TITLE_MAP.get(dataset_key, dataset_key), fontweight="bold")
        ax.set_xlabel("Abstention Rate α")
        ax.set_ylabel("Selective Accuracy")
        ax.legend(fontsize="x-small", loc="lower left",
                  title="σ in units of\nstd(min V̂)", title_fontsize="xx-small")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    for j in range(len(ordered_keys), len(axes4b)):
        axes4b[j].axis("off")
    save_fig(fig4b, "additive_noise")
    pd.DataFrame(todo4b_csv_rows).to_csv(
        os.path.join(OUT_DIR, "additive_noise.csv"), index=False)
    print("  Saved: additive_noise.csv")

    # ── TODO 4 Combined summary figure (illustrative dataset, 2 panels) ──────────
    if illustrative_key in todo4a_results and illustrative_key in todo4b_results:
        fig_comb, axes_comb = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        # Panel 1: monotone transforms
        ax = axes_comb[0]
        for j, (tname, seed_accs) in enumerate(todo4a_results[illustrative_key].items()):
            mean_acc = np.mean(seed_accs, axis=0)
            std_acc = np.std(seed_accs, axis=0)
            lw = 2.0
            ls = "-"
            ax.plot(ABSTENTION_RATIOS, mean_acc,
                    label=tname, color=plt.cm.tab10.colors[j], linewidth=lw, linestyle=ls, marker="o")
            if len(seed_accs) > 1:
                ax.fill_between(ABSTENTION_RATIOS, mean_acc - std_acc, mean_acc + std_acc,
                                alpha=0.15, color=plt.cm.tab10.colors[j])
        ax.set_title("(A) Monotone Bias\n(curves should overlap)", fontweight="bold")
        ax.set_xlabel("Abstention Rate α")
        ax.set_ylabel("Selective Accuracy")
        ax.legend(fontsize="small")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        # Panel 2: additive noise
        ax = axes_comb[1]
        for j, sigma in enumerate(NOISE_LEVELS):
            seed_accs = todo4b_results[illustrative_key][sigma]
            mean_acc = np.mean(seed_accs, axis=0)
            std_acc = np.std(seed_accs, axis=0)
            lw = 3.5 if sigma == 0.0 else 2.0
            label_comb = "σ=0" if sigma == 0.0 else f"σ={sigma}×std"
            ax.plot(ABSTENTION_RATIOS, mean_acc,
                    label=label_comb, color=noise_colors[j], linewidth=lw, marker="o")
            ax.fill_between(ABSTENTION_RATIOS,
                            mean_acc - std_acc, mean_acc + std_acc,
                            alpha=0.15, color=noise_colors[j])
        ax.set_title("(B) Additive Noise\n(σ in units of std(min V̂))", fontweight="bold")
        ax.set_xlabel("Abstention Rate α")
        ax.set_ylabel("Selective Accuracy")
        ax.legend(fontsize="small")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        save_fig(fig_comb, "miscalibration_robustness")

    # ── Summary table: mean absolute change in selective accuracy (TODO 4) ─────
    summary_rows = []
    for dataset_key in ordered_keys:
        baseline_accs = None
        if dataset_key in todo4a_results:
            baseline_accs = None
            for tname, seed_accs in todo4a_results[dataset_key].items():
                mean_acc = np.mean(seed_accs, axis=0)
                if baseline_accs is not None:
                    delta = float(np.mean(np.abs(mean_acc - baseline_accs)))
                else:
                    delta = np.nan
                summary_rows.append({"dataset_key": dataset_key,
                                     "perturbation_type": "monotone_transform",
                                     "perturbation": tname, "mean_abs_delta_accuracy": delta})
        # 4B
        if dataset_key in todo4b_results:
            b_entry = todo4b_results[dataset_key].get(0.0, [])
            b_vals = np.mean(b_entry, axis=0) if b_entry else None
            for sigma, seed_accs in todo4b_results[dataset_key].items():
                if sigma == 0.0:
                    continue
                mean_acc = np.mean(seed_accs, axis=0)
                delta = float(np.mean(np.abs(mean_acc - b_vals))) if b_vals is not None else np.nan
                summary_rows.append({"dataset_key": dataset_key,
                                     "perturbation_type": "additive_noise",
                                     "perturbation": f"sigma={sigma}x_std",
                                     "mean_abs_delta_accuracy": delta})
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, "perturbation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Saved: {summary_csv}")

    # ── TODO 5 FIGURE: cross-split calibration transfer ──────────────────────
    fig5b, axes5b = plt.subplots(nrows, ncols, figsize=(9 * ncols, 6 * nrows), constrained_layout=True)
    axes5b = np.array(axes5b).flatten()
    todo5_csv_rows = []

    for i, dataset_key in enumerate(ordered_keys):
        if dataset_key not in todo5_results:
            axes5b[i].axis("off")
            continue
        ax = axes5b[i]
        seed_dfs_5 = todo5_results[dataset_key]
        alphas_5 = seed_dfs_5[0]["alpha"].values
        all_achieved = np.array([s["mean_achieved"].values for s in seed_dfs_5])
        mean_achieved = all_achieved.mean(axis=0)
        std_achieved = all_achieved.std(axis=0)
        mean_error = mean_achieved - alphas_5
        mae = float(np.abs(mean_error).mean())
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2,
                label="Perfect calibration", zorder=0)
        ax.plot(alphas_5, mean_achieved,
                color=COLORS["full"], linewidth=2.5, marker="o", label="Achieved α (mean)")
        ax.fill_between(alphas_5,
                        mean_achieved - std_achieved,
                        mean_achieved + std_achieved,
                        alpha=0.2, color=COLORS["full"],
                        label=f"±1 std ({len(seed_dfs_5)} seeds, 20 splits each)")
        ax.set_xlabel("Target α")
        ax.set_ylabel("Achieved α on held-out half")
        ax.set_title(DATASET_TITLE_MAP.get(dataset_key, dataset_key), fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize="small")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.text(0.05, 0.92, f"MAE = {mae:.4f}", transform=ax.transAxes,
                fontsize=11, color=COLORS["full"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        for alpha_val, achieved_val, std_val, err_val in zip(alphas_5, mean_achieved, std_achieved, mean_error):
            todo5_csv_rows.append({"dataset_key": dataset_key,
                                   "alpha": alpha_val,
                                   "mean_achieved": achieved_val,
                                   "std_achieved": std_val,
                                   "mean_error": err_val})

    for j in range(len(ordered_keys), len(axes5b)):
        axes5b[j].axis("off")
    save_fig(fig5b, "cross_split_calibration")

    if todo5_csv_rows:
        pd.DataFrame(todo5_csv_rows).to_csv(
            os.path.join(OUT_DIR, "cross_split_calibration.csv"), index=False)
        print("  Saved: cross_split_calibration.csv")

    print("\n=== All done. Outputs in figures/rebuttal/ ===")


if __name__ == "__main__":
    main()

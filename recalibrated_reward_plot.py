from utils import *
import argparse
import os
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")

ABSTENTION_RATIOS = np.linspace(0.02, 0.98, 50)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_TITLE_MAP = {
    "gsm8k_qwen": "GSM8K (Qwen)",
    "gsm8k_phi3": "GSM8K (Phi-3)",
    "phi3_gsm8k": "GSM8K (Phi-3)",
    "qwen_gsm8k": "GSM8K (Qwen)",
    "olympiadMath_qwen": "OlympiadBench (Qwen)",
    "olympiadMath_phi3": "OlympiadBench (Phi-3)",
    "phi3_olympiadMath": "OlympiadBench (Phi-3)",
    "qwen_olympiadMath": "OlympiadBench (Qwen)",
    "rtp_qwen": "RealToxicityPrompts (Qwen)",
}

COLORS = {
    "full": "#d62728",
    "baseline": "#2ca02c",
    "baseline_cs": "#2ca02c",
    "lora_abstention": "#1f77b4",
    "self_assessment": "#ff7f0e",
    "lora_abstention_cs": "#1f77b4",
    "self_assessment_cs": "#ff7f0e",
    "no_abstention": "gray",
    "transfer": "#9467bd",
}
LINESTYLES = {
    "baseline_cs": "--",
    "lora_abstention_cs": "--",
    "self_assessment_cs": "--",
    "transfer": "--",
}
LABEL_MAP = {
    "full": "Dynamic (Ours)",
    "baseline": "Constant Step Probe",
    "baseline_cs": "Constant Step Probe (Const. Step)",
    "lora_abstention": "LoRA Abstention",
    "self_assessment": "Self-Assessment",
    "lora_abstention_cs": "LoRA (Const. Step)",
    "self_assessment_cs": "Self-Assess. (Const. Step)",
    "no_abstention": "No Abstention",
    "transfer": "Transfer",
}

# ── Args ──────────────────────────────────────────────────────────────────────

args = argparse.ArgumentParser().parse_args()

# ── Analysis ──────────────────────────────────────────────────────────────────

def run_recalibrated_reward_analysis(results, baselines=None):
    """Run reward analysis with per-threshold calibration for one seed. Returns list of record dicts."""
    if baselines is None:
        baselines = ["lora_abstention", "self_assessment", "baseline"]
    records = []

    # Baselines: fit IR on all samples, predict r_bot at T
    for b in baselines:
        b_results = results[results["model"] == b].copy()
        if len(b_results) == 0:
            continue
        iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso_reg.fit(b_results["trajectory_value"].values, 1 - b_results["should_abstain_label"].values)
        thresholds = b_results["trajectory_value"].quantile(ABSTENTION_RATIOS)
        for T, ratio in zip(thresholds, ABSTENTION_RATIOS):
            r_bot = float(iso_reg.predict([T])[0])
            if r_bot >= 1 - 1e-5:
                continue
            rows_surviving = get_exact_percentile_rows(b_results, T, ratio, "trajectory_value")
            selective_accuracy = float(1 - rows_surviving["should_abstain_label"].mean())
            records.append({"model": b, "abstention_ratio": ratio,
                            "reward": selective_accuracy * (1 - ratio) + r_bot * ratio,
                            "r_bot": r_bot, "selective_accuracy": selective_accuracy})

    # Full model: IR on abstention-time values per threshold
    full_results = results[results["model"] == "full"].copy()
    if len(full_results) > 0:
        sample_trajectories = {}
        sample_correctness = {}
        for sid, group in full_results.groupby("sample_index"):
            grp = group.sort_values("output_length")
            sample_trajectories[sid] = grp["trajectory_value"].values
            sample_correctness[sid] = float(1 - grp["should_abstain_label"].iloc[0])

        min_traj = full_results.loc[full_results.groupby("sample_index")["trajectory_value"].idxmin()].copy()
        thresholds = min_traj["trajectory_value"].quantile(ABSTENTION_RATIOS)

        for T, ratio in zip(thresholds, ABSTENTION_RATIOS):
            abstention_vals, correctness_labels = [], []
            for sid, traj in sample_trajectories.items():
                v = get_abstention_time_value(traj, T)
                if v is not None:
                    abstention_vals.append(v)
                    correctness_labels.append(sample_correctness[sid])
            r_bot = compute_r_bot(abstention_vals, correctness_labels, T)
            if r_bot >= 1 - 1e-5:
                continue
            rows_surviving = get_exact_percentile_rows(min_traj, T, ratio, "trajectory_value")
            selective_accuracy = float(1 - rows_surviving["should_abstain_label"].mean())
            records.append({"model": "full", "abstention_ratio": ratio,
                            "reward": selective_accuracy * (1 - ratio) + r_bot * ratio,
                            "r_bot": r_bot, "selective_accuracy": selective_accuracy})

        base_acc = float(1 - full_results.drop_duplicates("sample_index")["should_abstain_label"].mean())
        for ratio in ABSTENTION_RATIOS:
            records.append({"model": "no_abstention", "abstention_ratio": ratio,
                            "reward": base_acc, "r_bot": None, "selective_accuracy": base_acc})

    return records


def aggregate_recalibrated(seed_records_list):
    """Aggregate per-seed record lists to mean ± 95% CI, grouped by (model, abstention_ratio)."""
    all_df = pd.concat([pd.DataFrame(r) for r in seed_records_list], ignore_index=True)
    numeric_cols = ["reward", "r_bot", "selective_accuracy"]
    agg = (
        all_df.groupby(["model", "abstention_ratio"])[numeric_cols]
        .agg(["mean", ci95])
        .reset_index()
    )
    agg.columns = ["model", "abstention_ratio",
                   "reward", "rew_ci",
                   "r_bot", "rbot_ci",
                   "selective_accuracy", "selac_ci"]
    return agg


def _load_seed_file(fpath):
    try:
        df = pd.read_csv(fpath, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or "trajectory_value" not in df.columns:
        return pd.DataFrame()
    df["trajectory_value"] = pd.to_numeric(df["trajectory_value"], errors="coerce")
    df["output_length"] = pd.to_numeric(df["output_length"], errors="coerce")
    df = df.dropna(subset=["trajectory_value", "should_abstain_label"])
    df = add_tiebreak_noise(df)
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_reward_vs_rbot_matrix(all_data_df, output_dir, filename="reward_vs_rbot_matrix.png",
                                model_order=None):
    if model_order is None:
        model_order = ["full", "baseline", "lora_abstention", "self_assessment",
                       "baseline_cs", "lora_abstention_cs", "self_assessment_cs"]

    datasets = sorted(all_data_df["dataset_model_string"].unique())
    n = len(datasets)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.375 * ncols, 2.5 * nrows), constrained_layout=True)
    axes = np.array(axes).flatten()
    line_dict = {}
    seen_models = set()
    has_cs = False

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ddf = all_data_df[all_data_df["dataset_model_string"] == dataset]
        has_ci = "rew_ci" in ddf.columns

        methods_df = ddf[(ddf["model"] != "no_abstention") & ddf["r_bot"].notna()]
        if len(methods_df) > 0:
            x_min, x_max = methods_df["r_bot"].min(), methods_df["r_bot"].max()
            no_abs_rows = ddf[ddf["model"] == "no_abstention"]
            y_no_abs = no_abs_rows["reward"].iloc[0] if len(no_abs_rows) > 0 else None
            y_min = min(methods_df["reward"].min(), y_no_abs) if y_no_abs is not None else methods_df["reward"].min()
            y_max = max(methods_df["reward"].max(), y_no_abs) if y_no_abs is not None else methods_df["reward"].max()
            diag_x = np.linspace(x_min, x_max, 100)
            l_diag, = ax.plot(diag_x, diag_x, color="black", linestyle=":", linewidth=1.1)
            if i == 0:
                line_dict["diagonal"] = (l_diag, r"Estimated $J$ = Estimated $r_\bot$")

        no_abs = ddf[ddf["model"] == "no_abstention"]
        if len(no_abs) > 0:
            l = ax.axhline(y=no_abs["reward"].iloc[0], color="gray", linestyle="--", linewidth=1.5)
            if i == 0:
                line_dict["no_abstention"] = (l, LABEL_MAP["no_abstention"])

        for model in model_order:
            mdf = ddf[(ddf["model"] == model) & ddf["r_bot"].notna()].sort_values("r_bot")
            if len(mdf) == 0:
                continue
            ls = LINESTYLES.get(model, "-")
            ax.plot(mdf["r_bot"], mdf["reward"],
                    color=COLORS.get(model, "black"), linewidth=1.9, linestyle=ls)
            if has_ci:
                ax.fill_between(mdf["r_bot"],
                                mdf["reward"] - mdf["rew_ci"], mdf["reward"] + mdf["rew_ci"],
                                color=COLORS.get(model, "black"), alpha=0.12)
            seen_models.add(model)
            if model.endswith("_cs"):
                has_cs = True

        title = DATASET_TITLE_MAP.get(dataset, dataset)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel(r"Estimated $r_\bot$")
        ax.set_ylabel("Estimated Reward")
        ax.grid(True, linestyle="--", alpha=0.5)
        if len(methods_df) > 0:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # ── Split legend: color = method, line style = decision timing ────────────
    color_method_order = [
        ("full",            "Dynamic (Ours)"),
        ("baseline",        "Constant Step Probe"),
        ("lora_abstention", "LoRA Abstention"),
        ("self_assessment", "Self-Assessment"),
        ("no_abstention",   "No Abstention"),
    ]
    color_handles = [
        Line2D([0], [0], color=COLORS[m], linewidth=1.5, linestyle="-", label=lbl)
        for m, lbl in color_method_order
        if m in seen_models or m.replace("_cs", "") in seen_models
    ]
    color_handles.append(
        Line2D([0], [0], color="black", linewidth=1.1, linestyle=":", label=r"Estimated $J$ = Estimated $r_\bot$")
    )

    legend_handles = color_handles[:]
    if has_cs:
        style_handles = [
            Line2D([0], [0], color="black", linewidth=1.5, linestyle="-",  label="Dynamic / input-processing"),
            Line2D([0], [0], color="black", linewidth=1.5, linestyle="--", label="Fixed position ($k$ tokens)"),
        ]
        legend_handles += [Line2D([0], [0], color="none")] + style_handles

    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, handlelength=2.5)

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_reward_vs_abstention_matrix(all_data_df, output_dir,
                                      filename="reward_vs_abstention_ratio_matrix.png"):
    datasets = sorted(all_data_df["dataset_model_string"].unique())
    n = len(datasets)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.375 * ncols, 2.5 * nrows), constrained_layout=True)
    axes = np.array(axes).flatten()
    line_dict = {}

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ddf = all_data_df[all_data_df["dataset_model_string"] == dataset]
        has_ci = "rew_ci" in ddf.columns

        no_abs = ddf[ddf["model"] == "no_abstention"]
        if len(no_abs) > 0:
            l = ax.axhline(y=no_abs["reward"].iloc[0], color="gray", linestyle="--", linewidth=1.5)
            if i == 0:
                line_dict["no_abstention"] = (l, LABEL_MAP["no_abstention"])

        mdf = ddf[(ddf["model"] == "full") & ddf["abstention_ratio"].notna()].sort_values("abstention_ratio")
        if len(mdf) > 0:
            l, = ax.plot(mdf["abstention_ratio"], mdf["reward"],
                         color=COLORS["full"], linewidth=1.9)
            if has_ci:
                ax.fill_between(mdf["abstention_ratio"],
                                mdf["reward"] - mdf["rew_ci"], mdf["reward"] + mdf["rew_ci"],
                                color=COLORS["full"], alpha=0.12)
            if i == 0:
                line_dict["full"] = (l, LABEL_MAP["full"])

        title = DATASET_TITLE_MAP.get(dataset, dataset)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Reward")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(0.0, 1.0)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    legend_order = ["full", "no_abstention"]
    lines = [line_dict[k][0] for k in legend_order if k in line_dict]
    labels = [line_dict[k][1] for k in legend_order if k in line_dict]
    fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

print("Recalibration Analysis (Per-Threshold Calibration, r_bot x-axis)")
print("=" * 50)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
traj_dir = os.path.join(BASE_DIR, "traj_csvs", "main")
file_groups = discover_trajectory_files(traj_dir)

print(f"Found {len(file_groups)} dataset/model group(s):")
for key, files in file_groups.items():
    print(f"  {key}: {len(files)} seed(s)")
print()

RTP_KEYS = {"rtp_qwen"}

all_group_dfs = []
for dataset_model_string, seed_files in file_groups.items():
    if dataset_model_string in RTP_KEYS:
        continue
    print(f"Processing {dataset_model_string} ({len(seed_files)} seed(s))...")
    seed_records = [run_recalibrated_reward_analysis(_load_seed_file(f)) for f in seed_files]
    agg = aggregate_recalibrated(seed_records)
    agg["dataset_model_string"] = dataset_model_string
    agg["n_seeds"] = len(seed_records)
    all_group_dfs.append(agg)

final_df = pd.concat(all_group_dfs, ignore_index=True) if all_group_dfs else pd.DataFrame()
if not final_df.empty:
    plot_reward_vs_rbot_matrix(final_df, OUTPUT_DIR)
    plot_reward_vs_abstention_matrix(final_df, OUTPUT_DIR)

# ── Constant-step comparison (optional) ──────────────────────────────────────

cs_dir = os.path.join(BASE_DIR, "traj_csvs", "constant_step")
cs_group_dfs = []
if os.path.isdir(cs_dir):
    cs_groups = discover_trajectory_files(cs_dir)
    for cs_key, cs_files in cs_groups.items():
        print(f"\nConstant-step: {cs_key} ({len(cs_files)} seed(s))...")
        cs_baselines = ["baseline_cs", "lora_abstention_cs", "self_assessment_cs"]
        cs_seed_records = []
        for fpath in cs_files:
            df_raw = _load_seed_file(fpath)
            if df_raw.empty:
                continue
            df_raw["model"] = df_raw["model"].map(
                lambda m: m + "_cs" if m in ["baseline", "lora_abstention", "self_assessment"] else m
            )
            cs_seed_records.append(run_recalibrated_reward_analysis(df_raw, baselines=cs_baselines))
        if not cs_seed_records:
            continue
        cs_agg = aggregate_recalibrated(cs_seed_records)
        cs_agg["n_seeds"] = len(cs_seed_records)

        dyn_files = file_groups.get(cs_key, [])
        if dyn_files:
            print(f"  Merging with dynamic data: {cs_key} ({len(dyn_files)} seed(s))...")
            dyn_seed_records = [run_recalibrated_reward_analysis(_load_seed_file(f)) for f in dyn_files]
            dyn_agg = aggregate_recalibrated(dyn_seed_records)
            dyn_agg = dyn_agg[dyn_agg["model"].isin(["full", "baseline", "no_abstention"])].copy()
            dyn_agg["n_seeds"] = len(dyn_seed_records)
            combined = pd.concat([dyn_agg, cs_agg], ignore_index=True)
        else:
            combined = cs_agg

        combined["dataset_model_string"] = cs_key
        cs_group_dfs.append(combined)

if cs_group_dfs and all_group_dfs:
    cs_only = pd.concat(cs_group_dfs, ignore_index=True)
    cs_only = cs_only[cs_only["model"].isin(["baseline_cs", "lora_abstention_cs", "self_assessment_cs"])].copy()
    combined_df = pd.concat([final_df, cs_only], ignore_index=True)
    plot_reward_vs_rbot_matrix(
        combined_df, OUTPUT_DIR,
        filename="combined_rbot_matrix.png",
    )

# ── Cross-domain transfer ─────────────────────────────────────────────────────

transfer_dir = os.path.join(BASE_DIR, "traj_csvs", "transfer")
if os.path.isdir(transfer_dir):
    tr_groups = discover_trajectory_files(transfer_dir)
    target_agg_map = {}
    for tr_key, tr_files in tr_groups.items():
        m = re.match(r"^(.+)_to_(.+)$", tr_key)
        if not m:
            print(f"WARNING: could not parse transfer key '{tr_key}', skipping")
            continue
        source, target = m.group(1), m.group(2)
        print(f"\nTransfer: {source} → {target} ({len(tr_files)} seed(s))...")
        seed_records = []
        for fpath in tr_files:
            df_raw = _load_seed_file(fpath)
            records = run_recalibrated_reward_analysis(df_raw)
            renamed = [dict(r, model="transfer") for r in records if r["model"] == "full"]
            seed_records.append(renamed)
        if not seed_records:
            continue
        tr_agg = aggregate_recalibrated(seed_records)
        tr_agg["n_seeds"] = len(seed_records)
        target_agg_map.setdefault(target, []).append(tr_agg)

    tr_combined_dfs = []
    for target, tr_agg_list in target_agg_map.items():
        main_target = final_df[final_df["dataset_model_string"] == target] if all_group_dfs else pd.DataFrame()
        pieces = [main_target] + [a.assign(dataset_model_string=target) for a in tr_agg_list]
        tr_combined_dfs.append(pd.concat(pieces, ignore_index=True))

    if tr_combined_dfs:
        tr_df = pd.concat(tr_combined_dfs, ignore_index=True)
        plot_reward_vs_rbot_matrix(
            tr_df, OUTPUT_DIR,
            filename="transfer_rbot_matrix.png",
            model_order=["full", "baseline", "lora_abstention", "self_assessment", "transfer"],
        )

print(f"\nAnalysis complete. Results in: {OUTPUT_DIR}")

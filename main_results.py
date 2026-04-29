from utils import *
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")

ABSTENTION_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    "best_baseline": "#2ca02c",
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
    "best_baseline": "Best Baseline",
}
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

# ── Args ──────────────────────────────────────────────────────────────────────

args = argparse.ArgumentParser().parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_seed_file(fpath):
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


def _aggregate_group(seed_files, baselines=None, rename_cs=False):
    """Load seed files, optionally rename models to _cs, return aggregated metrics DataFrame."""
    seed_records = []
    for fpath in seed_files:
        df_raw = _load_seed_file(fpath)
        if df_raw.empty:
            continue
        if rename_cs:
            df_raw["model"] = df_raw["model"].map(
                lambda m: m + "_cs" if m in ["baseline", "lora_abstention", "self_assessment"] else m
            )
        seed_records.append(compute_seed_metrics(df_raw, ABSTENTION_RATIOS, baselines=baselines))
    if not seed_records:
        return pd.DataFrame()
    agg = aggregate_seed_metrics(seed_records)
    agg["n_seeds"] = len(seed_records)
    return agg


# ── Load and aggregate: standard dynamic experiment ───────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
traj_dir = os.path.join(BASE_DIR, "traj_csvs", "main")
file_groups = discover_trajectory_files(traj_dir)

print(f"Found {len(file_groups)} dataset/model group(s):")
for key, files in file_groups.items():
    print(f"  {key}: {len(files)} seed(s)")
print()

RTP_KEYS = {"rtp_qwen"}

all_group_dfs = []
rtp_group_dfs = []
for dataset_model_string, seed_files in file_groups.items():
    print(f"Processing {dataset_model_string} ({len(seed_files)} seed(s))...")
    agg = _aggregate_group(seed_files)
    agg["dataset_model_string"] = dataset_model_string
    if dataset_model_string in RTP_KEYS:
        rtp_group_dfs.append(agg)
    else:
        all_group_dfs.append(agg)

df = pd.concat(all_group_dfs, ignore_index=True) if all_group_dfs else pd.DataFrame()
rtp_df = pd.concat(rtp_group_dfs, ignore_index=True) if rtp_group_dfs else pd.DataFrame()

# ── Load and aggregate: constant-step comparison (optional) ───────────────────

cs_df = pd.DataFrame()
cs_dir = os.path.join(BASE_DIR, "traj_csvs", "constant_step")
if os.path.isdir(cs_dir):
    cs_groups = discover_trajectory_files(cs_dir)
    for cs_key, cs_files in cs_groups.items():
        print(f"\nConstant-step: {cs_key} ({len(cs_files)} seed(s))...")
        cs_baselines = ["baseline_cs", "lora_abstention_cs", "self_assessment_cs"]
        cs_agg = _aggregate_group(cs_files, baselines=cs_baselines, rename_cs=True)
        if cs_agg.empty:
            continue

        dyn_files = file_groups.get(cs_key, [])
        if dyn_files:
            print(f"  Merging with dynamic data: {cs_key} ({len(dyn_files)} seed(s))...")
            dyn_seed_records = []
            for fpath in dyn_files:
                df_raw = _load_seed_file(fpath)
                dyn_seed_records.append(compute_seed_metrics(df_raw, ABSTENTION_RATIOS))
            dyn_agg = aggregate_seed_metrics(dyn_seed_records)
            dyn_agg = dyn_agg[dyn_agg["model"].isin(["full", "baseline", "no_abstention"])].copy()
            dyn_agg["n_seeds"] = len(dyn_seed_records)
            combined = pd.concat([dyn_agg, cs_agg], ignore_index=True)
        else:
            combined = cs_agg

        combined["dataset_model_string"] = cs_key
        cs_df = pd.concat([cs_df, combined], ignore_index=True)

# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_matrix(plot_df, y_col, ci_col, ylabel, filename,
                 model_order=None, ylim=None, xlim=None,
                 ytick_format="%.2f", show_panel_title=True, legend_side=False,
                 legend_ncol=4, legend_y_offset=-0.20):
    if model_order is None:
        model_order = ["full", "baseline", "lora_abstention", "self_assessment",
                       "baseline_cs", "lora_abstention_cs", "self_assessment_cs"]

    datasets = sorted(plot_df["dataset_model_string"].unique())
    n = len(datasets)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.375 * ncols, 2.5 * nrows), constrained_layout=True)
    axes = np.array(axes).flatten()

    # Track which models and line styles actually appear in the data
    seen_models = set()
    has_cs = False

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        ax.yaxis.set_major_formatter(FormatStrFormatter(ytick_format))
        ddf = plot_df[plot_df["dataset_model_string"] == dataset]

        no_abs = ddf[ddf["model"] == "no_abstention"]
        if len(no_abs) > 0:
            ax.plot(no_abs["abstention_ratio"], no_abs[y_col],
                    linestyle="--", color=COLORS["no_abstention"], alpha=0.7, linewidth=1.5)
            ax.fill_between(no_abs["abstention_ratio"],
                            no_abs[y_col] - no_abs[ci_col], no_abs[y_col] + no_abs[ci_col],
                            color=COLORS["no_abstention"], alpha=0.12)
            seen_models.add("no_abstention")

        for model in model_order:
            mdf = ddf[ddf["model"] == model]
            if len(mdf) == 0:
                continue
            ls = LINESTYLES.get(model, "-")
            lw = 1.9 if model == "full" else 1.5
            mk = "o" if model == "full" else None
            ax.plot(mdf["abstention_ratio"], mdf[y_col],
                    color=COLORS.get(model, "black"), linewidth=lw,
                    linestyle=ls, marker=mk, markersize=3)
            ax.fill_between(mdf["abstention_ratio"],
                            mdf[y_col] - mdf[ci_col], mdf[y_col] + mdf[ci_col],
                            color=COLORS.get(model, "black"), alpha=0.12)
            seen_models.add(model)
            if model.endswith("_cs"):
                has_cs = True

            # Token-savings annotation for full model (accuracy plot only)
            if model == "full" and y_col == "accuracy":
                probe = ddf[ddf["model"] == "baseline"]
                for _, row in mdf.iterrows():
                    match = probe[probe["abstention_ratio"] == row["abstention_ratio"]]
                    if len(match) > 0 and match["saved_tokens"].values[0] > 0:
                        ratio_val = row["saved_tokens"] / match["saved_tokens"].values[0]
                        ax.annotate(f"{ratio_val:.0%}",
                                    xy=(row["abstention_ratio"], row[y_col]),
                                    xytext=(-4, 5), textcoords="offset points",
                                    fontsize="small", color=COLORS["full"], ha="center", va="bottom")

        if show_panel_title:
            title = DATASET_TITLE_MAP.get(dataset, dataset)
            ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel("Abstention Rate")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(min(ABSTENTION_RATIOS) - 0.1, max(ABSTENTION_RATIOS) + 0.06)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + 0.07 * (ymax - ymin))

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # ── Split legend: color = method, line style = decision timing ────────────
    # Method color legend (use solid lines for all to show color only)
    color_method_order = [
        ("full",            "Dynamic (Ours)"),
        ("best_baseline",   "Best Baseline"),
        ("baseline",        "Constant Step Probe"),
        ("lora_abstention", "LoRA Abstention"),
        ("self_assessment", "Self-Assessment"),
        ("no_abstention",   "No Abstention"),
        ("transfer",        "Transfer"),
    ]
    color_handles = [
        Line2D([0], [0], color=COLORS[m], linewidth=1.5,
               linestyle="-", marker=("o" if m == "full" else None),
               markersize=3, label=lbl)
        for m, lbl in color_method_order
        if m in seen_models or m.replace("_cs", "") in seen_models
    ]

    legend_handles = color_handles[:]

    # Line style legend (only when constant-step data is present)
    if has_cs:
        style_handles = [
            Line2D([0], [0], color="black", linewidth=1.5, linestyle="-",  label="Dynamic / input-processing"),
            Line2D([0], [0], color="black", linewidth=1.5, linestyle="--", label="Fixed position ($k$ tokens)"),
        ]
        legend_handles += [Line2D([0], [0], color="none")] + style_handles  # spacer

    if legend_side:
        fig.legend(handles=legend_handles, loc="center left",
                   bbox_to_anchor=(1.0, 0.5), ncol=2, frameon=False,
                   handlelength=2.5)
    else:
        legend_y = legend_y_offset / nrows
        fig.legend(handles=legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, legend_y), ncol=legend_ncol, frameon=False,
                   handlelength=2.5)

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


if not df.empty:
    _plot_matrix(df, y_col="accuracy", ci_col="acc_ci",
                 ylabel="Selective Accuracy", filename="main_results_matrix.png")
    _plot_matrix(df, y_col="reward", ci_col="rew_ci",
                 ylabel="Reward  J = (1-α)·S + α·r_bot", filename="reward_matrix.png")

if not rtp_df.empty:
    _rtp_acc_min = float(rtp_df["accuracy"].min())
    _rtp_acc_max = float(rtp_df["accuracy"].max())
    _rtp_pad = max((_rtp_acc_max - _rtp_acc_min) * 0.3, 0.002)
    _rtp_ylim = (max(0.0, _rtp_acc_min - _rtp_pad), min(1.05, _rtp_acc_max + _rtp_pad))
    _plot_matrix(rtp_df, y_col="accuracy", ci_col="acc_ci",
                 ylabel="Non-Toxic Response Rate", filename="rtp_results_matrix.png",
                 model_order=["full", "baseline", "lora_abstention", "self_assessment"],
                 ylim=_rtp_ylim,
                 xlim=(min(ABSTENTION_RATIOS) - 0.1, max(ABSTENTION_RATIOS)+0.06),
                 ytick_format="%.3f", show_panel_title=False, legend_side=True)

# Combined plot: main results + constant-step overlaid (cs models as dotted lines)
if not df.empty and not cs_df.empty:
    cs_only = cs_df[cs_df["model"].isin(["baseline_cs", "lora_abstention_cs", "self_assessment_cs"])].copy()
    combined_df = pd.concat([df, cs_only], ignore_index=True)
    _plot_matrix(combined_df, y_col="accuracy", ci_col="acc_ci",
                 ylabel="Selective Accuracy",
                 filename="combined_accuracy_matrix.png",
                 legend_ncol=3, legend_y_offset=-0.30)
    _plot_matrix(combined_df, y_col="reward", ci_col="rew_ci",
                 ylabel="Reward  J = (1-α)·S + α·r_bot",
                 filename="combined_reward_matrix.png")

# ── Load and aggregate: cross-domain transfer ─────────────────────────────────

transfer_dir = os.path.join(BASE_DIR, "traj_csvs", "transfer")
if os.path.isdir(transfer_dir):
    tr_groups = discover_trajectory_files(transfer_dir)
    # key format: "{source}_to_{target}" — group by target, rename full→transfer
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
            records = compute_seed_metrics(df_raw, ABSTENTION_RATIOS)
            # rename full→transfer, drop no_abstention (comes from main data)
            renamed = [dict(r, model="transfer") for r in records if r["model"] == "full"]
            seed_records.append(renamed)
        if not seed_records:
            continue
        tr_agg = aggregate_seed_metrics(seed_records)
        tr_agg["n_seeds"] = len(seed_records)
        target_agg_map.setdefault(target, []).append(tr_agg)

    # Build combined df: main + constant-step data for each target + its transfer rows
    tr_combined_dfs = []
    for target, tr_agg_list in target_agg_map.items():
        main_target = df[df["dataset_model_string"] == target] if not df.empty else pd.DataFrame()
        cs_target = cs_df[cs_df["dataset_model_string"] == target] if not cs_df.empty else pd.DataFrame()
        cs_target = cs_target[cs_target["model"].isin(["baseline_cs", "lora_abstention_cs", "self_assessment_cs"])]
        pieces = [main_target, cs_target] + [a.assign(dataset_model_string=target) for a in tr_agg_list]
        tr_combined_dfs.append(pd.concat(pieces, ignore_index=True))

    if tr_combined_dfs:
        tr_df = pd.concat(tr_combined_dfs, ignore_index=True)

        def _with_best_baseline(df, y_col):
            """Replace individual baseline rows with a single pointwise-best-baseline row."""
            bl_models = ["baseline", "lora_abstention", "self_assessment",
                         "baseline_cs", "lora_abstention_cs", "self_assessment_cs"]
            non_bl = df[~df["model"].isin(bl_models)].copy()
            bl = df[df["model"].isin(bl_models)]
            best = (
                bl.loc[bl.groupby(["dataset_model_string", "abstention_ratio"])[y_col].idxmax()]
                  .copy()
            )
            best["model"] = "best_baseline"
            return pd.concat([non_bl, best], ignore_index=True)

        _plot_matrix(_with_best_baseline(tr_df, "accuracy"),
                     y_col="accuracy", ci_col="acc_ci",
                     ylabel="Selective Accuracy",
                     filename="transfer_accuracy_matrix.png",
                     model_order=["best_baseline", "full", "transfer"])
        _plot_matrix(_with_best_baseline(tr_df, "reward"),
                     y_col="reward", ci_col="rew_ci",
                     ylabel="Reward  J = (1-α)·S + α·r_bot",
                     filename="transfer_reward_matrix.png",
                     model_order=["best_baseline", "full", "transfer"])

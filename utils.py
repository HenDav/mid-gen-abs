import hashlib
import os
import glob
import re

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def add_tiebreak_noise(df, base_seed=1):
    """Add tiny deterministic noise to trajectory_value to break ties.

    Noise scale is 1% of each model's value std (floor 1e-6), so it is
    method-aware but never large enough to meaningfully change rankings
    where values genuinely differ.

    Noise is seeded by (base_seed, model_name) and applied in
    (sample_index, token_index) order, so the same sample always receives
    the same perturbation regardless of which data-seed file is loaded.
    """
    df = df.copy()
    for model, sub in df.groupby("model"):
        vals = sub["trajectory_value"].values
        scale = max(float(np.std(vals)) * 0.01, 1e-6)
        model_hash = int(hashlib.md5(model.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(base_seed ^ model_hash)
        # Sort by (sample_index, token_index) for a stable, seed-independent order
        order = np.argsort(
            sub["sample_index"].values.astype(np.int64) * 100000
            + sub["token_index"].values.astype(np.int64)
        )
        noise = np.empty(len(sub))
        noise[order] = rng.normal(0, scale, len(sub))
        df.loc[sub.index, "trajectory_value"] = vals + noise
    return df


def break_ties(group):
    if len(group) == 1:
        return group
    n = len(group)
    shuffled_idx = np.random.permutation(group.index)
    # get the scalar value of the first trajectory_value in this group
    base_value = abs(group["trajectory_value"].iloc[0])
    if base_value == 0:
        base_value = 1
    increment = base_value * 1e-8  # scale small relative to the magnitude
    # create small distinct offsets
    offsets = np.arange(1, n + 1) * increment
    np.random.shuffle(offsets)
    group.loc[shuffled_idx, "trajectory_value"] += offsets
    return group


def weighted_threshold_discrete(df, value_col, weight_col, X):
    """
    Find threshold v (an actual value from the dataset) such that
    the sum of weights where value < v is approximately X% of the total.
    """
    df_sorted = df.sort_values(value_col).reset_index(drop=True)
    df_sorted["cum_weight"] = df_sorted[weight_col].cumsum()
    total_weight = df_sorted[weight_col].sum()
    df_sorted["cum_frac"] = df_sorted["cum_weight"] / total_weight

    idx = (df_sorted["cum_frac"] >= X).idxmax()  # first index meeting condition
    v = df_sorted.loc[idx, value_col]

    # Compute actual achieved fraction
    actual_weight_above = df_sorted.loc[df_sorted[value_col] > v, weight_col].sum()
    actual_frac = actual_weight_above / total_weight

    return v, actual_frac


def find_threshold_for_generative_abstention(df, value_col, X, total_weight):
    df_sorted = df.sort_values(value_col).reset_index(drop=True)
    seen_samples = {}
    cumsum_values = []
    value = 0
    for _, row in df_sorted.iterrows():
        s, t, row_total = row["sample_index"], row['token_index'], row['output_length']
        if s not in seen_samples.keys():
            value += row_total - t
            seen_samples[s] = t
        else:
            prev_t = seen_samples[s]
            if prev_t > t:
                value += prev_t - t
                seen_samples[s] = t
            # else no need to change anything
        cumsum_values.append(value)
    df_sorted["token_cumsum"] = cumsum_values
    df_sorted["cum_frac"] = df_sorted["token_cumsum"] / total_weight
    idx = (df_sorted["cum_frac"] >= X).idxmax()  # first index meeting condition
    v = df_sorted.loc[idx, value_col]
    return v, df_sorted.loc[idx]["cum_frac"]


def select_row_min_token_abstained(group, T):
    below_T = group[group["trajectory_value"] < T]
    if not below_T.empty:
        return below_T.loc[below_T["token_index"].idxmin()]
    else:
        row = group.loc[group["token_index"].idxmax()].copy()
        row["token_index"] += 1
        return row


def get_exact_percentile_rows(data, T, ratio, column):
    target_n = int(len(data) * (1 - ratio))
    above_T = data[data[column] > T]
    equal_T = data[data[column] == T]
    needed = target_n - len(above_T)
    if needed > 0:
        sampled_equal_T = equal_T.sample(n=needed, random_state=42)
        selected = pd.concat([above_T, sampled_equal_T], ignore_index=True)
    else:
        selected = above_T.head(target_n)
    return selected


def add_saved_token_stats(df, baselines):
    full_df = df[df["model"] == "full"][["abstention_ratio", "saved_tokens"]].rename(
        columns={"saved_tokens": "full_saved_tokens"}
    )
    other_df = (
        df[df["model"].isin(baselines)]
            .groupby("abstention_ratio", as_index=False)["saved_tokens"]
            .mean()
            .rename(columns={"saved_tokens": "avg_other_saved_tokens"})
    )
    result = pd.merge(full_df, other_df, on="abstention_ratio", how="inner")
    print(result)
    result["full_vs_avg_pct"] = (
            result["full_saved_tokens"] / result["avg_other_saved_tokens"] * 100
    )
    df = df.merge(result, on='abstention_ratio', how='left')
    df['full_vs_avg_pct'] = df.apply(
        lambda r: r['full_vs_avg_pct'] if r['model'] == 'full' else None, axis=1
    )
    return df


def draw_graph_main_results(df, title_string, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df[df["model"] != "no_abstention"],
        x="abstention_ratio",
        y="accuracy",
        hue="model",
        marker="o"
    )
    no_abst_df = df[df["model"] == "no_abstention"]
    plt.plot(
        no_abst_df["abstention_ratio"],
        no_abst_df["accuracy"],
        color="grey",
        marker="o",
        label="no_abstention",
        linewidth=2,
        linestyle=(0, (5, 5)),
        markersize=4
    )
    full_df = df[df["model"] == "full"]
    for _, row in full_df.iterrows():
        plt.text(
            row["abstention_ratio"],
            row["accuracy"],
            f'{row["full_vs_avg_pct"]:.1f}%',  # format e.g. 95.3%
            fontsize=9,
            color="black",
            ha="right",
            va="bottom"
        )
    plt.title("Accuracy vs. Abstention Ratio per Model for " + title_string)
    plt.xlabel("Abstention Ratio")
    plt.ylabel("Accuracy")
    plt.legend(title="Model")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    else:
        plt.show()
    plt.close()


def draw_graph_saved_tokens(df, title_string, abstention_acc, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df[df["model"] != "no_abstention"],
        x="saved_tokens",
        y="accuracy",
        hue="model",
        marker="o"
    )
    no_abst_df = df[df["model"] == "no_abstention"]
    plt.plot(
        no_abst_df["saved_tokens"],
        no_abst_df["accuracy"],
        color="grey",
        marker="o",
        label="no_abstention",
        linewidth=2,
        linestyle=(0, (5, 5)),
        markersize=4
    )
    plt.title("Accuracy vs. Saved for " + title_string + ". Abstention reward for accuracy calc: " + str(abstention_acc))
    plt.xlabel("Saved tokens")
    plt.ylabel("Accuracy")
    plt.legend(title="Model")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    else:
        plt.show()
    plt.close()


def ci95(values):
    """95% CI half-width using t-distribution (appropriate for small n)."""
    from scipy import stats
    n = len(values)
    if n < 2:
        return 0.0
    return float(stats.t.ppf(0.975, df=n - 1) * stats.sem(values))


def get_abstention_time_value(trajectory_values, threshold):
    """Return the first trajectory value that drops below threshold, or None if it never does."""
    for v in trajectory_values:
        if v < threshold:
            return v
    return None


def compute_r_bot(abstention_vals, correctness_labels, T):
    """Estimate r_bot via isotonic regression on abstention-time values (from reward_logic_ver.py)."""
    from sklearn.isotonic import IsotonicRegression
    if len(abstention_vals) > 1:
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(abstention_vals, correctness_labels)
        return float(iso.predict([T])[0])
    return float(np.mean(correctness_labels)) if correctness_labels else 0.5


def compute_seed_metrics(results, abstention_ratios, baselines=None):
    """Compute accuracy, calibrated reward, and saved_tokens per (model, ratio) for one seed DataFrame."""
    if baselines is None:
        baselines = ["lora_abstention", "self_assessment", "baseline"]
    records = []

    for b in baselines:
        b_df = results[results["model"] == b].copy()
        if len(b_df) == 0:
            continue
        tokens_generated = b_df["output_length"].sum()
        thresholds = b_df["trajectory_value"].quantile(abstention_ratios)
        for T, ratio in zip(thresholds, abstention_ratios):
            surviving = get_exact_percentile_rows(b_df, T, ratio, "trajectory_value")
            accuracy = float(1 - surviving["should_abstain_label"].mean())
            saved_tokens = float((tokens_generated - surviving["output_length"].sum()) / tokens_generated) if tokens_generated > 0 else 0.0
            # r_bot: vectorized — for single-row-per-sample baselines, v < T iff abstains
            mask = b_df["trajectory_value"] < T
            abstention_vals = b_df.loc[mask, "trajectory_value"].tolist()
            correctness_labels = (1.0 - b_df.loc[mask, "should_abstain_label"]).tolist()
            r_bot = compute_r_bot(abstention_vals, correctness_labels, T)
            records.append({"model": b, "abstention_ratio": ratio, "accuracy": accuracy,
                            "saved_tokens": saved_tokens, "reward": accuracy * (1 - ratio) + r_bot * ratio})

    full_df = results[results["model"] == "full"].copy()
    if len(full_df) > 0:
        sample_trajectories = {}
        sample_correctness = {}
        for sid, grp in full_df.groupby("sample_index"):
            grp_s = grp.sort_values("output_length")
            sample_trajectories[sid] = grp_s["trajectory_value"].values
            sample_correctness[sid] = float(1 - grp_s["should_abstain_label"].iloc[0])

        min_traj = full_df.loc[full_df.groupby("sample_index")["trajectory_value"].idxmin()]
        tokens_generated = min_traj["output_length"].sum()
        thresholds = min_traj["trajectory_value"].quantile(abstention_ratios)

        for T, ratio in zip(thresholds, abstention_ratios):
            surviving = get_exact_percentile_rows(min_traj, T, ratio, "trajectory_value")
            accuracy = float(1 - surviving["should_abstain_label"].mean())
            # Vectorized token savings: for each sample, earliest token where value < T,
            # or (last token + 1) if it never crosses (no abstention).
            below = full_df[full_df["trajectory_value"] < T]
            abstain_tok = below.groupby("sample_index")["token_index"].min()
            no_abstain_sids = np.setdiff1d(min_traj["sample_index"].values, abstain_tok.index.values)
            no_abstain_tok = (
                full_df[full_df["sample_index"].isin(no_abstain_sids)]
                .groupby("sample_index")["token_index"].max() + 1
            )
            all_tok = pd.concat([abstain_tok, no_abstain_tok])
            saved_tokens = float((tokens_generated - all_tok.sum()) / tokens_generated) if tokens_generated > 0 else 0.0
            # r_bot: first value below T per sample, via numpy argmax on boolean array
            abstention_vals, correctness_labels = [], []
            for sid, traj in sample_trajectories.items():
                below_mask = traj < T
                if below_mask.any():
                    abstention_vals.append(traj[below_mask.argmax()])
                    correctness_labels.append(sample_correctness[sid])
            r_bot = compute_r_bot(abstention_vals, correctness_labels, T)
            records.append({"model": "full", "abstention_ratio": ratio, "accuracy": accuracy,
                            "saved_tokens": saved_tokens, "reward": accuracy * (1 - ratio) + r_bot * ratio})

        base_acc = float(1 - full_df.drop_duplicates("sample_index")["should_abstain_label"].mean())
        for ratio in abstention_ratios:
            records.append({"model": "no_abstention", "abstention_ratio": ratio,
                            "accuracy": base_acc, "saved_tokens": 0.0, "reward": base_acc})
    else:
        # No full model: derive no_abstention from the first available baseline
        all_baseline_rows = results[results["model"].isin(baselines)]
        if len(all_baseline_rows) > 0:
            first_b = all_baseline_rows["model"].iloc[0]
            b_df = results[results["model"] == first_b]
            base_acc = float(1 - b_df["should_abstain_label"].mean())
            for ratio in abstention_ratios:
                records.append({"model": "no_abstention", "abstention_ratio": ratio,
                                "accuracy": base_acc, "saved_tokens": 0.0, "reward": base_acc})

    return records


def aggregate_seed_metrics(seed_records_list):
    """Aggregate a list of per-seed record lists into mean ± 95% CI per (model, abstention_ratio)."""
    all_df = pd.concat([pd.DataFrame(r) for r in seed_records_list], ignore_index=True)
    return (
        all_df.groupby(["model", "abstention_ratio"])
        .agg(
            accuracy=("accuracy", "mean"),
            acc_ci=("accuracy", ci95),
            reward=("reward", "mean"),
            rew_ci=("reward", ci95),
            saved_tokens=("saved_tokens", "mean"),
            saved_tokens_ci=("saved_tokens", ci95),
        )
        .reset_index()
    )


def align_samples(df, label=""):
    """Warn and filter to the intersection of sample_indices across all models.

    When CSVs from different experiments are concatenated, a model may cover
    a different set of samples than another.  This function detects that and
    restricts every model to the common subset so downstream metrics are
    always computed on the same samples.
    """
    counts = df.groupby("model")["sample_index"].apply(set)
    common = set.intersection(*counts.values)
    model_sizes = {m: len(s) for m, s in counts.items()}
    all_same = len(set(model_sizes.values())) == 1 and all(s == common for s in counts.values)

    if not all_same:
        tag = f" [{label}]" if label else ""
        print(f"WARNING{tag}: sample mismatch across models — {model_sizes}")
        print(f"  Filtering to {len(common)} common samples.")
        df = df[df["sample_index"].isin(common)].copy()
    return df


def discover_nested_trajectory_files(base_dir):
    """Scan base_dir for nested seed structure: base_dir/seed_N/**/trajectory_values.csv.

    Returns dict mapping the base_dir folder name -> sorted list of file paths.
    Example: gsm8k_qwen/seed_42/plots/abs_rate_analysis/trajectory_values.csv
             → {"gsm8k_qwen": [...]}
    """
    pattern = os.path.join(base_dir, "seed_*", "**", "trajectory_values.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    key = os.path.basename(os.path.normpath(base_dir))
    return {key: files} if files else {}


def discover_trajectory_files(traj_dir):
    """Scan traj_dir for CSV files and group seeds by dataset/model key.

    Files matching *_seedN.csv are grouped together (seed suffix stripped).
    All other CSVs are treated as single-file groups.
    Returns: dict mapping dataset_model_string -> sorted list of file paths.
    """
    files = sorted(glob.glob(os.path.join(traj_dir, "*.csv")))
    groups = {}
    for f in files:
        stem = os.path.basename(f).replace("trajectory_values_", "").replace(".csv", "")
        key = re.sub(r"_seed\d+$", "", stem)
        groups.setdefault(key, []).append(f)
    return groups


def draw_graph_abstention_rate_reward(df, title_string, save_path=None):
    plt.figure(figsize=(12, 8))
    
    # Get unique models (excluding no_abstention)
    models = [m for m in df["model"].unique() if m != "no_abstention"]
    colors = plt.cm.tab10.colors
    model_colors = {m: colors[i] for i, m in enumerate(models)}
    
    # Plot each model with annotations
    for model in models:
        model_df = df[df["model"] == model]
        plt.plot(
            model_df["abstention_ratio"],
            model_df["reward"],
            marker="o",
            label=model,
            color=model_colors[model]
        )
        # Annotate with abstention_reward values
        for _, row in model_df.iterrows():
            if row["abstention_reward"] is not None:
                plt.annotate(
                    f'{row["abstention_reward"]:.2f}',
                    (row["abstention_ratio"], row["reward"]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center',
                    fontsize=6,
                    color=model_colors[model],
                    alpha=0.8
                )
    
    # Plot no_abstention baseline
    no_abst_df = df[df["model"] == "no_abstention"]
    plt.plot(
        no_abst_df["abstention_ratio"],
        no_abst_df["reward"],
        color="grey",
        marker="o",
        label="no_abstention",
        linewidth=2,
        linestyle=(0, (5, 5)),
        markersize=4
    )
    
    plt.title("Reward vs. Abstention Rate: " + title_string + "\n(annotations show threshold T used as r_⊥)")
    plt.xlabel("Abstention Rate")
    plt.ylabel("Reward")
    plt.legend(title="Model", loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    else:
        plt.show()
    plt.close()
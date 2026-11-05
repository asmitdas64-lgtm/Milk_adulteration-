"""Exploratory Data Analysis (EDA) and Data Card generation."""

import json
from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_summary_stats(df: pd.DataFrame, target_col: str = "label") -> Dict[str, Any]:
    """Compute dataset summary statistics and class distribution.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.

    Returns:
        Dictionary of summary statistics.
    """
    row_count = int(len(df))
    col_count = int(len(df.columns))
    duplicate_count = int(df.duplicated().sum())

    missing_counts = {str(k): int(v) for k, v in df.isna().sum().to_dict().items()}
    missing_total = int(sum(missing_counts.values()))

    # Class balance
    counts = {str(k): int(v) for k, v in df[target_col].value_counts().to_dict().items()}
    proportions = {str(k): float(v / row_count) for k, v in counts.items()}

    # Numeric summary per feature
    numeric_summary = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_summary[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()) if row_count > 1 else 0.0,
                "min": float(df[col].min()),
                "25%": float(df[col].quantile(0.25)),
                "50%": float(df[col].median()),
                "75%": float(df[col].quantile(0.75)),
                "max": float(df[col].max()),
            }

    return {
        "row_count": row_count,
        "column_count": col_count,
        "duplicate_count": duplicate_count,
        "missing_counts": missing_counts,
        "missing_total": missing_total,
        "class_balance": {
            "counts": counts,
            "proportions": proportions,
        },
        "feature_summary": numeric_summary,
    }


def compute_iqr_outliers(df: pd.DataFrame, features: list, target_col: str = "label") -> Dict[str, Any]:
    """Compute count of IQR-based outliers for each feature across classes."""
    outlier_report = {}
    for col in features:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        is_outlier = (df[col] < lower) | (df[col] > upper)
        total_outliers = int(is_outlier.sum())
        
        # Breakdown by target class
        by_class = {}
        for c in df[target_col].unique():
            cls_mask = df[target_col] == c
            by_class[str(c)] = int(((df[col] < lower) | (df[col] > upper))[cls_mask].sum())

        outlier_report[col] = {
            "total_outliers": total_outliers,
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "by_class": by_class,
        }
    return outlier_report


def plot_class_balance(df: pd.DataFrame, target_col: str, save_path: Path):
    """Generate and save class balance bar chart."""
    plt.figure(figsize=(6, 4))
    counts = df[target_col].value_counts().sort_index()
    labels = ["Pure / High (0)" if idx == 0 else "Adulterated / Substandard (1)" for idx in counts.index]
    
    bars = plt.bar(labels, counts.values, color=["#2ca02c", "#d62728"], alpha=0.85, edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts.values) * 0.01,
            f"{int(height)} ({height / len(df):.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.title("Binary Target Class Balance", fontsize=12, fontweight="bold")
    plt.ylabel("Sample Count", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, save_path: Path):
    """Generate and save Pearson correlation heatmap."""
    plt.figure(figsize=(8, 6))
    corr = df.corr()
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation"},
        vmin=-1.0,
        vmax=1.0,
    )
    plt.title("Feature Pearson Correlation Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_distributions(df: pd.DataFrame, features: list, target_col: str, save_path: Path):
    """Plot feature distributions comparing pure vs adulterated classes."""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feat in enumerate(features):
        ax = axes[idx]
        for cls_val, color, label in [(0, "#2ca02c", "Pure (0)"), (1, "#d62728", "Adulterated (1)")]:
            subset = df[df[target_col] == cls_val][feat].dropna()
            sns.histplot(subset, ax=ax, color=color, label=label, kde=True, stat="density", common_norm=False, alpha=0.4)
        ax.set_title(f"Distribution: {feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel(feat)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    # Hide extra unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Adulteration Class", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_data_card(df: pd.DataFrame, summary: Dict[str, Any], outliers: Dict[str, Any], config: Dict[str, Any], save_path: Path):
    """Generate comprehensive Markdown Data Card per context/03_DATA_SPEC.md."""
    target_col = config["data"].get("target", "label")
    features = config["data"].get("features", [])
    
    cb_counts = summary["class_balance"]["counts"]
    cb_props = summary["class_balance"]["proportions"]
    
    lines = [
        "# DATA CARD — Milk Quality & Adulteration Dataset",
        "",
        "## 1. Overview & Provenance",
        f"- **Dataset Name:** Milk Quality Dataset (`clean.csv`)",
        f"- **Domain / Food Category:** Bovine Milk (Physicochemical & Organoleptic Screening)",
        f"- **Total Samples:** {summary['row_count']}",
        f"- **Total Columns:** {summary['column_count']} (7 features + 1 target)",
        f"- **Duplicate Rows (Discrete Combinations):** {summary['duplicate_count']}",
        f"- **Total Missing Values:** {summary['missing_total']}",
        "- **Target Definition:** Binary classification where `0 = Pure / High Quality` and `1 = Adulterated / Substandard`.",
        "",
        "## 2. Target Class Balance",
        f"- **Class 0 (Pure / High Quality):** {cb_counts.get('0', 0)} samples ({cb_props.get('0', 0):.2%})",
        f"- **Class 1 (Adulterated / Substandard):** {cb_counts.get('1', 0)} samples ({cb_props.get('1', 0):.2%})",
        f"- **Minority Class Proportion:** {min(cb_props.values()):.2%}",
        "- **Decision D-07 Application:** Since the minority class is >= 24.2%, standard stratified CV is appropriate without mandatory synthetic oversampling (SMOTE).",
        "",
        "## 3. Feature Descriptions & Statistics",
        "| Feature | Type | Range / Values | Mean ± Std | Median | Outliers (IQR) | Description |",
        "|---|---|---|---|---|---|---|",
    ]

    for feat in features:
        if feat in summary["feature_summary"]:
            stats = summary["feature_summary"][feat]
            out_info = outliers.get(feat, {})
            out_count = out_info.get("total_outliers", 0)
            lines.append(
                f"| `{feat}` | Continuous/Discrete | [{stats['min']:.1f}, {stats['max']:.1f}] | "
                f"{stats['mean']:.2f} ± {stats['std']:.2f} | {stats['50%']:.2f} | {out_count} | "
                f"Physicochemical inspection parameter |"
            )

    lines.extend([
        "",
        "## 4. IQR Outlier Analysis & Food Safety Note",
        "In food safety and adulteration screening, extreme measurements (such as highly abnormal pH or temperature) are often genuine indicators of spoilage, chemical neutralizing agents, or watered-down milk.",
        "Under decision **D-05**, outlier clipping (winsorization) is preferred over deleting rows, preserving the critical positive class signal.",
        "",
        "## 5. Artifacts Generated",
        "- `reports/eda/summary.json`: Complete numerical statistics snapshot.",
        "- `reports/figures/eda_class_balance.png`: Target distribution chart.",
        "- `reports/figures/eda_correlation_matrix.png`: Pearson correlation heatmap.",
        "- `reports/figures/eda_distributions.png`: Per-class feature distributions.",
    ])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_eda(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute complete EDA pipeline and generate all artifacts."""
    interim_path = Path(config["paths"]["interim"])
    reports_dir = Path(config["paths"]["reports_dir"])
    target_col = config["data"].get("target", "label")
    features = config["data"].get("features", [])

    df = pd.read_csv(interim_path)

    # 1. Summary stats
    summary = generate_summary_stats(df, target_col=target_col)
    outliers = compute_iqr_outliers(df, features, target_col=target_col)

    # 2. Save summary.json
    eda_dir = reports_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    with open(eda_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "outliers": outliers}, f, indent=2)

    # 3. Plots
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_class_balance(df, target_col, fig_dir / "eda_class_balance.png")
    plot_correlation_matrix(df, fig_dir / "eda_correlation_matrix.png")
    plot_distributions(df, features, target_col, fig_dir / "eda_distributions.png")

    # 4. Generate Data Card
    generate_data_card(df, summary, outliers, config, eda_dir / "data_card.md")

    return {
        "summary": summary,
        "outliers": outliers,
    }

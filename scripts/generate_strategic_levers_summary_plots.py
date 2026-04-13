import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from reporting_variant_paths import asset_path, data_path, validate_variant

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["dynamic", "static"], default="dynamic")
    return parser.parse_args()


def usd_compact(value: float, _position: float = 0.0) -> str:
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.0f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def pct_fmt(value: float, _position: float) -> str:
    return f"{value:.0%}"


def build_focus_plot(variant: str) -> None:
    df = pd.read_csv(data_path(ROOT, "sim_focus_gain_main.csv", variant))
    peak = df.loc[df["focus_revenue"].idxmax()]

    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.8, 4.1), constrained_layout=True)

    ax.fill_between(
        df["top_share"],
        df["net_vs_baseline_usd"],
        0,
        where=df["net_vs_baseline_usd"] >= 0,
        alpha=0.18,
        color="#2563EB",
        interpolate=True,
    )
    ax.fill_between(
        df["top_share"],
        df["net_vs_baseline_usd"],
        0,
        where=df["net_vs_baseline_usd"] < 0,
        alpha=0.18,
        color="#DC2626",
        interpolate=True,
    )
    ax.plot(
        df["top_share"],
        df["net_vs_baseline_usd"],
        color="#2563EB",
        linewidth=2.8,
    )
    ax.axhline(0, color="#0F172A", linewidth=1.2, linestyle="--")
    ax.scatter(
        [peak["top_share"]],
        [peak["net_vs_baseline_usd"]],
        color="#DC2626",
        s=68,
        zorder=5,
        edgecolors="white",
        linewidths=0.9,
    )
    ax.annotate(
        f"Best: +{usd_compact(peak['net_vs_baseline_usd'])}\n@ top {peak['top_share']:.0%}",
        xy=(peak["top_share"], peak["net_vs_baseline_usd"]),
        xytext=(0.61, peak["net_vs_baseline_usd"] - 35_000_000),
        fontsize=9,
        color="#DC2626",
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": "#DC2626", "lw": 0.8},
    )
    ax.set_title("Lever 1 — Net revenue delta vs baseline", fontsize=12, weight="bold")
    ax.set_xlabel("Share of ranked opportunities worked")
    ax.set_ylabel("USD")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd_compact))
    output_path = asset_path(ROOT, "strategic_lever_focus_summary.png", variant)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_sales_plot(variant: str) -> None:
    df = pd.read_csv(data_path(ROOT, "sim_salesforce_reduction.csv", variant))
    peak = df.loc[df["net_margin"].idxmax()]
    baseline = float(df["net_margin"].iloc[0])

    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.8, 4.1), constrained_layout=True)

    ax.plot(df["n_reps"], df["net_margin"], color="#2563EB", linewidth=2.8)
    ax.axhline(baseline, color="#0F172A", linewidth=1.2, linestyle=":")
    ax.scatter(
        [peak["n_reps"]],
        [peak["net_margin"]],
        color="#DC2626",
        s=68,
        zorder=5,
        edgecolors="white",
        linewidths=0.9,
    )
    ax.annotate(
        f"Best: {int(peak['n_reps'])} reps\n+{usd_compact(peak['net_margin_vs_baseline'])} vs baseline",
        xy=(peak["n_reps"], peak["net_margin"]),
        xytext=(peak["n_reps"] + 12, peak["net_margin"] - 12_000_000),
        fontsize=9,
        color="#DC2626",
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": "#DC2626", "lw": 0.8},
    )
    ax.set_title(
        "Lever 2 — Gross profit after selling costs by headcount",
        fontsize=12,
        weight="bold",
    )
    ax.set_xlabel("Sales reps (headcount)")
    ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd_compact))
    output_path = asset_path(ROOT, "strategic_lever_salesforce_summary.png", variant)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    variant = validate_variant(args.variant)
    build_focus_plot(variant)
    build_sales_plot(variant)


if __name__ == "__main__":
    main()

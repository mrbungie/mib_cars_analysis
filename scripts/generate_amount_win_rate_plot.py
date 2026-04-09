from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/intermediate/cleaned_data.parquet"
ASSET_PATH = ROOT / "slidedeck/assets/eda_amount_win_rate.png"


def usd_compact(value: float, _position: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def pct_fmt(value: float, _position: float) -> str:
    return f"{value:.0%}"


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df = df[["Opportunity Amount USD", "Opportunity Result"]].dropna().copy()
    df = df[df["Opportunity Result"].isin(["Won", "Loss"])].copy()
    df["won_flag"] = df["Opportunity Result"].eq("Won").astype(int)

    amount_cap = float(df["Opportunity Amount USD"].quantile(0.95))
    hist_df = df.loc[df["Opportunity Amount USD"] <= amount_cap].copy()

    amount_bins = pd.qcut(df["Opportunity Amount USD"], q=12, duplicates="drop")
    bin_summary = (
        df.assign(amount_bin=amount_bins)
        .groupby("amount_bin", observed=False)
        .agg(
            opportunities=("won_flag", "size"),
            win_rate=("won_flag", "mean"),
        )
        .reset_index()
    )
    bin_summary["bin_left"] = (
        bin_summary["amount_bin"]
        .apply(lambda interval: float(interval.left))
        .astype(float)
    )
    bin_summary["bin_right"] = (
        bin_summary["amount_bin"]
        .apply(lambda interval: float(interval.right))
        .astype(float)
    )
    bin_summary["bin_mid"] = (bin_summary["bin_left"] + bin_summary["bin_right"]) / 2
    bin_summary["bin_index"] = range(len(bin_summary))
    bin_summary["bin_label"] = bin_summary.apply(
        lambda row: f"${row['bin_left'] / 1000:.0f}K–${row['bin_right'] / 1000:.0f}K",
        axis=1,
    )

    win_loss = (
        df["Opportunity Result"]
        .value_counts()
        .reindex(["Loss", "Won"])
        .rename_axis("result")
        .reset_index(name="opportunities")
    )
    win_loss["share"] = win_loss["opportunities"] / win_loss["opportunities"].sum()

    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(11.2, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.12, wspace=0.16)

    ax_hist = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_joint = fig.add_subplot(gs[1, :])

    sns.histplot(
        hist_df["Opportunity Amount USD"],
        bins=32,
        color="#14B8A6",
        edgecolor="white",
        linewidth=0.7,
        ax=ax_hist,
    )
    ax_hist.set_title(
        "Amount histogram (capped at 95th pct)", fontsize=12, weight="bold"
    )
    ax_hist.set_xlabel("Opportunity amount")
    ax_hist.set_ylabel("Opportunity count")
    ax_hist.xaxis.set_major_formatter(FuncFormatter(usd_compact))

    sns.barplot(
        data=win_loss,
        x="result",
        y="opportunities",
        hue="result",
        palette={"Loss": "#0F172A", "Won": "#2563EB"},
        dodge=False,
        legend=False,
        ax=ax_bar,
    )
    ax_bar.set_title("Win/loss mix", fontsize=12, weight="bold")
    ax_bar.set_xlabel("")
    ax_bar.set_ylabel("Opportunity count")
    for patch, share in zip(ax_bar.patches, win_loss["share"]):
        ax_bar.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + win_loss["opportunities"].max() * 0.015,
            f"{share:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    ax_joint.bar(
        bin_summary["bin_index"],
        bin_summary["opportunities"],
        width=0.8,
        color="#BFDBFE",
        edgecolor="#93C5FD",
        label="Opportunity count",
    )
    ax_joint.set_title(
        "Observed win rate rises across higher amount bands", fontsize=13, weight="bold"
    )
    ax_joint.set_xlabel("Opportunity amount bin")
    ax_joint.set_ylabel("Opportunity count", color="#1D4ED8")
    ax_joint.tick_params(axis="y", labelcolor="#1D4ED8")
    ax_joint.set_xticks(bin_summary["bin_index"])
    ax_joint.set_xticklabels(bin_summary["bin_label"], rotation=28, ha="right")

    ax_joint_rate = ax_joint.twinx()
    ax_joint_rate.plot(
        bin_summary["bin_index"],
        bin_summary["win_rate"],
        color="#DC2626",
        linewidth=2.6,
        marker="o",
        markersize=6,
        label="Observed win rate",
    )
    ax_joint_rate.set_ylabel("Observed win rate", color="#DC2626")
    ax_joint_rate.tick_params(axis="y", labelcolor="#DC2626")
    ax_joint_rate.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax_joint_rate.set_ylim(0, max(0.4, bin_summary["win_rate"].max() * 1.15))

    overall_win_rate = df["won_flag"].mean()
    ax_joint_rate.axhline(
        overall_win_rate, color="#7C3AED", linestyle="--", linewidth=1.8
    )
    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {ASSET_PATH}")


if __name__ == "__main__":
    main()

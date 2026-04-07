from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/intermediate/cleaned_data.parquet"
ASSET_PATH = ROOT / "slidedeck/assets/eda_distribution_by_type_subtype_region.png"


def usd_compact(value: float, _position: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def short_label(value: str) -> str:
    mapping = {
        "Batteries & Accessories": "Batteries & Acc.",
        "Exterior Accessories": "Exterior Acc.",
        "Garage & Car Care": "Garage & Care",
        "Interior Accessories": "Interior Acc.",
        "Performance Parts": "Perf. Parts",
        "Replacement Parts": "Replacement",
        "Towing & Hitches": "Towing & Hitch.",
    }
    return mapping.get(value, value)


def summarize(df: pd.DataFrame, col: str) -> pd.DataFrame:
    summary = (
        df.groupby(col, observed=False)
        .agg(
            opportunities=("won_flag", "size"),
            win_rate=("won_flag", "mean"),
            p25=("Opportunity Amount USD", lambda s: float(s.quantile(0.25))),
            median_amount=("Opportunity Amount USD", "median"),
            p75=("Opportunity Amount USD", lambda s: float(s.quantile(0.75))),
        )
        .reset_index()
        .sort_values(["win_rate", "median_amount"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary["loss_rate"] = 1.0 - summary["win_rate"]
    summary["label"] = summary[col].map(short_label)
    return summary


def plot_mix(ax: plt.Axes, summary: pd.DataFrame, title: str, show_x: bool) -> None:
    y = list(range(len(summary)))
    ax.set_facecolor("#F8FAFC")
    ax.barh(y, summary["win_rate"], color="#2563EB", height=0.64)
    ax.barh(
        y, summary["loss_rate"], left=summary["win_rate"], color="#E2E8F0", height=0.64
    )
    ax.set_yticks(y)
    ax.set_yticklabels(summary["label"], fontsize=8.5, color="#0F172A")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=6, color="#0F172A")
    ax.grid(axis="x", alpha=0.12, color="#94A3B8")
    ax.grid(axis="y", visible=False)
    if show_x:
        ax.set_xlabel("Opportunity share", fontsize=9.5, color="#334155")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="x", labelsize=8.5, colors="#475569")
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    for pos, win_rate in enumerate(summary["win_rate"]):
        ax.text(
            max(0.03, min(0.98, float(win_rate) - 0.02)),
            pos,
            f"{win_rate:.0%}",
            va="center",
            ha="right",
            fontsize=8,
            color="white",
            weight="bold",
        )


def plot_amount(ax: plt.Axes, summary: pd.DataFrame, show_x: bool) -> None:
    y = list(range(len(summary)))
    ax.set_facecolor("#F8FAFC")
    ax.hlines(y, summary["p25"], summary["p75"], color="#94A3B8", linewidth=3.0)
    ax.scatter(summary["median_amount"], y, color="#0F766E", s=42, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0, 200_000)
    ax.xaxis.set_major_formatter(FuncFormatter(usd_compact))
    ax.grid(axis="x", alpha=0.12, color="#94A3B8")
    ax.grid(axis="y", visible=False)
    if show_x:
        ax.set_xlabel("Deal amount in USD (0 to $200K)", fontsize=9.5, color="#334155")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="x", labelsize=8.5, colors="#475569")
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df = df[
        [
            "Supplies Group",
            "Supplies Subgroup",
            "Region",
            "Opportunity Result",
            "Opportunity Amount USD",
        ]
    ].dropna()
    df = df[df["Opportunity Result"].isin(["Won", "Loss"])].copy()
    df["won_flag"] = df["Opportunity Result"].eq("Won").astype(int)

    sns.set_theme(style="white")
    plt.style.use("seaborn-v0_8-white")

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13.6, 9.6),
        constrained_layout=True,
        sharex="col",
    )
    configs = [
        ("Supplies Group", "Type"),
        ("Supplies Subgroup", "Subtype"),
        ("Region", "Region"),
    ]

    for idx, (col, label) in enumerate(configs):
        summary = summarize(df, col)
        show_x = idx == len(configs) - 1
        plot_mix(axes[idx, 0], summary, label, show_x=show_x)
        plot_amount(axes[idx, 1], summary, show_x=show_x)

    fig.patch.set_facecolor("white")
    fig.text(
        0.25,
        0.965,
        "Conversion mix",
        ha="center",
        va="top",
        fontsize=12.5,
        weight="bold",
        color="#0F172A",
    )
    fig.text(
        0.75,
        0.965,
        "Typical deal size",
        ha="center",
        va="top",
        fontsize=12.5,
        weight="bold",
        color="#0F172A",
    )
    fig.suptitle(
        "Commercial patterns already differ by type, subtype, and region",
        fontsize=15,
        weight="bold",
        y=1.01,
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.99,
        "Shared axes make the contrast comparable across all three cuts before modeling.",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#475569",
    )

    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {ASSET_PATH}")


if __name__ == "__main__":
    main()

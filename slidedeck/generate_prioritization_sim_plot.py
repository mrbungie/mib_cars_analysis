from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_PATH = ROOT / "slidedeck/data/classification_model_report.xlsx"
ASSET_PATH = ROOT / "slidedeck/assets/sim_prioritization.png"


def main() -> None:
    lift = pd.read_excel(CLASSIFICATION_PATH, sheet_name="prioritization_lift")
    plot_df = lift.sort_values("score_decile").copy()
    plot_df["score_decile"] = plot_df["score_decile"].astype(int)
    overall_win_rate = plot_df["wins"].sum() / plot_df["opportunities"].sum()
    plot_df["lift"] = plot_df["observed_win_rate"] / overall_win_rate

    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8.1, 5.1))
    sns.barplot(
        data=plot_df,
        x="score_decile",
        y="lift",
        color="#2563EB",
        ax=ax,
    )

    ax.axhline(
        1.0,
        color="#0F172A",
        linestyle="--",
        linewidth=1.8,
        label="Average held-out win rate = 1.0× lift",
    )

    for idx, value in enumerate(plot_df["lift"]):
        ax.text(
            idx, value + 0.06, f"{value:.2f}×", ha="center", va="bottom", fontsize=9
        )

    ax.set_title("Held-out lift by propensity decile", fontsize=13, weight="bold")
    ax.set_xlabel(
        "Propensity-score decile (1 = lowest predicted win probability, 10 = highest)"
    )
    ax.set_ylabel("Lift vs. overall held-out win rate")
    ax.set_ylim(0, max(plot_df["lift"].max() * 1.15, 1.6))
    ax.legend(loc="upper left", frameon=False)

    ax.text(
        0.02,
        0.94,
        f"Overall held-out win rate: {overall_win_rate:.1%}. Bars above 1.0× win more often than average.",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
        va="top",
    )

    fig.tight_layout()
    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {ASSET_PATH}")


if __name__ == "__main__":
    main()

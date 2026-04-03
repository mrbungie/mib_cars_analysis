from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_PATH = ROOT / "slidedeck/data/classification_model_report.xlsx"
REGRESSION_PATH = ROOT / "slidedeck/data/regression_model_report.xlsx"
ASSET_PATH = ROOT / "slidedeck/assets/sim_expected_value_threshold.png"


def usd_compact(value: float, _position: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.0f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def pct_fmt(value: float, _position: float) -> str:
    return f"{value:.0%}"


def build_curve() -> pd.DataFrame:
    classification = pd.read_excel(CLASSIFICATION_PATH, sheet_name="test_predictions")
    regression = pd.read_excel(REGRESSION_PATH, sheet_name="test_predictions")

    df = classification.merge(
        regression, on="row_id", how="inner", validate="one_to_one"
    )
    df["expected_value"] = df["predicted_win_probability"] * df["predicted_amount"]
    df["actual_won_amount"] = df["actual_amount"] * df["actual_result"]

    df = df.sort_values("expected_value", ascending=False).reset_index(drop=True)
    total_rows = len(df)
    total_actual_won = float(df["actual_won_amount"].sum())

    levels = []
    for pct in range(1, 101):
        keep_n = max(1, int(round(total_rows * pct / 100)))
        selected = df.iloc[:keep_n]
        threshold_value = float(selected["expected_value"].iloc[-1])
        estimated_value = float(selected["expected_value"].sum())
        actual_captured = float(selected["actual_won_amount"].sum())
        levels.append(
            {
                "top_share_selected": pct / 100,
                "selected_rows": keep_n,
                "threshold_value": threshold_value,
                "estimated_value": estimated_value,
                "actual_captured": actual_captured,
                "capture_rate": actual_captured / total_actual_won
                if total_actual_won
                else 0.0,
            }
        )

    return pd.DataFrame(levels)


def main() -> None:
    curve = build_curve()

    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8.6, 5.3))

    ax.plot(
        curve["top_share_selected"],
        curve["estimated_value"],
        color="#2563EB",
        linewidth=2.7,
        label="Sum of estimated E[X] × P(win)",
    )
    ax.plot(
        curve["top_share_selected"],
        curve["actual_captured"],
        color="#14B8A6",
        linewidth=2.7,
        label="Actual won amount captured",
    )
    ax.set_xlabel("Top share of opportunities kept by expected-value threshold")
    ax.set_ylabel("USD")
    ax.set_xlim(0.01, 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(usd_compact))

    ax_rate = ax.twinx()
    ax_rate.plot(
        curve["top_share_selected"],
        curve["capture_rate"],
        color="#DC2626",
        linewidth=2.0,
        linestyle="--",
        label="Share of total won amount captured",
    )
    ax_rate.set_ylabel("Captured share of total won amount", color="#DC2626")
    ax_rate.tick_params(axis="y", labelcolor="#DC2626")
    ax_rate.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax_rate.set_ylim(0, 1.02)

    for mark in [0.10, 0.25, 0.50]:
        row = curve.loc[(curve["top_share_selected"] - mark).abs().idxmin()]
        ax_rate.scatter(
            [row["top_share_selected"]],
            [row["capture_rate"]],
            color="#DC2626",
            s=24,
            zorder=5,
        )
        ax_rate.text(
            row["top_share_selected"] + 0.012,
            min(1.0, row["capture_rate"] + 0.03),
            f"Top {int(mark * 100)}%: {row['capture_rate']:.0%}",
            color="#DC2626",
            fontsize=9,
            weight="bold",
        )

    ax.text(
        0.02,
        0.96,
        "Threshold = minimum predicted expected value kept",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
        va="top",
    )

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax_rate.get_legend_handles_labels()
    ax.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="lower right",
        frameon=False,
    )

    fig.tight_layout()
    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {ASSET_PATH}")


if __name__ == "__main__":
    main()

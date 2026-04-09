from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_PATH = ROOT / "slidedeck/data/classification_model_report.xlsx"
REGRESSION_PATH = ROOT / "slidedeck/data/regression_model_report.xlsx"
ASSET_PATH = ROOT / "slidedeck/assets/annex_ev_probability_quadrants.png"


def usd_compact(value: float, _position: float = 0.0) -> str:
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.0f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def pct_fmt(value: float, _position: float = 0.0) -> str:
    return f"{value:.0%}"


def load_df() -> pd.DataFrame:
    classification = pd.read_excel(CLASSIFICATION_PATH, sheet_name="test_predictions")
    regression = pd.read_excel(REGRESSION_PATH, sheet_name="test_predictions")
    df = classification.merge(
        regression, on="row_id", how="inner", validate="one_to_one"
    )
    df["expected_value"] = df["predicted_win_probability"] * df["predicted_amount"]
    df["actual_won_amount"] = df["actual_amount"] * df["actual_result"]
    return df


def summarize_quadrants(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    p_threshold = float(df["predicted_win_probability"].median())
    ev_threshold = float(df["expected_value"].median())

    def label_row(row: pd.Series) -> str:
        high_p = row["predicted_win_probability"] >= p_threshold
        high_ev = row["expected_value"] >= ev_threshold
        if high_p and high_ev:
            return "Focus\nHigh E[value] • High P(win)"
        if high_p and not high_ev:
            return "Accelerate\nLow E[value] • High P(win)"
        if not high_p and high_ev:
            return "Intervene\nHigh E[value] • Low P(win)"
        return "Deprioritize\nLow E[value] • Low P(win)"

    df = df.copy()
    df["quadrant"] = df.apply(label_row, axis=1)

    summary = (
        df.groupby("quadrant", observed=False)
        .agg(
            count=("row_id", "size"),
            win_rate=("actual_result", "mean"),
            realized_revenue=("actual_won_amount", "sum"),
        )
        .reset_index()
    )
    return summary, p_threshold, ev_threshold


def main() -> None:
    df = load_df()
    summary, p_threshold, ev_threshold = summarize_quadrants(df)

    sns.set_theme(style="white")
    plt.style.use("seaborn-v0_8-white")

    fig, ax = plt.subplots(figsize=(10.8, 6.8), constrained_layout=True)
    ax.set_facecolor("#F8FAFC")

    colors = {
        "Deprioritize\nLow E[value] • Low P(win)": "#E5E7EB",
        "Accelerate\nLow E[value] • High P(win)": "#CCFBF1",
        "Intervene\nHigh E[value] • Low P(win)": "#FEF3C7",
        "Focus\nHigh E[value] • High P(win)": "#DBEAFE",
    }

    sample = df.sample(min(len(df), 1600), random_state=42).copy()
    ax.scatter(
        sample["predicted_win_probability"],
        sample["expected_value"],
        s=10,
        alpha=0.08,
        color="#64748B",
        edgecolors="none",
        zorder=1,
    )

    x0, x1 = 0.0, float(min(1.0, df["predicted_win_probability"].quantile(0.995)))
    y0, y1 = 0.0, float(df["expected_value"].quantile(0.995))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    rects = [
        (
            x0,
            y0,
            p_threshold - x0,
            ev_threshold - y0,
            colors["Deprioritize\nLow E[value] • Low P(win)"],
        ),
        (
            p_threshold,
            y0,
            x1 - p_threshold,
            ev_threshold - y0,
            colors["Accelerate\nLow E[value] • High P(win)"],
        ),
        (
            x0,
            ev_threshold,
            p_threshold - x0,
            y1 - ev_threshold,
            colors["Intervene\nHigh E[value] • Low P(win)"],
        ),
        (
            p_threshold,
            ev_threshold,
            x1 - p_threshold,
            y1 - ev_threshold,
            colors["Focus\nHigh E[value] • High P(win)"],
        ),
    ]
    for rx, ry, rw, rh, color in rects:
        ax.add_patch(plt.Rectangle((rx, ry), rw, rh, color=color, alpha=0.35, zorder=0))

    ax.axvline(p_threshold, color="#475569", linestyle="--", linewidth=1.5, zorder=2)
    ax.axhline(ev_threshold, color="#475569", linestyle="--", linewidth=1.5, zorder=2)

    positions = {
        "Deprioritize\nLow E[value] • Low P(win)": (
            p_threshold * 0.5,
            ev_threshold * 0.48,
        ),
        "Accelerate\nLow E[value] • High P(win)": (
            p_threshold + (x1 - p_threshold) * 0.52,
            ev_threshold * 0.48,
        ),
        "Intervene\nHigh E[value] • Low P(win)": (
            p_threshold * 0.5,
            ev_threshold + (y1 - ev_threshold) * 0.55,
        ),
        "Focus\nHigh E[value] • High P(win)": (
            p_threshold + (x1 - p_threshold) * 0.52,
            ev_threshold + (y1 - ev_threshold) * 0.55,
        ),
    }

    for _, row in summary.iterrows():
        x, y = positions[row["quadrant"]]
        ax.text(
            x,
            y,
            f"{row['quadrant']}\n"
            f"n {int(row['count']):,}  |  win {row['win_rate']:.0%}\n"
            f"rev {usd_compact(float(row['realized_revenue']))}",
            ha="center",
            va="center",
            fontsize=10.2,
            weight="bold",
            color="#0F172A",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "white",
                "edgecolor": "#E2E8F0",
                "linewidth": 0.8,
                "alpha": 0.96,
            },
            zorder=4,
        )

    ax.text(
        p_threshold,
        y1 * 0.985,
        f"Median P(win) {p_threshold:.1%}",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#334155",
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )
    ax.text(
        x1 * 0.988,
        ev_threshold,
        f"Median E[value]\n{usd_compact(ev_threshold)}",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#334155",
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )

    ax.set_title(
        "EV vs P(win) segmentation",
        fontsize=13,
        weight="bold",
        pad=12,
    )
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Expected value = P(win) × predicted amount")
    ax.xaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(usd_compact))
    ax.grid(alpha=0.10, color="#94A3B8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {ASSET_PATH}")


if __name__ == "__main__":
    main()

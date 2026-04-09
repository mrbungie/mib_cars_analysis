import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from reporting_variant_paths import asset_path, model_report_path, validate_variant


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["dynamic", "static"], default="dynamic")
    return parser.parse_args()


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


def summarize_ranking(
    df: pd.DataFrame,
    rank_column: str,
    label: str,
    total_actual_won: float,
    estimate_column: str | None = None,
) -> pd.DataFrame:
    ranked = df.sort_values(rank_column, ascending=False).reset_index(drop=True)
    total_rows = len(ranked)

    levels = []
    for pct in range(1, 101):
        keep_n = max(1, int(round(total_rows * pct / 100)))
        selected = ranked.iloc[:keep_n]
        actual_captured = float(selected["actual_won_amount"].sum())
        levels.append(
            {
                "strategy": label,
                "top_share_selected": pct / 100,
                "selected_rows": keep_n,
                "threshold_value": float(selected[rank_column].iloc[-1]),
                "actual_captured": actual_captured,
                "estimated_value": (
                    float(selected[estimate_column].sum()) if estimate_column else None
                ),
                "capture_rate": actual_captured / total_actual_won
                if total_actual_won
                else 0.0,
            }
        )

    return pd.DataFrame(levels)


def build_curve(variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification = pd.read_excel(
        model_report_path(ROOT, "classification", variant),
        sheet_name="test_predictions",
    )
    regression = pd.read_excel(
        model_report_path(ROOT, "regression", variant), sheet_name="test_predictions"
    )

    df = classification.merge(
        regression, on="row_id", how="inner", validate="one_to_one"
    )
    df["expected_value"] = df["predicted_win_probability"] * df["predicted_amount"]
    df["actual_won_amount"] = df["actual_amount"] * df["actual_result"]
    total_actual_won = float(df["actual_won_amount"].sum())

    expected_curve = summarize_ranking(
        df,
        "expected_value",
        "Expected value",
        total_actual_won,
        estimate_column="expected_value",
    )
    propensity_curve = summarize_ranking(
        df,
        "predicted_win_probability",
        "Propensity only",
        total_actual_won,
    )

    return expected_curve, propensity_curve


def main() -> None:
    args = parse_args()
    variant = validate_variant(args.variant)
    output_path = asset_path(ROOT, "sim_expected_value_threshold.png", variant)
    ev_curve, propensity_curve = build_curve(variant)

    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, (ax_amount, ax_share) = plt.subplots(
        2,
        1,
        figsize=(8.6, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 0.95], "hspace": 0.12},
        constrained_layout=True,
    )

    ax_amount.plot(
        ev_curve["top_share_selected"],
        ev_curve["estimated_value"],
        color="#2563EB",
        linewidth=2.7,
        label="Sum of estimated E[X] × P(win)",
    )
    ax_amount.plot(
        ev_curve["top_share_selected"],
        ev_curve["actual_captured"],
        color="#14B8A6",
        linewidth=2.7,
        label="Actual won amount captured (ranked by E[X] × P(win))",
    )
    ax_amount.set_ylabel("USD")
    ax_amount.set_xlim(0.01, 1.0)
    ax_amount.yaxis.set_major_formatter(FuncFormatter(usd_compact))
    ax_amount.set_title(
        "Expected-value thresholding vs propensity-only ranking",
        fontsize=13,
        weight="bold",
    )

    ax_share.plot(
        ev_curve["top_share_selected"],
        ev_curve["capture_rate"],
        color="#DC2626",
        linewidth=2.0,
        linestyle="--",
        label="Use E[X] × P(win)",
    )
    ax_share.plot(
        propensity_curve["top_share_selected"],
        propensity_curve["capture_rate"],
        color="#7C3AED",
        linewidth=2.0,
        linestyle=":",
        label="Use P(win)",
    )
    ax_share.set_ylabel("Captured share of total won amount")
    ax_share.set_xlabel("Top share of opportunities kept by threshold")
    ax_share.xaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax_share.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax_share.set_ylim(0, 1.02)
    for mark in [0.10, 0.25, 0.50]:
        row = ev_curve.loc[(ev_curve["top_share_selected"] - mark).abs().idxmin()]
        prop_row = propensity_curve.loc[
            (propensity_curve["top_share_selected"] - mark).abs().idxmin()
        ]
        ax_share.scatter(
            [row["top_share_selected"]],
            [row["capture_rate"]],
            color="#DC2626",
            s=24,
            zorder=5,
        )
        y_offset = 0.03 if mark < 0.5 else -0.08
        ax_share.text(
            row["top_share_selected"] + 0.012,
            min(0.96, max(0.08, row["capture_rate"] + y_offset)),
            f"EV {int(mark * 100)}%: {row['capture_rate']:.0%}\nP(win): {prop_row['capture_rate']:.0%}",
            color="#0F172A",
            fontsize=8.5,
            weight="bold",
        )

    handles_amount, labels_amount = ax_amount.get_legend_handles_labels()
    handles_share, labels_share = ax_share.get_legend_handles_labels()
    ax_amount.legend(
        handles_amount,
        labels_amount,
        loc="lower right",
        frameon=False,
    )
    ax_share.legend(
        handles_share,
        labels_share,
        loc="lower right",
        ncol=1,
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved ({variant}): {output_path}")


if __name__ == "__main__":
    main()

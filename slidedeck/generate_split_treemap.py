from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/intermediate/df_train_stratified.parquet"
TEST_PATH = ROOT / "data/intermediate/df_test_stratified.parquet"
ASSET_PATH = ROOT / "slidedeck/assets/model_split_treemap.png"

RESULT_ORDER = ["Loss", "Won"]
AMOUNT_ORDER = [
    "10K or less",
    "10K to 20K",
    "20K to 30K",
    "30K to 40K",
    "40K to 50K",
    "50K to 60K",
    "More than 60K",
]
AMOUNT_COLORS = {
    "10K or less": "#DBEAFE",
    "10K to 20K": "#BFDBFE",
    "20K to 30K": "#93C5FD",
    "30K to 40K": "#60A5FA",
    "40K to 50K": "#3B82F6",
    "50K to 60K": "#2563EB",
    "More than 60K": "#1D4ED8",
}
RESULT_EDGE = {"Loss": "#0F172A", "Won": "#059669"}


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    grouped = df.groupby(
        ["Opportunity Result", "Deal Size Category (USD)"],
        dropna=False,
        as_index=False,
    ).agg(count=("Opportunity Result", "size"))
    grouped["split"] = split_name
    return grouped


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def draw_split(
    ax: plt.Axes,
    data: pd.DataFrame,
    split_label: str,
    split_share: float,
) -> None:
    split_total = int(data["count"].sum())
    y_top = 1.0

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.axis("off")

    ax.text(
        0.5,
        1.025,
        f"{split_label}\n{split_total:,} rows ({fmt_pct(split_share)})",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
        color="#0F172A",
    )

    for result in RESULT_ORDER:
        result_data = data.loc[data["Opportunity Result"] == result].copy()
        result_total = int(result_data["count"].sum())
        result_height = result_total / split_total if split_total else 0
        y_bottom = y_top - result_height

        ax.add_patch(
            Rectangle(
                (0, y_bottom),
                1,
                result_height,
                fill=False,
                edgecolor=RESULT_EDGE[result],
                linewidth=2.2,
            )
        )

        running_x = 0.0
        for amount_bucket in AMOUNT_ORDER:
            bucket_rows = result_data.loc[
                result_data["Deal Size Category (USD)"] == amount_bucket, "count"
            ]
            bucket_count = int(bucket_rows.iloc[0]) if not bucket_rows.empty else 0
            if bucket_count == 0:
                continue

            bucket_width = bucket_count / result_total if result_total else 0
            ax.add_patch(
                Rectangle(
                    (running_x, y_bottom),
                    bucket_width,
                    result_height,
                    facecolor=AMOUNT_COLORS[amount_bucket],
                    edgecolor="white",
                    linewidth=1.0,
                )
            )

            if bucket_width > 0.11 and result_height > 0.13:
                ax.text(
                    running_x + bucket_width / 2,
                    y_bottom + result_height / 2,
                    f"{amount_bucket}\n{fmt_pct(bucket_count / split_total)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#0F172A",
                )

            running_x += bucket_width

        ax.text(
            0.012,
            y_bottom + result_height - 0.018,
            f"{result}: {fmt_pct(result_total / split_total)}",
            ha="left",
            va="top",
            fontsize=9,
            weight="bold",
            color=RESULT_EDGE[result],
        )
        y_top = y_bottom


def main() -> None:
    train = load_split(TRAIN_PATH, "Train")
    test = load_split(TEST_PATH, "Test")

    train_total = int(train["count"].sum())
    test_total = int(test["count"].sum())
    grand_total = train_total + test_total

    train_share = train_total / grand_total
    test_share = test_total / grand_total

    fig = plt.figure(figsize=(9.4, 5.9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[14, 2], wspace=0.14, hspace=0.02)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")

    draw_split(axes[0], train, "Train split", train_share)
    draw_split(axes[1], test, "Test split", test_share)

    legend_handles = [
        Patch(facecolor=AMOUNT_COLORS[b], edgecolor="white", label=b)
        for b in AMOUNT_ORDER
    ]
    legend_ax.legend(
        handles=legend_handles,
        title="Deal-size strata",
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {ASSET_PATH}")


if __name__ == "__main__":
    main()

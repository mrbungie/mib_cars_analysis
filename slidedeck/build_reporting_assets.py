from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
SLIDE_DATA_DIR = ROOT / "slidedeck/data"
ASSET_DIR = ROOT / "slidedeck/assets"
CLASSIFICATION_PATH = SLIDE_DATA_DIR / "classification_model_report.xlsx"
REGRESSION_PATH = SLIDE_DATA_DIR / "regression_model_report.xlsx"
MODEL_SLIDE_DATA_PATH = SLIDE_DATA_DIR / "model_slide_data.xlsx"
MODEL_METRIC_TABLES_PATH = SLIDE_DATA_DIR / "model_metric_tables.xlsx"


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


def build_workbooks() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    classification_summary = pd.read_excel(CLASSIFICATION_PATH, sheet_name="cv_summary")
    classification_cv_selected = pd.read_excel(
        CLASSIFICATION_PATH, sheet_name="cv_selected_model"
    )
    classification_cv_baseline = pd.read_excel(
        CLASSIFICATION_PATH, sheet_name="cv_baseline_model"
    )
    classification_test_metrics = pd.read_excel(
        CLASSIFICATION_PATH, sheet_name="test_metrics"
    )
    classification_importance = pd.read_excel(
        CLASSIFICATION_PATH, sheet_name="feature_importance"
    ).head(10)
    classification_lift = pd.read_excel(
        CLASSIFICATION_PATH, sheet_name="prioritization_lift"
    )
    classification_metadata = pd.read_excel(CLASSIFICATION_PATH, sheet_name="metadata")

    regression_summary = pd.read_excel(REGRESSION_PATH, sheet_name="cv_summary")
    regression_cv_selected = pd.read_excel(
        REGRESSION_PATH, sheet_name="cv_selected_model"
    )
    regression_cv_baseline = pd.read_excel(
        REGRESSION_PATH, sheet_name="cv_baseline_model"
    )
    regression_test_metrics = pd.read_excel(REGRESSION_PATH, sheet_name="test_metrics")
    regression_importance = pd.read_excel(
        REGRESSION_PATH, sheet_name="feature_importance"
    ).head(10)
    regression_forecast = pd.read_excel(REGRESSION_PATH, sheet_name="forecast_summary")
    regression_predictions = pd.read_excel(
        REGRESSION_PATH, sheet_name="test_predictions"
    )
    regression_metadata = pd.read_excel(REGRESSION_PATH, sheet_name="metadata")

    selected_models = pd.DataFrame(
        [
            {
                "model": "Dynamic win/loss classification",
                "target": "Opportunity Result",
                "business_use": "Prioritization and pipeline ranking",
                "data_used": "All available snapshot variables, including process timing fields",
                "notes": f"Selected model: {classification_metadata.loc[0, 'selected_experiment']} | Baseline: {classification_metadata.loc[0, 'baseline_experiment']}",
            },
            {
                "model": "Dynamic amount regression",
                "target": "Opportunity Amount USD",
                "business_use": "Sizing and aggregate forecasting",
                "data_used": "All available snapshot variables, including process timing fields",
                "notes": f"Selected model: {regression_metadata.loc[0, 'selected_experiment']} | Baseline: {regression_metadata.loc[0, 'baseline_experiment']}",
            },
        ]
    )

    with pd.ExcelWriter(MODEL_SLIDE_DATA_PATH, engine="openpyxl") as writer:
        selected_models.to_excel(writer, sheet_name="selected_models", index=False)
        classification_summary.to_excel(
            writer, sheet_name="classification_summary", index=False
        )
        classification_cv_selected.to_excel(
            writer, sheet_name="classification_cv_selected", index=False
        )
        classification_cv_baseline.to_excel(
            writer, sheet_name="classification_cv_baseline", index=False
        )
        classification_test_metrics.to_excel(
            writer, sheet_name="classification_test_metrics", index=False
        )
        classification_importance.to_excel(
            writer, sheet_name="classification_importance", index=False
        )
        classification_lift.to_excel(
            writer, sheet_name="prioritization_lift", index=False
        )
        classification_metadata.to_excel(
            writer, sheet_name="classification_metadata", index=False
        )
        regression_summary.to_excel(
            writer, sheet_name="regression_summary", index=False
        )
        regression_cv_selected.to_excel(
            writer, sheet_name="regression_cv_selected", index=False
        )
        regression_cv_baseline.to_excel(
            writer, sheet_name="regression_cv_baseline", index=False
        )
        regression_test_metrics.to_excel(
            writer, sheet_name="regression_test_metrics", index=False
        )
        regression_importance.to_excel(
            writer, sheet_name="regression_importance", index=False
        )
        regression_forecast.to_excel(writer, sheet_name="forecast_summary", index=False)
        regression_predictions.head(5000).to_excel(
            writer, sheet_name="forecast_predictions_sample", index=False
        )
        regression_metadata.to_excel(
            writer, sheet_name="regression_metadata", index=False
        )

    classification_metric_table = pd.concat(
        [
            classification_cv_selected.assign(model_role="selected"),
            classification_cv_baseline.assign(model_role="baseline"),
        ],
        ignore_index=True,
    )
    regression_metric_table = pd.concat(
        [
            regression_cv_selected.assign(model_role="selected"),
            regression_cv_baseline.assign(model_role="baseline"),
        ],
        ignore_index=True,
    )
    classification_test_table = classification_test_metrics.merge(
        pd.DataFrame(
            {
                "experiment": [
                    classification_metadata.loc[0, "selected_experiment"],
                    classification_metadata.loc[0, "baseline_experiment"],
                ],
                "model_role": ["selected", "baseline"],
            }
        ),
        on="experiment",
        how="left",
    )
    regression_test_table = regression_test_metrics.merge(
        pd.DataFrame(
            {
                "experiment": [
                    regression_metadata.loc[0, "selected_experiment"],
                    regression_metadata.loc[0, "baseline_experiment"],
                ],
                "model_role": ["selected", "baseline"],
            }
        ),
        on="experiment",
        how="left",
    )

    with pd.ExcelWriter(MODEL_METRIC_TABLES_PATH, engine="openpyxl") as writer:
        classification_metric_table.to_excel(
            writer, sheet_name="classification_cv", index=False
        )
        classification_test_table.to_excel(
            writer, sheet_name="classification_test", index=False
        )
        regression_metric_table.to_excel(
            writer, sheet_name="regression_cv", index=False
        )
        regression_test_table.to_excel(
            writer, sheet_name="regression_test", index=False
        )

    return (
        classification_lift,
        regression_predictions,
        classification_importance,
        regression_importance,
    )


def build_assets(
    classification_lift: pd.DataFrame,
    regression_predictions: pd.DataFrame,
    classification_importance: pd.DataFrame,
    regression_importance: pd.DataFrame,
) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")

    roc_df = pd.read_excel(CLASSIFICATION_PATH, sheet_name="roc_curve")
    class_test = pd.read_excel(CLASSIFICATION_PATH, sheet_name="test_metrics")
    class_selected = class_test.iloc[0]

    fig, (ax_roc, ax_imp) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.4),
        gridspec_kw={"height_ratios": [2.8, 1.6], "hspace": 0.34},
    )
    ax_roc.plot(
        roc_df["fpr"],
        roc_df["tpr"],
        color="#2563EB",
        linewidth=2.6,
        label=f"Selected model (AUC = {class_selected['roc_auc']:.3f})",
    )
    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#0F172A",
        linewidth=1.2,
        label="Random baseline",
    )
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("Hold-out ROC curve", fontsize=12, weight="bold")
    ax_roc.legend(frameon=False, loc="lower right")
    ax_roc.grid(alpha=0.25)

    class_plot = (
        classification_importance.sort_values("importance_mean", ascending=False)
        .head(4)
        .copy()
    )
    class_plot["feature_label"] = (
        class_plot["feature"]
        .astype(str)
        .str.replace(
            "Ratio Days Identified To Total Days",
            "Identified / total days",
            regex=False,
        )
        .str.replace(
            "Ratio Days Qualified To Total Days", "Qualified / total days", regex=False
        )
        .str.replace(
            "Ratio Days Validated To Total Days", "Validated / total days", regex=False
        )
        .str.replace(
            "Total Days Identified Through Qualified",
            "Days: identified → qualified",
            regex=False,
        )
        .str.replace(
            "Total Days Identified Through Closing",
            "Days: identified → closing",
            regex=False,
        )
        .str.replace(
            "Revenue From Client Past Two Years (USD)", "Revenue past 2y", regex=False
        )
        .str.replace("Client Size By Revenue (USD)", "Client size revenue", regex=False)
        .str.replace(
            "Elapsed Days In Sales Stage", "Elapsed days in stage", regex=False
        )
        .str.replace("Sales Stage Change Count", "Stage change count", regex=False)
        .str.slice(0, 28)
    )
    sns.barplot(
        data=class_plot.sort_values("importance_mean", ascending=True),
        x="importance_mean",
        y="feature_label",
        color="#2563EB",
        errorbar=None,
        ax=ax_imp,
    )
    ax_imp.set_title("Top feature importance", fontsize=10, weight="bold")
    ax_imp.set_xlabel("Importance")
    ax_imp.set_ylabel("")
    ax_imp.tick_params(axis="y", labelsize=9)
    ax_imp.tick_params(axis="x", labelsize=8)
    ax_imp.grid(False)
    fig.subplots_adjust(left=0.31, right=0.97, top=0.93, bottom=0.11, hspace=0.42)
    fig.savefig(ASSET_DIR / "classification_roc.png", dpi=220, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.barplot(
        data=classification_importance.sort_values("importance_mean", ascending=True),
        x="importance_mean",
        y="feature",
        color="#2563EB",
        ax=ax,
    )
    ax.set_title("Classification model importance", fontsize=12, weight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(
        ASSET_DIR / "classification_importance.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    reg_plot = regression_importance.copy().sort_values(
        "importance_mean", ascending=True
    )
    reg_plot["display_feature"] = (
        reg_plot["feature"]
        .astype(str)
        .str.replace("categorical__", "", regex=False)
        .str.replace("numerical__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )
    sns.barplot(
        data=reg_plot, x="importance_mean", y="display_feature", color="#14B8A6", ax=ax
    )
    ax.set_title("Regression model importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "regression_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lift = classification_lift.sort_values("score_decile")
    plot_lift["lift"] = (
        plot_lift["observed_win_rate"] / plot_lift["observed_win_rate"].mean()
    )
    sns.barplot(data=plot_lift, x="score_decile", y="lift", color="#2563EB", ax=ax)
    ax.axhline(
        1.0,
        color="#0F172A",
        linestyle="--",
        linewidth=1.8,
        label="Average held-out win rate = 1.0×",
    )
    ax.set_xlabel("Propensity-score decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Lift vs. overall held-out win rate")
    ax.set_title("Held-out lift by propensity decile")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "sim_prioritization.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = regression_predictions.sample(
        min(len(regression_predictions), 2000), random_state=42
    )
    sns.scatterplot(
        data=plot_df,
        x="actual_amount",
        y="predicted_amount",
        alpha=0.5,
        s=30,
        color="#2563EB",
        ax=ax,
    )
    max_val = float(
        max(plot_df["actual_amount"].max(), plot_df["predicted_amount"].max())
    )
    ax.plot(
        [0, max_val],
        [0, max_val],
        linestyle="--",
        color="#0F172A",
        label="Perfect forecast",
    )
    ax.set_xlabel("Actual amount")
    ax.set_ylabel("Predicted amount")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "sim_forecasting.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    outputs = build_workbooks()
    build_assets(*outputs)
    print(f"saved: {MODEL_SLIDE_DATA_PATH}")
    print(f"saved: {MODEL_METRIC_TABLES_PATH}")
    print("assets regenerated")


if __name__ == "__main__":
    main()

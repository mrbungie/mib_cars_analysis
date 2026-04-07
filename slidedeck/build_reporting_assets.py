from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from optbinning import ContinuousOptimalBinning, OptimalBinning
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


ROOT = Path(__file__).resolve().parents[1]
SLIDE_DATA_DIR = ROOT / "slidedeck/data"
ASSET_DIR = ROOT / "slidedeck/assets"
CLASSIFICATION_PATH = SLIDE_DATA_DIR / "classification_model_report.xlsx"
REGRESSION_PATH = SLIDE_DATA_DIR / "regression_model_report.xlsx"
MODEL_SLIDE_DATA_PATH = SLIDE_DATA_DIR / "model_slide_data.xlsx"
MODEL_METRIC_TABLES_PATH = SLIDE_DATA_DIR / "model_metric_tables.xlsx"
TRAIN_PATH = ROOT / "data/intermediate/df_train_stratified.parquet"

MODEL_FEATURES = [
    "Supplies Group",
    "Supplies Subgroup",
    "Region",
    "Route To Market",
    "Elapsed Days In Sales Stage",
    "Sales Stage Change Count",
    "Total Days Identified Through Closing",
    "Total Days Identified Through Qualified",
    "Client Size By Revenue (USD)",
    "Client Size By Employee Count",
    "Revenue From Client Past Two Years (USD)",
    "Competitor Type",
    "Ratio Days Identified To Total Days",
    "Ratio Days Validated To Total Days",
    "Ratio Days Qualified To Total Days",
]
CLASSIFICATION_TARGET = "Opportunity Result Bool"
REGRESSION_TARGET = "Opportunity Amount USD"


def tidy_feature_label(name: str) -> str:
    return (
        str(name)
        .replace("Ratio Days Identified To Total Days", "Identified / total days")
        .replace("Ratio Days Qualified To Total Days", "Qualified / total days")
        .replace("Ratio Days Validated To Total Days", "Validated / total days")
        .replace(
            "Total Days Identified Through Qualified", "Days: identified → qualified"
        )
        .replace("Total Days Identified Through Closing", "Days: identified → closing")
        .replace("Revenue From Client Past Two Years (USD)", "Revenue past 2y")
        .replace("Client Size By Revenue (USD)", "Client size revenue")
        .replace("Client Size By Employee Count", "Client size employees")
        .replace("Elapsed Days In Sales Stage", "Elapsed days in stage")
        .replace("Sales Stage Change Count", "Stage change count")
        .replace("Route To Market", "Route to market")
    )


def _feature_dtype(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    return "categorical"


def _monotonic_profile(feature: pd.Series, target: pd.Series) -> dict[str, object]:
    if _feature_dtype(feature) != "numerical":
        return {"direction": "unclear", "strict_monotonic": False}

    clean = pd.DataFrame({"x": feature, "y": target}).dropna().copy()
    if clean.empty or clean["x"].nunique(dropna=True) <= 2:
        return {"direction": "unclear", "strict_monotonic": False}

    try:
        clean["bin"] = pd.qcut(
            clean["x"], q=min(10, clean["x"].nunique()), duplicates="drop"
        )
    except ValueError:
        return {"direction": "unclear", "strict_monotonic": False}

    grouped = (
        clean.groupby("bin", observed=False)
        .agg(x_mean=("x", "mean"), y_mean=("y", "mean"))
        .dropna()
        .sort_values("x_mean")
    )
    if len(grouped) < 3:
        return {"direction": "unclear", "strict_monotonic": False}

    y_values = grouped["y_mean"].to_numpy(dtype=float)
    diffs = np.diff(y_values)
    non_zero_diffs = diffs[np.abs(diffs) > 1e-12]

    strict_increasing = bool(non_zero_diffs.size > 0 and np.all(non_zero_diffs > 0))
    strict_decreasing = bool(non_zero_diffs.size > 0 and np.all(non_zero_diffs < 0))
    strict_monotonic = strict_increasing or strict_decreasing

    rho = float(
        pd.Series(grouped["x_mean"]).corr(
            pd.Series(grouped["y_mean"]), method="spearman"
        )
    )
    if np.isnan(rho):
        direction = "unclear"
    elif strict_increasing or rho >= 0.3:
        direction = "increasing"
    elif strict_decreasing or rho <= -0.3:
        direction = "decreasing"
    else:
        direction = "unclear"

    return {"direction": direction, "strict_monotonic": strict_monotonic}


def _compute_binary_iv(
    feature: pd.Series, target: pd.Series, feature_name: str
) -> float:
    clean = pd.DataFrame({"x": feature, "y": target}).dropna()
    if clean.empty or clean["x"].nunique(dropna=True) <= 1:
        return np.nan
    optb = OptimalBinning(name=feature_name, dtype=_feature_dtype(clean["x"]))
    optb.fit(clean["x"], clean["y"].astype(int))
    table = optb.binning_table.build()
    return float(table.iloc[-1]["IV"]) if "IV" in table.columns else np.nan


def _compute_continuous_iv(
    feature: pd.Series, target: pd.Series, feature_name: str
) -> float:
    clean = pd.DataFrame({"x": feature, "y": target}).dropna()
    if clean.empty or clean["x"].nunique(dropna=True) <= 1:
        return np.nan
    cob = ContinuousOptimalBinning(name=feature_name, dtype=_feature_dtype(clean["x"]))
    cob.fit(clean["x"], clean["y"].astype(float))
    table = cob.binning_table.build()
    return float(table.iloc[-1]["IV"]) if "IV" in table.columns else np.nan


def build_eda_iv_assets() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    classification_rows = []
    regression_rows = []
    for feature_name in MODEL_FEATURES:
        feature = train_df[feature_name]
        classification_rows.append(
            {
                "feature": feature_name,
                "iv": _compute_binary_iv(
                    feature, train_df[CLASSIFICATION_TARGET], feature_name
                ),
                **_monotonic_profile(feature, train_df[CLASSIFICATION_TARGET]),
            }
        )
        regression_rows.append(
            {
                "feature": feature_name,
                "iv": _compute_continuous_iv(
                    feature, train_df[REGRESSION_TARGET], feature_name
                ),
                **_monotonic_profile(feature, train_df[REGRESSION_TARGET]),
            }
        )

    def save_plot(rows: list[dict[str, object]], title: str, output_name: str):
        plot_df = (
            pd.DataFrame(rows).sort_values("iv", ascending=True).reset_index(drop=True)
        )
        color_map = {
            "increasing": "#16A34A",
            "decreasing": "#DC2626",
            "unclear": "#6B7280",
        }
        plot_df["feature_label"] = plot_df.apply(
            lambda row: (
                tidy_feature_label(row["feature"])
                + (" *" if row["strict_monotonic"] else "")
            ),
            axis=1,
        )
        plot_df["bar_color"] = plot_df["direction"].map(color_map).fillna("#6B7280")
        fig_height = max(5.8, 0.32 * len(plot_df) + 1.2)
        fig, ax = plt.subplots(figsize=(8.6, fig_height))
        ax.barh(plot_df["feature_label"], plot_df["iv"], color=plot_df["bar_color"])
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Information value")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)
        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor="#16A34A", label="Increasing relationship"),
            Patch(facecolor="#DC2626", label="Decreasing relationship"),
            Patch(facecolor="#6B7280", label="Categorical / unclear"),
        ]
        ax.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(ASSET_DIR / output_name, dpi=220, bbox_inches="tight")
        plt.close(fig)

    save_plot(
        classification_rows,
        "IV by feature for win/loss target",
        "eda_iv_classification.png",
    )
    save_plot(
        regression_rows,
        "IV by feature for amount target",
        "eda_iv_regression.png",
    )


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
    )
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
    )
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
    build_eda_iv_assets()

    roc_df = pd.read_excel(CLASSIFICATION_PATH, sheet_name="roc_curve")
    class_test = pd.read_excel(CLASSIFICATION_PATH, sheet_name="test_metrics")
    class_selected = class_test.iloc[0]

    fig, ax_roc = plt.subplots(figsize=(7.2, 4.8))
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

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "classification_roc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

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

    class_compare = pd.read_excel(
        CLASSIFICATION_PATH, sheet_name="comparison_test_predictions"
    )
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for experiment_name, color, label in [
        ("logit_binned_ohe_balanced", "#0F172A", "Non-calibrated"),
        ("logit_binned_ohe_balanced_calibrated", "#2563EB", "Calibrated"),
    ]:
        subset = class_compare.loc[
            class_compare["experiment"] == experiment_name
        ].copy()
        frac_pos, mean_pred = calibration_curve(
            subset["actual_result"],
            subset["predicted_win_probability"],
            n_bins=10,
            strategy="quantile",
        )
        ax.plot(
            mean_pred, frac_pos, marker="o", linewidth=2.2, label=label, color=color
        )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#94A3B8",
        linewidth=1.2,
        label="Perfect calibration",
    )
    ax.set_xlabel("Mean predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Hold-out calibration: calibrated vs non-calibrated logit")
    ax.xaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(
        ASSET_DIR / "annex_calibration_comparison.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    selected_predictions = pd.read_excel(
        CLASSIFICATION_PATH, sheet_name="test_predictions"
    )
    class_threshold_compare = pd.read_excel(
        CLASSIFICATION_PATH, sheet_name="comparison_test_predictions"
    )
    class_metadata = pd.read_excel(CLASSIFICATION_PATH, sheet_name="metadata")
    train_selected_threshold = float(class_metadata.loc[0, "train_selected_threshold"])
    comparison_rows = []
    label_map = {
        "logit_binned_ohe_balanced_calibrated": "Selected @ optimal threshold",
        "random_classifier": "Random classifier",
        "always_true_classifier": "Always true",
    }
    for experiment_name, label in label_map.items():
        subset = class_threshold_compare.loc[
            class_threshold_compare["experiment"] == experiment_name
        ].copy()
        if experiment_name == "logit_binned_ohe_balanced_calibrated":
            predicted = (
                subset["predicted_win_probability"] >= train_selected_threshold
            ).astype(int)
            threshold_value = train_selected_threshold
        else:
            predicted = subset["predicted_label"].astype(int)
            threshold_value = np.nan
        tn, fp, fn, tp = confusion_matrix(
            subset["actual_result"], predicted, labels=[0, 1]
        ).ravel()
        comparison_rows.append(
            {
                "model": label,
                "threshold": threshold_value,
                "accuracy": accuracy_score(subset["actual_result"], predicted),
                "precision": precision_score(
                    subset["actual_result"], predicted, zero_division=0
                ),
                "recall": recall_score(
                    subset["actual_result"], predicted, zero_division=0
                ),
                "f1": f1_score(subset["actual_result"], predicted, zero_division=0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    optimal_threshold_df = pd.DataFrame(comparison_rows)
    selected_threshold_row = optimal_threshold_df.loc[
        optimal_threshold_df["model"] == "Selected @ optimal threshold"
    ].iloc[0]

    cm = np.array(
        [
            [selected_threshold_row["tn"], selected_threshold_row["fp"]],
            [selected_threshold_row["fn"], selected_threshold_row["tp"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar=False,
        ax=ax,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    ax.set_title("Confusion matrix at best threshold", fontsize=12, weight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticklabels(["Loss", "Win"])
    ax.set_yticklabels(["Loss", "Win"], rotation=0)
    fig.tight_layout()
    fig.savefig(
        ASSET_DIR / "classification_confusion_matrix.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    comparison_predictions = pd.read_excel(
        REGRESSION_PATH, sheet_name="comparison_test_predictions"
    )
    amount_order = [
        "10K or less",
        "10K to 20K",
        "20K to 30K",
        "30K to 40K",
        "40K to 50K",
        "50K to 60K",
        "More than 60K",
    ]
    amount_labels = {
        "10K or less": "≤10K",
        "10K to 20K": "10–20K",
        "20K to 30K": "20–30K",
        "30K to 40K": "30–40K",
        "40K to 50K": "40–50K",
        "50K to 60K": "50–60K",
        "More than 60K": ">60K",
    }
    error_summary = comparison_predictions.groupby(
        ["amount_bin", "experiment"], as_index=False
    ).agg(mae=("absolute_error", "mean"), median_ape=("ape", "median"))
    error_summary["amount_bin"] = pd.Categorical(
        error_summary["amount_bin"], categories=amount_order, ordered=True
    )
    error_summary = error_summary.sort_values("amount_bin")
    error_summary["amount_label"] = (
        error_summary["amount_bin"].astype(str).map(amount_labels)
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(8.2, 6.2), sharex=True, gridspec_kw={"hspace": 0.18}
    )
    experiment_labels = {
        "xgboost_boxcox": "XGBoost Box-Cox",
        "xgboost_log1p": "XGBoost log1p",
        "xgboost_yeojohnson": "XGBoost Yeo-Johnson",
        "random_forest_regressor": "Random Forest",
        "xgboost": "XGBoost",
        "classic_linear_boxcox_robust_scaler": "Linear Box-Cox",
    }
    palette = {
        "xgboost_boxcox": "#2563EB",
        "xgboost_log1p": "#14B8A6",
        "xgboost_yeojohnson": "#7C3AED",
        "random_forest_regressor": "#F59E0B",
        "xgboost": "#0F172A",
        "classic_linear_boxcox_robust_scaler": "#DC2626",
    }
    for experiment_name, label in experiment_labels.items():
        subset = error_summary.loc[error_summary["experiment"] == experiment_name]
        axes[0].plot(
            subset["amount_label"],
            subset["mae"],
            marker="o",
            linewidth=2,
            label=label,
            color=palette[experiment_name],
        )
        axes[1].plot(
            subset["amount_label"],
            subset["median_ape"],
            marker="o",
            linewidth=2,
            label=label,
            color=palette[experiment_name],
        )
    axes[0].set_ylabel("MAE")
    axes[0].yaxis.set_major_formatter(FuncFormatter(usd_compact))
    axes[1].set_ylabel("Median APE")
    axes[1].yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    axes[1].set_xlabel("Amount bin")
    axes[1].tick_params(axis="x", rotation=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Hold-out error by amount bin", fontsize=13, fontweight="bold", y=0.98)
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.subplots_adjust(top=0.8, bottom=0.1, hspace=0.2)
    fig.savefig(
        ASSET_DIR / "annex_regression_error_bins.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    with pd.ExcelWriter(
        MODEL_METRIC_TABLES_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        optimal_threshold_df.to_excel(
            writer, sheet_name="cls_optimal_threshold", index=False
        )


def main() -> None:
    outputs = build_workbooks()
    build_assets(*outputs)
    print(f"saved: {MODEL_SLIDE_DATA_PATH}")
    print(f"saved: {MODEL_METRIC_TABLES_PATH}")
    print("assets regenerated")


if __name__ == "__main__":
    main()

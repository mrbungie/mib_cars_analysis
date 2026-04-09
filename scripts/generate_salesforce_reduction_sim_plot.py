import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from reporting_variant_paths import (
    asset_path,
    data_path,
    model_report_path,
    validate_variant,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data/intermediate/df_test_stratified.parquet"

OTE_PER_REP = 120_000
BASE_OVERHEAD_PER_REP = 135_000
COMMISSION_RATE = 0.07
FULLY_LOADED_COST_PER_REP = BASE_OVERHEAD_PER_REP
OPPS_PER_REP = 100
MARGIN_BY_SUPPLIES_GROUP = {
    "Car Accessories": 0.68,
    "Performance & Non-auto": 0.62,
    "Tires & Wheels": 0.55,
    "Car Electronics": 0.58,
}
DEFAULT_MARGIN_PCT = 0.63
SERVICE_COST_PCT = 0.08
ACQUISITION_COST_BY_ROUTE = {
    "Fields Sales": 2_400,
    "Reseller": 1_600,
    "Telesales": 850,
    "Telecoverage": 650,
    "Other": 1_100,
}
DEFAULT_ACQUISITION_COST = 1_000


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


def pct_fmt(value: float, _position: float = 0.0) -> str:
    return f"{value:.0%}"


def load_merged(variant: str) -> pd.DataFrame:
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
    enrichment = pd.read_parquet(TEST_PATH).reset_index(drop=True)
    enrichment = enrichment.reset_index(names="row_id")
    keep_cols = [
        "row_id",
        "Supplies Group",
        "Route To Market",
        "Elapsed Days In Sales Stage",
        "Sales Stage Change Count",
        "Total Days Identified Through Closing",
        "Total Days Identified Through Qualified",
        "Client Size By Revenue (USD)",
        "Client Size By Employee Count",
        "Opportunity Amount USD",
    ]
    df = df.merge(enrichment[keep_cols], on="row_id", how="left", validate="one_to_one")
    df["expected_value"] = df["predicted_win_probability"] * df["predicted_amount"]
    df["actual_won_amount"] = df["actual_amount"] * df["actual_result"]
    return add_contribution_components(df)


def add_contribution_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["margin_pct"] = (
        out["Supplies Group"].map(MARGIN_BY_SUPPLIES_GROUP).fillna(DEFAULT_MARGIN_PCT)
    )
    out["service_cost_pct"] = SERVICE_COST_PCT
    out["acquisition_cost"] = (
        out["Route To Market"]
        .map(ACQUISITION_COST_BY_ROUTE)
        .fillna(DEFAULT_ACQUISITION_COST)
    )

    out["commission_cost"] = out["actual_won_amount"] * COMMISSION_RATE
    out["gross_margin_value"] = out["actual_won_amount"] * out["margin_pct"]
    out["serving_cost"] = out["actual_won_amount"] * out["service_cost_pct"]
    return out


def simulate_reduction(df: pd.DataFrame) -> pd.DataFrame:
    total_opps = len(df)
    baseline_reps = int(np.ceil(total_opps / OPPS_PER_REP))
    baseline_cost = baseline_reps * FULLY_LOADED_COST_PER_REP
    baseline_revenue = float(df["actual_won_amount"].sum())
    baseline_gross_margin = float(df["gross_margin_value"].sum())
    baseline_effort_cost = float(baseline_reps * BASE_OVERHEAD_PER_REP)
    baseline_acquisition_cost = float(df["acquisition_cost"].sum())
    baseline_commission_cost = float(df["commission_cost"].sum())
    baseline_serving_cost = float(df["serving_cost"].sum())
    baseline_total_cost = (
        baseline_effort_cost
        + baseline_commission_cost
        + baseline_acquisition_cost
        + baseline_serving_cost
    )
    baseline_contribution = float(baseline_gross_margin - baseline_total_cost)

    ranked = df.sort_values("expected_value", ascending=False).reset_index(drop=True)

    rows = []
    for n_reps in range(1, baseline_reps + 1):
        opps_covered = min(n_reps * OPPS_PER_REP, total_opps)
        selected = ranked.iloc[:opps_covered]
        last_batch = (
            ranked.iloc[opps_covered - OPPS_PER_REP : opps_covered]
            if opps_covered > OPPS_PER_REP
            else selected
        )
        revenue = float(selected["actual_won_amount"].sum())
        gross_margin = float(selected["gross_margin_value"].sum())
        effort_cost = float(n_reps * BASE_OVERHEAD_PER_REP)
        commission_cost = float(selected["commission_cost"].sum())
        acquisition_cost = float(selected["acquisition_cost"].sum())
        serving_cost = float(selected["serving_cost"].sum())
        contribution = float(
            gross_margin
            - effort_cost
            - commission_cost
            - acquisition_cost
            - serving_cost
        )
        contribution_vs_baseline = contribution - baseline_contribution
        marginal_gross_margin = float(last_batch["gross_margin_value"].sum())
        marginal_commission = float(last_batch["commission_cost"].sum())
        marginal_acquisition = float(last_batch["acquisition_cost"].sum())
        marginal_serving = float(last_batch["serving_cost"].sum())
        marginal_cost = float(
            BASE_OVERHEAD_PER_REP
            + marginal_commission
            + marginal_acquisition
            + marginal_serving
        )
        marginal_contribution = float(marginal_gross_margin - marginal_cost)

        rows.append(
            {
                "n_reps": n_reps,
                "opps_covered": opps_covered,
                "revenue_retained": revenue,
                "gross_margin_retained": gross_margin,
                "sales_effort_cost": effort_cost,
                "commission_cost": commission_cost,
                "acquisition_cost": acquisition_cost,
                "serving_cost": serving_cost,
                "salesforce_cost": effort_cost
                + commission_cost
                + acquisition_cost
                + serving_cost,
                "net_margin": contribution,
                "net_margin_vs_baseline": contribution_vs_baseline,
                "revenue_share_retained": revenue / baseline_revenue,
                "contribution_share_of_baseline": contribution / baseline_contribution,
                "cost_share_of_baseline": (
                    (effort_cost + commission_cost + acquisition_cost + serving_cost)
                    / max(
                        baseline_effort_cost
                        + baseline_commission_cost
                        + baseline_acquisition_cost
                        + baseline_serving_cost,
                        1.0,
                    )
                )
                if baseline_total_cost
                else np.nan,
                "marginal_revenue_last_batch": marginal_contribution,
                "marginal_cost_last_rep": marginal_cost,
                "marginal_roi": marginal_contribution / max(marginal_cost, 1.0),
            }
        )

    result = pd.DataFrame(rows)
    result["baseline_reps"] = baseline_reps
    result["baseline_cost"] = baseline_cost
    result["baseline_revenue"] = baseline_revenue
    result["baseline_gross_margin"] = baseline_gross_margin
    result["baseline_effort_cost"] = baseline_effort_cost
    result["baseline_commission_cost"] = baseline_commission_cost
    result["baseline_acquisition_cost"] = baseline_acquisition_cost
    result["baseline_serving_cost"] = baseline_serving_cost
    result["baseline_total_cost"] = baseline_total_cost
    result["baseline_contribution"] = baseline_contribution
    return result


def plot_main(sim: pd.DataFrame) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")

    baseline_reps = int(sim["baseline_reps"].iloc[0])
    baseline_revenue = float(sim["baseline_revenue"].iloc[0])
    baseline_total_cost = float(sim["baseline_total_cost"].iloc[0])
    baseline_contribution = float(sim["baseline_contribution"].iloc[0])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 8.5),
        gridspec_kw={"hspace": 0.38, "wspace": 0.32},
        constrained_layout=True,
    )
    ax_rev, ax_cost = axes[0]
    ax_net, ax_mroi = axes[1]

    ax_rev.plot(
        sim["n_reps"],
        sim["revenue_retained"],
        color="#2563EB",
        linewidth=2.4,
        label="Revenue retained",
    )
    ax_rev.axvline(baseline_reps, color="#94A3B8", linewidth=1.2, linestyle=":")
    ax_rev.axhline(
        baseline_revenue,
        color="#0F172A",
        linewidth=1.1,
        linestyle="--",
        label=f"Baseline = {usd_compact(baseline_revenue)}",
    )
    ax_rev.yaxis.set_major_formatter(mticker.FuncFormatter(usd_compact))
    ax_rev.set_xlabel("Sales reps (headcount)")
    ax_rev.set_ylabel("Revenue retained (USD)")
    ax_rev.set_title("Revenue retained by headcount", weight="bold")
    ax_rev.legend(frameon=False, fontsize=8.5)

    ax_cost.plot(
        sim["n_reps"],
        sim["salesforce_cost"],
        color="#F59E0B",
        linewidth=2.4,
        label="Opportunity operating cost",
    )
    ax_cost.axvline(baseline_reps, color="#94A3B8", linewidth=1.2, linestyle=":")
    ax_cost.axhline(
        baseline_total_cost,
        color="#0F172A",
        linewidth=1.1,
        linestyle="--",
        label=f"Baseline = {usd_compact(baseline_total_cost)}",
    )
    ax_cost.yaxis.set_major_formatter(mticker.FuncFormatter(usd_compact))
    ax_cost.set_xlabel("Sales reps (headcount)")
    ax_cost.set_ylabel("Operating cost (USD)")
    ax_cost.set_title("Operating cost by headcount", weight="bold")
    ax_cost.legend(frameon=False, fontsize=8.5)

    ax_net.plot(
        sim["n_reps"],
        sim["net_margin"],
        color="#14B8A6",
        linewidth=2.4,
        label="Contribution margin",
    )
    ax_net.axhline(
        baseline_contribution,
        color="#0F172A",
        linewidth=1.1,
        linestyle="--",
        label=f"Baseline contribution = {usd_compact(baseline_contribution)}",
    )
    best_row = sim.loc[sim["net_margin"].idxmax()]
    ax_net.scatter(
        [best_row["n_reps"]],
        [best_row["net_margin"]],
        color="#14B8A6",
        s=70,
        zorder=6,
        edgecolors="white",
        linewidths=0.8,
    )
    ax_net.annotate(
        f"Optimal: {int(best_row['n_reps'])} reps\n{usd_compact(best_row['net_margin'])} contribution",
        xy=(best_row["n_reps"], best_row["net_margin"]),
        xytext=(
            best_row["n_reps"] + baseline_reps * 0.06,
            best_row["net_margin"] * 0.97,
        ),
        fontsize=8.5,
        color="#14B8A6",
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": "#14B8A6", "lw": 0.8},
    )
    ax_net.yaxis.set_major_formatter(mticker.FuncFormatter(usd_compact))
    ax_net.set_xlabel("Sales reps (headcount)")
    ax_net.set_ylabel("Contribution margin (USD)")
    ax_net.set_title("Contribution margin by headcount", weight="bold")
    ax_net.legend(frameon=False, fontsize=8.5)

    ax_mroi.plot(
        sim["n_reps"],
        sim["marginal_roi"],
        color="#7C3AED",
        linewidth=2.4,
        label="Marginal ROI (last batch)",
    )
    ax_mroi.axhline(
        1.0,
        color="#DC2626",
        linewidth=1.4,
        linestyle="--",
        label="ROI = 1.0× (incremental contribution pays for incremental cost)",
    )
    ax_mroi.axvline(baseline_reps, color="#94A3B8", linewidth=1.2, linestyle=":")

    be_rows = sim[sim["marginal_roi"] >= 1.0]
    if not be_rows.empty:
        last_be = be_rows.iloc[-1]
        ax_mroi.scatter(
            [last_be["n_reps"]],
            [1.0],
            color="#DC2626",
            s=55,
            zorder=6,
            edgecolors="white",
            linewidths=0.8,
        )
        ax_mroi.text(
            last_be["n_reps"] + baseline_reps * 0.03,
            1.05,
            f"Last ROI≥1 rep: {int(last_be['n_reps'])}",
            fontsize=8.5,
            color="#DC2626",
            weight="bold",
        )

    ax_mroi.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}×"))
    ax_mroi.set_xlabel("Sales reps (headcount)")
    ax_mroi.set_ylabel("Marginal ROI (contribution / incremental cost)")
    ax_mroi.set_title("Marginal ROI of last rep batch", weight="bold")
    ax_mroi.legend(frameon=False, fontsize=8.5)

    fig.suptitle(
        f"Salesforce reduction analysis — holdout set "
        f"(baseline: {baseline_reps} reps, contribution = gross margin − effort − acquisition − serving, "
        f"{OPPS_PER_REP} opps/rep, opportunities ranked by expected value)",
        fontsize=11,
        weight="bold",
        y=1.01,
    )

    return fig


def plot_marginal_curve(sim: pd.DataFrame) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")

    baseline_reps = int(sim["baseline_reps"].iloc[0])

    bar_colors = [
        "#2563EB" if v >= c else "#DC2626"
        for v, c in zip(
            sim["marginal_revenue_last_batch"], sim["marginal_cost_last_rep"]
        )
    ]
    above_patch = plt.Rectangle((0, 0), 1, 1, fc="#2563EB", alpha=0.85)
    below_patch = plt.Rectangle((0, 0), 1, 1, fc="#DC2626", alpha=0.85)

    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)

    ax.bar(
        sim["n_reps"],
        sim["marginal_revenue_last_batch"],
        color=bar_colors,
        width=0.85,
        alpha=0.85,
    )
    ax.legend(
        [above_patch, below_patch],
        [
            "Marginal contribution ≥ incremental cost — keep the rep",
            "Marginal contribution < incremental cost — consider cutting",
        ],
        frameon=False,
        fontsize=8.5,
    )
    ax.axhline(
        0,
        color="#0F172A",
        linewidth=1.6,
        linestyle="--",
    )
    ax.axvline(baseline_reps, color="#94A3B8", linewidth=1.2, linestyle=":")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd_compact))
    ax.set_xlabel(
        "Sales reps (headcount) — bars ordered from highest-EV to lowest-EV batch"
    )
    ax.set_ylabel("Marginal contribution from last batch of 100 opportunities")
    ax.set_title(
        "Marginal contribution per rep batch vs. incremental cost (annex)",
        fontsize=12,
        weight="bold",
    )

    ax.text(
        0.02,
        0.97,
        "Blue bars: marginal contribution exceeds incremental cost. "
        "Red bars: below break-even.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#475569",
        va="top",
    )

    return fig


def main() -> None:
    args = parse_args()
    variant = validate_variant(args.variant)

    df = load_merged(variant)
    sim = simulate_reduction(df)

    sim_csv = data_path(ROOT, "sim_salesforce_reduction.csv", variant)
    sim_csv.parent.mkdir(parents=True, exist_ok=True)
    sim.to_csv(sim_csv, index=False)
    print(f"saved ({variant}): {sim_csv}")

    main_fig = plot_main(sim)
    main_png = asset_path(ROOT, "sim_salesforce_reduction_main.png", variant)
    main_png.parent.mkdir(parents=True, exist_ok=True)
    main_fig.savefig(main_png, dpi=220, bbox_inches="tight")
    plt.close(main_fig)
    print(f"saved ({variant}): {main_png}")

    marginal_fig = plot_marginal_curve(sim)
    marginal_png = asset_path(ROOT, "sim_salesforce_reduction_marginal.png", variant)
    marginal_fig.savefig(marginal_png, dpi=220, bbox_inches="tight")
    plt.close(marginal_fig)
    print(f"saved ({variant}): {marginal_png}")

    baseline_reps = int(sim["baseline_reps"].iloc[0])
    best_row = sim.loc[sim["net_margin"].idxmax()]
    last_roi_row = sim[sim["marginal_roi"] >= 1.0]
    last_roi_rep = int(last_roi_row.iloc[-1]["n_reps"]) if not last_roi_row.empty else 0

    print(
        f"\nBaseline: {baseline_reps} reps, contribution={usd_compact(float(sim['baseline_contribution'].iloc[0]))}"
    )
    print(
        f"Optimal headcount by contribution margin: {int(best_row['n_reps'])} reps → contribution {usd_compact(best_row['net_margin'])}"
    )
    print(f"Last rep batch with ROI ≥ 1: rep #{last_roi_rep}")
    print(
        f"Revenue at optimal: {usd_compact(best_row['revenue_retained'])} "
        f"({best_row['revenue_share_retained']:.1%} of baseline)"
    )


if __name__ == "__main__":
    main()

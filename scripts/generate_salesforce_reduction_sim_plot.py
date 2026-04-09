from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_PATH = ROOT / "slidedeck/data/classification_model_report.xlsx"
REGRESSION_PATH = ROOT / "slidedeck/data/regression_model_report.xlsx"
DATA_DIR = ROOT / "slidedeck/data"
ASSET_DIR = ROOT / "slidedeck/assets"

OTE_PER_REP = 120_000
FULLY_LOADED_COST_PER_REP = 180_000
OPPS_PER_REP = 100


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


def load_merged() -> pd.DataFrame:
    classification = pd.read_excel(CLASSIFICATION_PATH, sheet_name="test_predictions")
    regression = pd.read_excel(REGRESSION_PATH, sheet_name="test_predictions")
    df = classification.merge(
        regression, on="row_id", how="inner", validate="one_to_one"
    )
    df["expected_value"] = df["predicted_win_probability"] * df["predicted_amount"]
    df["actual_won_amount"] = df["actual_amount"] * df["actual_result"]
    return df


def simulate_reduction(df: pd.DataFrame) -> pd.DataFrame:
    total_opps = len(df)
    baseline_reps = int(np.ceil(total_opps / OPPS_PER_REP))
    baseline_cost = baseline_reps * FULLY_LOADED_COST_PER_REP
    baseline_revenue = float(df["actual_won_amount"].sum())

    ranked = df.sort_values("expected_value", ascending=False).reset_index(drop=True)

    rows = []
    for n_reps in range(1, baseline_reps + 1):
        opps_covered = min(n_reps * OPPS_PER_REP, total_opps)
        selected = ranked.iloc[:opps_covered]
        revenue = float(selected["actual_won_amount"].sum())
        cost = n_reps * FULLY_LOADED_COST_PER_REP
        net_profit_vs_baseline = (revenue - cost) - (baseline_revenue - baseline_cost)
        marginal_revenue = float(
            ranked.iloc[opps_covered - OPPS_PER_REP : opps_covered][
                "actual_won_amount"
            ].sum()
            if opps_covered > OPPS_PER_REP
            else selected["actual_won_amount"].sum()
        )

        rows.append(
            {
                "n_reps": n_reps,
                "opps_covered": opps_covered,
                "revenue_retained": revenue,
                "salesforce_cost": cost,
                "net_margin": revenue - cost,
                "net_margin_vs_baseline": net_profit_vs_baseline,
                "revenue_share_retained": revenue / baseline_revenue,
                "cost_share_of_baseline": cost / baseline_cost,
                "marginal_revenue_last_batch": marginal_revenue,
                "marginal_cost_last_rep": float(FULLY_LOADED_COST_PER_REP),
                "marginal_roi": marginal_revenue / FULLY_LOADED_COST_PER_REP,
            }
        )

    result = pd.DataFrame(rows)
    result["baseline_reps"] = baseline_reps
    result["baseline_cost"] = baseline_cost
    result["baseline_revenue"] = baseline_revenue
    return result


def plot_main(sim: pd.DataFrame) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")

    baseline_reps = int(sim["baseline_reps"].iloc[0])
    baseline_revenue = float(sim["baseline_revenue"].iloc[0])
    baseline_cost = float(sim["baseline_cost"].iloc[0])
    baseline_net = baseline_revenue - baseline_cost

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
        label="Salesforce cost (fully-loaded)",
    )
    ax_cost.axvline(baseline_reps, color="#94A3B8", linewidth=1.2, linestyle=":")
    ax_cost.axhline(
        baseline_cost,
        color="#0F172A",
        linewidth=1.1,
        linestyle="--",
        label=f"Baseline = {usd_compact(baseline_cost)}",
    )
    ax_cost.yaxis.set_major_formatter(mticker.FuncFormatter(usd_compact))
    ax_cost.set_xlabel("Sales reps (headcount)")
    ax_cost.set_ylabel("Salesforce cost (USD)")
    ax_cost.set_title("Salesforce cost by headcount", weight="bold")
    ax_cost.legend(frameon=False, fontsize=8.5)

    ax_net.plot(
        sim["n_reps"],
        sim["net_margin"],
        color="#14B8A6",
        linewidth=2.4,
        label="Net margin (revenue − cost)",
    )
    ax_net.axhline(
        baseline_net,
        color="#0F172A",
        linewidth=1.1,
        linestyle="--",
        label=f"Baseline net = {usd_compact(baseline_net)}",
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
        f"Optimal: {int(best_row['n_reps'])} reps\n{usd_compact(best_row['net_margin'])} net",
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
    ax_net.set_ylabel("Net margin (USD)")
    ax_net.set_title("Net margin by headcount", weight="bold")
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
        label="ROI = 1.0× (break-even: each rep pays for itself)",
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
    ax_mroi.set_ylabel("Marginal ROI (revenue / fully-loaded cost per 100 opps)")
    ax_mroi.set_title("Marginal ROI of last rep batch", weight="bold")
    ax_mroi.legend(frameon=False, fontsize=8.5)

    fig.suptitle(
        f"Salesforce reduction analysis — holdout set "
        f"(baseline: {baseline_reps} reps × ${OTE_PER_REP:,.0f} OTE / ${FULLY_LOADED_COST_PER_REP:,.0f} fully-loaded, "
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
        "#2563EB" if v >= FULLY_LOADED_COST_PER_REP else "#DC2626"
        for v in sim["marginal_revenue_last_batch"]
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
            f"Marginal revenue ≥ fully-loaded cost ($180K) — keep the rep",
            f"Marginal revenue < fully-loaded cost ($180K) — consider cutting",
        ],
        frameon=False,
        fontsize=8.5,
    )
    ax.axhline(
        FULLY_LOADED_COST_PER_REP,
        color="#0F172A",
        linewidth=1.6,
        linestyle="--",
    )
    ax.axvline(baseline_reps, color="#94A3B8", linewidth=1.2, linestyle=":")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd_compact))
    ax.set_xlabel(
        "Sales reps (headcount) — bars ordered from highest-EV to lowest-EV batch"
    )
    ax.set_ylabel("Marginal won revenue from last batch of 100 opportunities")
    ax.set_title(
        "Marginal revenue per rep batch vs. rep cost (annex)",
        fontsize=12,
        weight="bold",
    )

    ax.text(
        0.02,
        0.97,
        "Blue bars: marginal revenue exceeds rep cost (keep the rep). "
        "Red bars: below break-even (consider cutting).",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#475569",
        va="top",
    )

    return fig


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    df = load_merged()
    sim = simulate_reduction(df)

    sim_csv = DATA_DIR / "sim_salesforce_reduction.csv"
    sim.to_csv(sim_csv, index=False)
    print(f"saved: {sim_csv}")

    main_fig = plot_main(sim)
    main_png = ASSET_DIR / "sim_salesforce_reduction_main.png"
    main_fig.savefig(main_png, dpi=220, bbox_inches="tight")
    plt.close(main_fig)
    print(f"saved: {main_png}")

    marginal_fig = plot_marginal_curve(sim)
    marginal_png = ASSET_DIR / "sim_salesforce_reduction_marginal.png"
    marginal_fig.savefig(marginal_png, dpi=220, bbox_inches="tight")
    plt.close(marginal_fig)
    print(f"saved: {marginal_png}")

    baseline_reps = int(sim["baseline_reps"].iloc[0])
    best_row = sim.loc[sim["net_margin"].idxmax()]
    last_roi_row = sim[sim["marginal_roi"] >= 1.0]
    last_roi_rep = int(last_roi_row.iloc[-1]["n_reps"]) if not last_roi_row.empty else 0

    print(
        f"\nBaseline: {baseline_reps} reps, cost={usd_compact(float(sim['baseline_cost'].iloc[0]))}"
    )
    print(
        f"Optimal headcount by net margin: {int(best_row['n_reps'])} reps → net {usd_compact(best_row['net_margin'])}"
    )
    print(f"Last rep batch with ROI ≥ 1: rep #{last_roi_rep}")
    print(
        f"Revenue at optimal: {usd_compact(best_row['revenue_retained'])} "
        f"({best_row['revenue_share_retained']:.1%} of baseline)"
    )


if __name__ == "__main__":
    main()

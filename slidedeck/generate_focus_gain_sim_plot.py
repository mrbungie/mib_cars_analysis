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

FEATURED_LAMBDA_MAX = 1.50
SWEEP_LAMBDA_MAXES = [1.1, 1.2, 1.3, 1.5, 2.0, 3.0]


def usd_compact(value: float, _position: float = 0.0) -> str:
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.0f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def pct_fmt(value: float, _position: float) -> str:
    return f"{value:.0%}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_merged() -> pd.DataFrame:
    classification = pd.read_excel(CLASSIFICATION_PATH, sheet_name="test_predictions")
    regression = pd.read_excel(REGRESSION_PATH, sheet_name="test_predictions")
    df = classification.merge(
        regression, on="row_id", how="inner", validate="one_to_one"
    )
    df["expected_value"] = df["predicted_win_probability"] * df["predicted_amount"]
    df["actual_won_amount"] = df["actual_amount"] * df["actual_result"]
    return df


def simulate_focus_gain(
    df: pd.DataFrame,
    lam_max: float,
    steps: int = 200,
) -> pd.DataFrame:
    ranked = df.sort_values("expected_value", ascending=False).reset_index(drop=True)
    total_actual_won = float(df["actual_won_amount"].sum())
    total_opps = len(ranked)

    percentiles = np.linspace(1 / steps, 1.0, steps)
    rows = []
    for p in percentiles:
        keep_n = max(1, int(round(total_opps * p)))
        selected = ranked.iloc[:keep_n]

        lam = 1.0 + (lam_max - 1.0) * (1.0 - p)
        baseline_selected = float(selected["actual_won_amount"].sum())
        focus_revenue = lam * baseline_selected

        rows.append(
            {
                "lambda_max": lam_max,
                "effective_lambda": round(lam, 6),
                "top_share": round(p, 6),
                "selected_opps": keep_n,
                "baseline_selected_revenue": baseline_selected,
                "focus_revenue": focus_revenue,
                "total_unmodified": total_actual_won,
                "net_vs_baseline_usd": focus_revenue - total_actual_won,
                "net_vs_baseline_pct": (focus_revenue / total_actual_won - 1) * 100,
            }
        )
    return pd.DataFrame(rows)


def build_all_curves(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_curve = simulate_focus_gain(df, FEATURED_LAMBDA_MAX)
    sweep_curves = pd.concat(
        [simulate_focus_gain(df, lam) for lam in SWEEP_LAMBDA_MAXES],
        ignore_index=True,
    )
    return main_curve, sweep_curves


PALETTE = {
    1.1: "#60A5FA",
    1.2: "#2563EB",
    1.3: "#14B8A6",
    1.5: "#F59E0B",
    2.0: "#DC2626",
    3.0: "#7C3AED",
}


def _annotate_breakeven(
    ax: plt.Axes, curve: pd.DataFrame, color: str, y_col: str
) -> None:
    above = curve[curve[y_col] >= 0]
    if above.empty:
        return
    be_row = above.iloc[0]
    y_val = float(be_row[y_col])
    ax.scatter(
        [be_row["top_share"]],
        [y_val],
        color=color,
        s=55,
        zorder=6,
        edgecolors="white",
        linewidths=0.8,
    )
    ax.annotate(
        f"Break-even\n@ top {be_row['top_share']:.0%}",
        xy=(be_row["top_share"], y_val),
        xytext=(be_row["top_share"] + 0.05, y_val + 15_000_000),
        fontsize=8.5,
        color=color,
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
    )


def plot_main(
    main_curve: pd.DataFrame,
    df_total: pd.DataFrame,
    total_actual_won: float,
) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 0.9], "hspace": 0.10},
        constrained_layout=True,
    )

    ax_top.plot(
        main_curve["top_share"],
        main_curve["focus_revenue"],
        color="#2563EB",
        linewidth=2.6,
        label=f"Revenue with focus gain (λ_max = {FEATURED_LAMBDA_MAX:.1f}×, varies with slice size)",
    )
    ax_top.plot(
        main_curve["top_share"],
        main_curve["baseline_selected_revenue"],
        color="#14B8A6",
        linewidth=1.8,
        linestyle="--",
        label="Actual won revenue in selected slice (no gain)",
    )
    ax_top.axhline(
        total_actual_won,
        color="#0F172A",
        linewidth=1.4,
        linestyle=":",
        label=f"Total holdout won revenue (all opps) = {usd_compact(total_actual_won)}",
    )
    ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(usd_compact))
    ax_top.set_ylabel("USD")
    ax_top.legend(frameon=False, fontsize=9, loc="lower right")
    ax_top.set_title(
        f"Revenue impact of focusing on top-ranked opportunities (λ_max = {FEATURED_LAMBDA_MAX:.1f}×)",
        fontsize=13,
        weight="bold",
    )

    color_main = "#2563EB"
    ax_bot.fill_between(
        main_curve["top_share"],
        main_curve["net_vs_baseline_usd"],
        0,
        where=main_curve["net_vs_baseline_usd"] >= 0,
        alpha=0.18,
        color=color_main,
        interpolate=True,
    )
    ax_bot.fill_between(
        main_curve["top_share"],
        main_curve["net_vs_baseline_usd"],
        0,
        where=main_curve["net_vs_baseline_usd"] < 0,
        alpha=0.18,
        color="#DC2626",
        interpolate=True,
    )
    ax_bot.plot(
        main_curve["top_share"],
        main_curve["net_vs_baseline_usd"],
        color=color_main,
        linewidth=2.4,
        label=f"Net revenue delta vs. unmodified baseline (λ_max = {FEATURED_LAMBDA_MAX:.1f}×)",
    )
    ax_bot.axhline(0, color="#0F172A", linewidth=1.2, linestyle="--")
    ax_bot.set_ylabel("Net revenue delta vs. baseline (USD)")
    ax_bot.set_xlabel("Top share of opportunities worked (ranked by expected value)")
    ax_bot.xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax_bot.yaxis.set_major_formatter(mticker.FuncFormatter(usd_compact))
    ax_bot.legend(frameon=False, fontsize=9, loc="lower right")

    _annotate_breakeven(ax_bot, main_curve, color_main, "net_vs_baseline_usd")

    peak_row = main_curve.loc[main_curve["focus_revenue"].idxmax()]
    ax_top.scatter(
        [peak_row["top_share"]],
        [peak_row["focus_revenue"]],
        color="#DC2626",
        s=65,
        zorder=7,
        edgecolors="white",
        linewidths=0.9,
    )
    ax_top.annotate(
        f"Peak: {usd_compact(peak_row['focus_revenue'])}\n"
        f"@ top {peak_row['top_share']:.0%} (λ={peak_row['effective_lambda']:.2f}×)",
        xy=(peak_row["top_share"], peak_row["focus_revenue"]),
        xytext=(peak_row["top_share"] - 0.22, peak_row["focus_revenue"] * 0.93),
        fontsize=8.5,
        color="#DC2626",
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": "#DC2626", "lw": 0.8},
    )
    ax_bot.scatter(
        [peak_row["top_share"]],
        [peak_row["net_vs_baseline_usd"]],
        color="#DC2626",
        s=65,
        zorder=7,
        edgecolors="white",
        linewidths=0.9,
    )
    ax_bot.annotate(
        f"+{usd_compact(peak_row['net_vs_baseline_usd'])} vs baseline",
        xy=(peak_row["top_share"], peak_row["net_vs_baseline_usd"]),
        xytext=(
            peak_row["top_share"] - 0.20,
            peak_row["net_vs_baseline_usd"] + 12_000_000,
        ),
        fontsize=8.5,
        color="#DC2626",
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": "#DC2626", "lw": 0.8},
    )

    for mark in [0.10, 0.25]:
        row = main_curve.loc[(main_curve["top_share"] - mark).abs().idxmin()]
        ax_top.scatter(
            [row["top_share"]], [row["focus_revenue"]], color="#2563EB", s=28, zorder=5
        )
        ax_top.text(
            row["top_share"] + 0.012,
            row["focus_revenue"] * 1.02,
            f"{usd_compact(row['focus_revenue'])}\n(λ={row['effective_lambda']:.2f}×)",
            fontsize=8.0,
            color="#0F172A",
            weight="bold",
        )

    ax_top.text(
        0.02,
        0.05,
        (
            f"Holdout set: {len(df_total):,} opportunities. "
            f"λ_max = {FEATURED_LAMBDA_MAX:.1f}× applied at the smallest slice; "
            "λ decreases linearly to 1.0× when working the full pipeline."
        ),
        transform=ax_top.transAxes,
        fontsize=8.5,
        color="#475569",
        va="bottom",
    )

    return fig


def plot_sweep(sweep_curves: pd.DataFrame) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)

    for lam in SWEEP_LAMBDA_MAXES:
        subset = sweep_curves[sweep_curves["lambda_max"] == lam].copy()
        color = PALETTE[lam]
        lw = 2.6 if lam == FEATURED_LAMBDA_MAX else 1.6
        ls = "-" if lam == FEATURED_LAMBDA_MAX else "--"
        label = (
            f"λ_max = {lam:.1f}×{' (featured)' if lam == FEATURED_LAMBDA_MAX else ''}"
        )
        ax.plot(
            subset["top_share"],
            subset["net_vs_baseline_usd"],
            color=color,
            linewidth=lw,
            linestyle=ls,
            label=label,
            zorder=4 if lam == FEATURED_LAMBDA_MAX else 3,
        )

    ax.axhline(0, color="#0F172A", linewidth=1.2, linestyle=":", alpha=0.7)
    ax.fill_between(
        sweep_curves[sweep_curves["lambda_max"] == FEATURED_LAMBDA_MAX]["top_share"],
        sweep_curves[sweep_curves["lambda_max"] == FEATURED_LAMBDA_MAX][
            "net_vs_baseline_usd"
        ],
        0,
        where=sweep_curves[sweep_curves["lambda_max"] == FEATURED_LAMBDA_MAX][
            "net_vs_baseline_usd"
        ]
        >= 0,
        alpha=0.10,
        color=PALETTE[FEATURED_LAMBDA_MAX],
        interpolate=True,
    )

    ax.set_xlabel("Top share of opportunities worked (ranked by expected value)")
    ax.set_ylabel("Net revenue delta vs. unmodified baseline (USD)")
    ax.set_title(
        "Focus-gain sensitivity: net revenue delta across λ_max values (annex)",
        fontsize=12,
        weight="bold",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd_compact))
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax.text(
        0.02,
        0.97,
        "λ_max = max focus-gain multiplier (applied at smallest slice; decays to 1.0× at full pipeline).",
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
    total_actual_won = float(df["actual_won_amount"].sum())

    main_curve, sweep_curves = build_all_curves(df)

    main_csv = DATA_DIR / "sim_focus_gain_main.csv"
    sweep_csv = DATA_DIR / "sim_focus_gain_sweep.csv"
    main_curve.to_csv(main_csv, index=False)
    sweep_curves.to_csv(sweep_csv, index=False)
    print(f"saved: {main_csv}")
    print(f"saved: {sweep_csv}")

    main_fig = plot_main(main_curve, df, total_actual_won)
    main_png = ASSET_DIR / "sim_focus_gain_main.png"
    main_fig.savefig(main_png, dpi=220, bbox_inches="tight")
    plt.close(main_fig)
    print(f"saved: {main_png}")

    sweep_fig = plot_sweep(sweep_curves)
    sweep_png = ASSET_DIR / "sim_focus_gain_sweep.png"
    sweep_fig.savefig(sweep_png, dpi=220, bbox_inches="tight")
    plt.close(sweep_fig)
    print(f"saved: {sweep_png}")


if __name__ == "__main__":
    main()

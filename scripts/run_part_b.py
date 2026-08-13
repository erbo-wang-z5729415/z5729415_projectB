"""Reproduce your Part B results. Run from the project root:

    python scripts/run_part_b.py
"""
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
from src import data_access  # noqa: E402
from src.features import daily_returns  # noqa: E402
from src.fusion import apply_sentiment, apply_sentiment_shock  # noqa: E402
from src.portfolios import oos_backtest, performance_metrics  # noqa: E402
from src.sentiment import (
    score_headlines,
    sector_sentiment_index,
    sentiment_shock_index,
)  # noqa: E402

def returns_from_weights(returns, weights):
    weights = weights.copy()
    weights["date"] = pd.to_datetime(weights["date"])

    weights = weights.set_index("date")
    weights = weights.drop(
        columns=["method"],
        errors="ignore",
    )

    asset_columns = [
        col
        for col in weights.columns
        if col in returns.columns
    ]

    weights = weights[asset_columns]
    weights = weights.reindex(returns.index).ffill()

    aligned_returns = returns[asset_columns]

    portfolio_returns = (
        aligned_returns * weights
    ).sum(
        axis=1,
        min_count=1,
    )

    return portfolio_returns.dropna()


def main():
    eq = data_access.load_equity_prices()
    cr = data_access.load_crypto_prices()
    news = data_access.load_news_headlines()

    print(
        "equities:", eq.shape,
        "crypto:", cr.shape,
        "news:", news.shape,
    )

    eq_returns = daily_returns(eq)
    cr_returns = daily_returns(cr)

    cr_returns = cr_returns[
        cr_returns["date"] <= pd.Timestamp("2023-12-31")
    ]

    eq_wide = eq_returns.pivot(
        index="date",
        columns="ticker",
        values="return",
    )

    cr_wide = cr_returns.pivot(
        index="date",
        columns="ticker",
        values="return",
    )

    eq_wide.columns = [
        f"equity_{ticker}"
        for ticker in eq_wide.columns
    ]

    cr_wide.columns = [
        f"crypto_{ticker}"
        for ticker in cr_wide.columns
    ]

    combined_returns = eq_wide.join(
        cr_wide,
        how="left",
    )
    combined_returns = combined_returns.dropna(how="any")
    min_variance = oos_backtest(
        combined_returns,
        method="min_variance",
    )

    max_sharpe = oos_backtest(
        combined_returns,
        method="max_sharpe",
    )

    project_root = pathlib.Path(__file__).resolve().parent.parent

    results_data = (
        project_root
        / "results"
        / "data"
    )

    results_tables = (
        project_root
        / "results"
        / "tables"
    )
    results_figures = project_root / "results" / "figures"

    results_figures.mkdir(
        parents=True,
        exist_ok=True,
    )
    results_data.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_tables.mkdir(
        parents=True,
        exist_ok=True,
    )

    performance_table = pd.DataFrame([
        {
            "method": "min_variance",
            **min_variance["metrics"],
        },
        {
            "method": "max_sharpe",
            **max_sharpe["metrics"],
        },
    ])

    performance_table.to_csv(
        results_tables / "performance_metrics.csv",
        index=False,
    )

    news = news.copy()

    news["date"] = (
        pd.to_datetime(
            news["date"],
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    news = news.drop_duplicates(
        subset=["date", "ticker", "title"]
    )

    equity_calendar = pd.DataFrame({
        "trading_date": (
            pd.to_datetime(eq["date"])
            .dt.normalize()
            .drop_duplicates()
            .sort_values()
        )
    })

    news["date"] = pd.to_datetime(
        news["date"]
    ).astype("datetime64[ns]")

    equity_calendar["trading_date"] = pd.to_datetime(
        equity_calendar["trading_date"]
    ).astype("datetime64[ns]")

    news = news.sort_values("date")

    equity_calendar = equity_calendar.sort_values(
        "trading_date"
    )

    mapped_news = pd.merge_asof(
        news,
        equity_calendar,
        left_on="date",
        right_on="trading_date",
        direction="forward",
    )

    mapped_news = mapped_news.dropna(
        subset=["trading_date"]
    )

    mapped_news = mapped_news.drop(
        columns=["date"]
    ).rename(
        columns={
            "trading_date": "date"
        }
    )

    headline_scores = score_headlines(
        mapped_news
    )

    sector_index = sector_sentiment_index(
        headline_scores
    )
    shock_index = sentiment_shock_index(sector_index)

    shock_index.to_csv(
        results_data / "sector_sentiment_shock.csv",
        index=False,
    )
    sector_index.to_csv(
        results_data / "sector_sentiment_index.csv",
        index=False,
    )

    sector_universe = data_access.load_sector_universe()

    min_tilted_weights = apply_sentiment(
        min_variance["weights"],
        sector_index,
        sector_universe,
    )

    max_tilted_weights = apply_sentiment(
        max_sharpe["weights"],
        sector_index,
        sector_universe,
    )
    min_shock_weights = apply_sentiment_shock(
        min_variance["weights"],
        shock_index,
        sector_universe,
    )

    max_shock_weights = apply_sentiment_shock(
        max_sharpe["weights"],
        shock_index,
        sector_universe,
    )
    min_tilted_returns = returns_from_weights(
        combined_returns,
        min_tilted_weights,
    )

    max_tilted_returns = returns_from_weights(
        combined_returns,
        max_tilted_weights,
    )
    min_shock_returns = returns_from_weights(
        combined_returns,
        min_shock_weights,
    )

    max_shock_returns = returns_from_weights(
        combined_returns,
        max_shock_weights,
    )
    min_tilted_metrics = performance_metrics(
        min_tilted_returns,
        periods_per_year=252,
    )

    max_tilted_metrics = performance_metrics(
        max_tilted_returns,
        periods_per_year=252,
    )
    min_shock_metrics = performance_metrics(
        min_shock_returns,
        periods_per_year=252,
    )

    max_shock_metrics = performance_metrics(
        max_shock_returns,
        periods_per_year=252,
    )
    fusion_metrics = pd.DataFrame([
        {
            "method": "min_variance",
            **min_variance["metrics"],
        },
        {
            "method": "min_variance_sentiment",
            **min_tilted_metrics,
        },
        {
            "method": "max_sharpe",
            **max_sharpe["metrics"],
        },
        {
            "method": "max_sharpe_sentiment",
            **max_tilted_metrics,
        },
    ])

    fusion_metrics.to_csv(
        results_tables / "fusion_metrics.csv",
        index=False,
    )
    shock_metrics = pd.DataFrame(
        [
            {
                "method": "min_variance",
                **min_variance["metrics"],
            },
            {
                "method": "min_variance_basic_sentiment",
                **min_tilted_metrics,
            },
            {
                "method": "min_variance_sentiment_shock",
                **min_shock_metrics,
            },
            {
                "method": "max_sharpe",
                **max_sharpe["metrics"],
            },
            {
                "method": "max_sharpe_basic_sentiment",
                **max_tilted_metrics,
            },
            {
                "method": "max_sharpe_sentiment_shock",
                **max_shock_metrics,
            },
        ]
    )

    shock_metrics.to_csv(
        results_tables / "sentiment_shock_metrics.csv",
        index=False,
    )
    return_index = min_variance[
        "daily_returns"
    ].index

    fund_returns = pd.DataFrame(
        index=return_index
    )

    fund_returns["min_variance"] = (
        min_variance["daily_returns"]
    )

    fund_returns["max_sharpe"] = (
        max_sharpe["daily_returns"]
    )

    fund_returns["min_variance_sentiment"] = (
        min_tilted_returns.reindex(
            return_index
        )
    )

    fund_returns["max_sharpe_sentiment"] = (
        max_tilted_returns.reindex(
            return_index
        )
    )
    fund_returns["min_variance_sentiment_shock"] = min_shock_returns.reindex(return_index)

    fund_returns["max_sharpe_sentiment_shock"] = max_shock_returns.reindex(return_index)
    fund_returns = fund_returns.reset_index()

    fund_returns = fund_returns.rename(
        columns={
            "index": "date"
        }
    )

    fund_returns.to_csv(
        results_data / "fund_returns.csv",
        index=False,
    )
    plot_returns = fund_returns.copy()
    plot_returns["date"] = pd.to_datetime(plot_returns["date"])

    growth = plot_returns.set_index("date").dropna()
    growth = (1 + growth).cumprod()

    plt.figure(figsize=(10, 6))

    for column in growth.columns:
        plt.plot(
            growth.index,
            growth[column],
            label=column,
        )

    plt.title("Out-of-Sample Growth of $1")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_figures / "growth_of_1.png",
        dpi=300,
    )

    plt.close()

    base_growth = (1 + min_variance["daily_returns"]).cumprod()

    drawdown = base_growth / base_growth.cummax() - 1

    plt.figure(figsize=(10, 5))

    plt.plot(
        drawdown.index,
        drawdown,
    )

    plt.title("Minimum-Variance Fund Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()

    plt.savefig(
        results_figures / "drawdown_min_variance.png",
        dpi=300,
    )

    plt.close()

    weights_plot = min_variance["weights"].copy()
    weights_plot["date"] = pd.to_datetime(weights_plot["date"])

    weight_assets = [col for col in weights_plot.columns if col not in ["date", "method"]]

    average_weights = weights_plot[weight_assets].mean().sort_values(ascending=False)

    top_assets = average_weights.head(10).index

    plt.figure(figsize=(11, 6))

    for column in top_assets:
        plt.plot(
            weights_plot["date"],
            weights_plot[column],
            label=column,
        )

    plt.title("Minimum-Variance Portfolio Weights Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Weight")
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()

    plt.savefig(
        results_figures / "weights_over_time.png",
        dpi=300,
    )

    plt.close()

    sharpe_plot = fusion_metrics[["method", "sharpe"]].copy()

    plt.figure(figsize=(9, 5))

    plt.bar(
        sharpe_plot["method"],
        sharpe_plot["sharpe"],
    )

    plt.title("Out-of-Sample Sharpe Ratios")
    plt.xlabel("Fund")
    plt.ylabel("Sharpe Ratio")
    plt.xticks(
        rotation=20,
        ha="right",
    )
    plt.tight_layout()

    plt.savefig(
        results_figures / "sharpe_comparison.png",
        dpi=300,
    )

    plt.close()

    sentiment_plot = sector_index.copy()

    sentiment_wide = sentiment_plot.pivot(
        index="date",
        columns="sector",
        values="sentiment",
    )

    plt.figure(figsize=(11, 7))

    for sector in sentiment_wide.columns:
        plt.plot(
            sentiment_wide.index,
            sentiment_wide[sector],
            label=sector,
            alpha=0.8,
        )

    plt.title("Sector News Sentiment Index")
    plt.xlabel("Date")
    plt.ylabel("VADER Compound Sentiment")
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()

    plt.savefig(
        results_figures / "sector_sentiment_index.png",
        dpi=300,
    )

    plt.close()

    fusion_growth = growth[
        [
            "max_sharpe",
            "max_sharpe_sentiment",
        ]
    ]

    plt.figure(figsize=(10, 6))

    for column in fusion_growth.columns:
        plt.plot(
            fusion_growth.index,
            fusion_growth[column],
            label=column,
        )

    plt.title("Maximum-Sharpe Fund: Base vs Sentiment Tilt")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_figures / "fusion_comparison.png",
        dpi=300,
    )
    shock_growth = (
        1
        + fund_returns.set_index("date")[
            [
                "max_sharpe",
                "max_sharpe_sentiment",
                "max_sharpe_sentiment_shock",
            ]
        ]
    ).cumprod()

    plt.figure(figsize=(10, 6))

    for column in shock_growth.columns:
        plt.plot(
            shock_growth.index,
            shock_growth[column],
            label=column,
        )

    plt.title("Maximum-Sharpe Fund: Sentiment Extension Comparison")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_figures / "sentiment_shock_comparison.png",
        dpi=300,
    )

    plt.close()
    plt.close()
    min_weights = min_variance[
        "weights"
    ].copy()

    max_weights = max_sharpe[
        "weights"
    ].copy()

    min_tilted_weights[
        "method"
    ] = "min_variance_sentiment"

    max_tilted_weights[
        "method"
    ] = "max_sharpe_sentiment"
    min_shock_weights["method"] = "min_variance_sentiment_shock"

    max_shock_weights["method"] = "max_sharpe_sentiment_shock"
    fund_weights = pd.concat(
        [
            min_weights,
            max_weights,
            min_tilted_weights,
            max_tilted_weights,
            min_shock_weights,
            max_shock_weights,
        ],
        ignore_index=True,
    )

    fund_weights.to_csv(
        results_data / "fund_weights.csv",
        index=False,
    )
    asset_columns = [col for col in fund_weights.columns if col not in ["date", "method"]]

    current_holdings = fund_weights.sort_values("date").groupby("method").tail(1)

    current_holdings = current_holdings.melt(
        id_vars=["date", "method"],
        value_vars=asset_columns,
        var_name="asset",
        value_name="weight",
    )

    current_holdings = current_holdings[current_holdings["weight"].fillna(0) > 0]

    current_holdings = current_holdings.sort_values(
        ["method", "weight"],
        ascending=[True, False],
    )

    current_holdings.to_csv(
        results_tables / "current_holdings.csv",
        index=False,
    )
    print("\nPerformance metrics:")
    print(
        performance_table.to_string(
            index=False
        )
    )

    print("\nFusion metrics:")
    print(
        fusion_metrics.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(
        results_data
        / "fund_returns.csv"
    )
    print(
        results_data
        / "fund_weights.csv"
    )
    print(
        results_data
        / "sector_sentiment_index.csv"
    )
    print(
        results_tables
        / "performance_metrics.csv"
    )
    print(
        results_tables
        / "fusion_metrics.csv"
    )
    print("\nFigures saved:")
    print(results_figures / "growth_of_1.png")
    print(results_figures / "drawdown_min_variance.png")
    print(results_figures / "weights_over_time.png")
    print(results_figures / "sharpe_comparison.png")
    print(results_figures / "sector_sentiment_index.png")
    print(results_figures / "fusion_comparison.png")
    print(results_tables / "current_holdings.csv")
    print(results_data / "sector_sentiment_shock.csv")

    print(results_tables / "sentiment_shock_metrics.csv")

    print(results_figures / "sentiment_shock_comparison.png")
    # TODO: returns -> out-of-sample funds + fact sheets (Station 3)
    # TODO: sentiment index + fusion extension (Station 3)
    # TODO: save figures/tables under results/ for the app and the report


if __name__ == "__main__":
    main()

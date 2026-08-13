""""FinTech Project - your investment app (starter).

This thin starter proves the app deploys and loads the hosted data. Build your real
dashboard on top of it: a fund picker, each fund's fact sheet (growth of $1,
drawdown, Sharpe, holdings), an allocation control, and your sentiment analytics.

Run locally:   streamlit run streamlit_app.py
Deploy:        push this folder to a public GitHub repo, then connect it on
               share.streamlit.io with entrypoint streamlit_app.py (see brief App. D).
"""
import pathlib

import pandas as pd
import streamlit as st

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

RESULTS_DATA = PROJECT_ROOT / "results" / "data"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"

st.set_page_config(
    page_title="SignalHarbour",
    layout="wide",
)

st.title("SignalHarbour")
st.caption("Systematic multi-asset funds with news-sentiment analytics")


@st.cache_data
def load_results():
    fund_returns = pd.read_csv(
        RESULTS_DATA / "fund_returns.csv",
        parse_dates=["date"],
    )

    fund_weights = pd.read_csv(
        RESULTS_DATA / "fund_weights.csv",
        parse_dates=["date"],
    )

    sector_sentiment = pd.read_csv(
        RESULTS_DATA / "sector_sentiment_index.csv",
        parse_dates=["date"],
    )

    performance = pd.read_csv(
        RESULTS_TABLES / "performance_metrics.csv"
    )

    fusion = pd.read_csv(
        RESULTS_TABLES / "fusion_metrics.csv"
    )

    holdings = pd.read_csv(
        RESULTS_TABLES / "current_holdings.csv",
        parse_dates=["date"],
    )

    return (
        fund_returns,
        fund_weights,
        sector_sentiment,
        performance,
        fusion,
        holdings,
    )


(
    fund_returns,
    fund_weights,
    sector_sentiment,
    performance,
    fusion,
    holdings,
) = load_results()


fund_names = {
    "min_variance": "Combined Minimum Variance",
    "max_sharpe": "Combined Maximum Sharpe",
    "min_variance_sentiment": "Minimum Variance + Sentiment",
    "max_sharpe_sentiment": "Maximum Sharpe + Sentiment",
}


tab_funds, tab_allocation, tab_sentiment = st.tabs(
    ["Funds", "Allocation", "Sentiment"]
)


with tab_funds:
    st.subheader("Fund Comparison")

    display_metrics = fusion.copy()

    display_metrics["Fund"] = (
        display_metrics["method"]
        .map(fund_names)
        .fillna(display_metrics["method"])
    )

    display_metrics["Annualised Return"] = (
        display_metrics["annualised_return"] * 100
    )

    display_metrics["Annualised Volatility"] = (
        display_metrics["annualised_volatility"] * 100
    )

    display_metrics["Maximum Drawdown"] = (
        display_metrics["max_drawdown"] * 100
    )

    st.dataframe(
        display_metrics[
            [
                "Fund",
                "Annualised Return",
                "Annualised Volatility",
                "sharpe",
                "Maximum Drawdown",
            ]
        ].style.format(
            {
                "Annualised Return": "{:.2f}%",
                "Annualised Volatility": "{:.2f}%",
                "sharpe": "{:.2f}",
                "Maximum Drawdown": "{:.2f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Growth of $1")

    growth_data = fund_returns.copy()
    growth_data = growth_data.set_index("date")

    growth_columns = [
        col
        for col in growth_data.columns
        if col in fund_names
    ]

    growth = (
        1 + growth_data[growth_columns]
    ).cumprod()

    growth = growth.rename(
        columns=fund_names
    )

    st.line_chart(growth)

    st.subheader("Fund Fact Sheet")

    selected_method = st.selectbox(
        "Select a fund",
        options=list(fund_names.keys()),
        format_func=lambda x: fund_names[x],
    )

    selected_row = fusion[
        fusion["method"] == selected_method
    ]

    if not selected_row.empty:
        selected_row = selected_row.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Annualised Return",
            f"{selected_row['annualised_return'] * 100:.2f}%",
        )

        c2.metric(
            "Annualised Volatility",
            f"{selected_row['annualised_volatility'] * 100:.2f}%",
        )

        c3.metric(
            "Sharpe Ratio",
            f"{selected_row['sharpe']:.2f}",
        )

        c4.metric(
            "Maximum Drawdown",
            f"{selected_row['max_drawdown'] * 100:.2f}%",
        )

    selected_returns = (
        fund_returns[
            ["date", selected_method]
        ]
        .dropna()
        .set_index("date")
    )

    if not selected_returns.empty:
        selected_growth = (
            1 + selected_returns[selected_method]
        ).cumprod()

        selected_drawdown = (
            selected_growth
            / selected_growth.cummax()
            - 1
        )

        left, right = st.columns(2)

        with left:
            st.markdown("#### Growth of $1")
            st.line_chart(
                selected_growth.rename(
                    fund_names[selected_method]
                )
            )

        with right:
            st.markdown("#### Drawdown")
            st.line_chart(
                selected_drawdown.rename(
                    "Drawdown"
                )
            )

    st.markdown("#### Current Holdings")

    selected_holdings = holdings[
        holdings["method"] == selected_method
    ].copy()

    selected_holdings = selected_holdings[
        selected_holdings["weight"] > 0.0001
    ]

    selected_holdings["weight"] = (
        selected_holdings["weight"] * 100
    )

    selected_holdings["asset"] = (
        selected_holdings["asset"]
        .str.replace("equity_", "", regex=False)
        .str.replace("crypto_", "", regex=False)
    )

    st.dataframe(
        selected_holdings[
            ["asset", "weight"]
        ].rename(
            columns={
                "asset": "Asset",
                "weight": "Weight",
            }
        ).style.format(
            {"Weight": "{:.2f}%"}
        ),
        width="stretch",
        hide_index=True,
    )


with tab_allocation:
    st.subheader("Fund Allocation")

    st.write(
        "Set an allocation across the two base systematic funds."
    )

    min_allocation = st.slider(
        "Minimum Variance allocation (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
    )

    max_allocation = 100 - min_allocation

    st.write(
        f"Maximum Sharpe allocation: {max_allocation}%"
    )

    allocation_returns = fund_returns[
        [
            "date",
            "min_variance",
            "max_sharpe",
        ]
    ].dropna()

    allocation_returns["portfolio_return"] = (
        allocation_returns["min_variance"]
        * (min_allocation / 100)
        +
        allocation_returns["max_sharpe"]
        * (max_allocation / 100)
    )

    allocation_returns["growth_of_1"] = (
        1 + allocation_returns["portfolio_return"]
    ).cumprod()

    st.line_chart(
        allocation_returns.set_index("date")[
            "growth_of_1"
        ]
    )

    final_value = (
        allocation_returns["growth_of_1"].iloc[-1]
    )

    st.metric(
        "Final value of $1",
        f"${final_value:.2f}",
    )


with tab_sentiment:
    st.subheader("Sector News Sentiment")

    sectors = sorted(
        sector_sentiment["sector"]
        .dropna()
        .unique()
    )

    selected_sectors = st.multiselect(
        "Select sectors",
        options=sectors,
        default=sectors[:3],
    )

    sentiment_display = sector_sentiment[
        sector_sentiment["sector"].isin(
            selected_sectors
        )
    ].copy()

    sentiment_chart = sentiment_display.pivot(
        index="date",
        columns="sector",
        values="sentiment",
    )

    st.line_chart(sentiment_chart)

    st.caption(
        "The sentiment series is based on VADER headline scores. "
        "The trading signal uses the lagged version to avoid look-ahead bias."
    )

    st.subheader("Latest Sector Sentiment")

    latest_sentiment = (
        sector_sentiment
        .sort_values("date")
        .groupby("sector")
        .tail(1)
        .sort_values(
            "sentiment",
            ascending=False,
        )
    )

    st.dataframe(
        latest_sentiment[
            [
                "sector",
                "sentiment",
                "sentiment_lag1",
            ]
        ].rename(
            columns={
                "sector": "Sector",
                "sentiment": "Current Sentiment",
                "sentiment_lag1": "Lagged Sentiment",
            }
        ).style.format(
            {
                "Current Sentiment": "{:.3f}",
                "Lagged Sentiment": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Sentiment Fusion Result")

    fusion_display = fusion.copy()

    fusion_display["Fund"] = (
        fusion_display["method"]
        .map(fund_names)
        .fillna(fusion_display["method"])
    )

    st.dataframe(
        fusion_display[
            [
                "Fund",
                "annualised_return",
                "annualised_volatility",
                "sharpe",
                "max_drawdown",
            ]
        ].style.format(
            {
                "annualised_return": "{:.2%}",
                "annualised_volatility": "{:.2%}",
                "sharpe": "{:.2f}",
                "max_drawdown": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

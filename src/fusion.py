"""Station 3 (extension) - fuse sentiment into the funds.

Tilt or factor: combine your sentiment signal with the portfolio weights,
look-ahead safe, then test whether it adds value. An honest negative result,
explained, is good work.
"""
import pandas as pd


def apply_sentiment(
    weights: pd.DataFrame,
    sentiment: pd.DataFrame,
    sector_universe: pd.DataFrame,
):
    """TODO: your fusion rule (for example tilt weights toward high-sentiment names)."""
    weights = weights.copy()
    sentiment = sentiment.copy()
    sector_universe = sector_universe.copy()

    weights["date"] = pd.to_datetime(weights["date"])
    sentiment["date"] = pd.to_datetime(sentiment["date"])

    sector_map = dict(
        zip(
            sector_universe["ticker"],
            sector_universe["sector"],
        )
    )

    sentiment_wide = sentiment.pivot(
        index="date",
        columns="sector",
        values="sentiment_lag1",
    )

    asset_columns = [
        col
        for col in weights.columns
        if col not in ["date", "method"]
    ]

    equity_columns = [
        col
        for col in asset_columns
        if col.startswith("equity_")
    ]

    tilted = weights.copy()

    for i in tilted.index:
        date = tilted.at[i, "date"]

        if date not in sentiment_wide.index:
            continue

        day_sentiment = sentiment_wide.loc[date]

        for col in equity_columns:
            ticker = col.replace("equity_", "")
            sector = sector_map.get(ticker)

            if sector in day_sentiment.index:
                score = day_sentiment[sector]

                if pd.notna(score):
                    tilted.at[i, col] = (
                        tilted.at[i, col]
                        * (1 + 0.20 * score)
                    )

        total = tilted.loc[i, asset_columns].sum()

        if total > 0:
            tilted.loc[i, asset_columns] = (
                tilted.loc[i, asset_columns]
                / total
            )

    return tilted

"""Station 2 - your features: return features and text assembly.

Build your return features here, and assemble the headlines into a daily text
panel. Scoring the text is the Station 3 sentiment model (see src/sentiment.py).
"""
import pandas as pd


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Simple daily returns per ticker. Use adjClose.

    TODO: pivot to wide (date x ticker) or keep long, your choice; pct_change
    within each ticker.
    """
    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ticker", "date"])

    data["return"] = (
        data.groupby("ticker")[price_col]
        .pct_change(fill_method=None)
    )

    return data[["date", "ticker", "return"]]


def assemble_headline_panel(headlines: pd.DataFrame) -> pd.DataFrame:
    """Assemble the headlines into a daily panel per ticker and sector.

    Station 2 is assembly only: structure the text and date-align it to the
    trading calendar. Scoring the text - and lagging the signal - is the
    Station 3 model.
    """
    data = headlines.copy()

    data["date"] = (
        pd.to_datetime(data["date"], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    data = data.drop_duplicates(
        subset=["date", "ticker", "title"]
    )

    panel = (
        data.groupby(
            ["date", "ticker", "sector"],
            as_index=False,
        )
        .agg(
            headline_count=("title", "size"),
            headlines=("title", lambda x: " | ".join(x)),
        )
    )

    return panel

"""Station 3 - your sentiment model and index from news headlines.

This is the model step: score each headline, aggregate to a daily per-ticker score,
then to an equal-weight sector index. Headlines are a noisy proxy, so lag to avoid
look-ahead.
"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply a sentiment model (VADER or another) to the assembled headlines.

    TODO: return a per-headline or per-ticker-day sentiment score. VADER uses
    casing, punctuation, and negation, so do not strip them. VADER also needs a
    one-time nltk.download('vader_lexicon') before it scores (a build step, not the
    deployed app).
    """
    data = panel.copy()
    analyser = SentimentIntensityAnalyzer()

    data["sentiment"] = data["title"].apply(
        lambda text: analyser.polarity_scores(str(text))["compound"]
    )

    return data


def sector_sentiment_index(scores: pd.DataFrame) -> pd.DataFrame:
    """TODO: build a daily sentiment index per sector (equal-weight across names)."""
    ticker_daily = (
        scores.groupby(
            ["date", "ticker", "sector"],
            as_index=False,
        )
        .agg(sentiment=("sentiment", "mean"))
    )

    sector_daily = (
        ticker_daily.groupby(
            ["date", "sector"],
            as_index=False,
        )
        .agg(sentiment=("sentiment", "mean"))
    )

    sector_daily = sector_daily.sort_values(
        ["sector", "date"]
    )

    sector_daily["sentiment_lag1"] = (
        sector_daily.groupby("sector")["sentiment"].shift(1)
    )

    return sector_daily
def sentiment_shock_index(
    sector_index: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    data = sector_index.copy()
    data = data.sort_values(["sector", "date"])

    data["sentiment_mean_20"] = (
        data.groupby("sector")["sentiment"]
        .transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
    )

    data["sentiment_std_20"] = (
        data.groupby("sector")["sentiment"]
        .transform(
            lambda x: x.shift(1).rolling(window).std()
        )
    )

    data["sentiment_shock"] = (
        (
            data["sentiment_lag1"]
            - data["sentiment_mean_20"]
        )
        / data["sentiment_std_20"]
    )

    data["sentiment_shock"] = (
        data["sentiment_shock"]
        .clip(-2, 2)
    )

    return data

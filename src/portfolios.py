"""Station 3 - your funds: optimal portfolios + out-of-sample backtest.

Build at least a combined equity-plus-crypto fund with two optimisation methods.
Backtest rules: walk-forward, no look-ahead, weights from past data only, annualise
with 252 (equity) or 365 (crypto). See the brief, Part B.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _min_variance_weights(returns: pd.DataFrame) -> np.ndarray:
    covariance = returns.cov().values * 252
    n_assets = len(returns.columns)

    initial_weights = np.repeat(1 / n_assets, n_assets)

    result = minimize(
        lambda w: w @ covariance @ w,
        initial_weights,
        method="SLSQP",
        bounds=[(0, 1)] * n_assets,
        constraints={
            "type": "eq",
            "fun": lambda w: w.sum() - 1,
        },
    )

    if result.success:
        return result.x

    return initial_weights


def _max_sharpe_weights(returns: pd.DataFrame) -> np.ndarray:
    mean_returns = returns.mean().values * 252
    covariance = returns.cov().values * 252
    n_assets = len(returns.columns)

    initial_weights = np.repeat(1 / n_assets, n_assets)

    def objective(weights):
        portfolio_return = weights @ mean_returns
        portfolio_volatility = np.sqrt(
            weights @ covariance @ weights
        )

        if portfolio_volatility == 0:
            return 0

        return -(portfolio_return / portfolio_volatility)

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=[(0, 1)] * n_assets,
        constraints={
            "type": "eq",
            "fun": lambda w: w.sum() - 1,
        },
    )

    if result.success:
        return result.x

    return initial_weights


def oos_backtest(returns: pd.DataFrame, method: str = "min_variance"):
    """Your walk-forward out-of-sample backtest.

    TODO: return at least the daily portfolio returns, the weights over time,
    growth of $1, and metrics (annualised return, volatility, Sharpe, max drawdown).
    """
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()

    lookback = 252
    rebalance_every = 21

    portfolio_returns = pd.Series(
        index=returns.index,
        dtype=float,
    )

    weight_records = []
    current_weights = None

    for i in range(lookback, len(returns)):
        if current_weights is None or (i - lookback) % rebalance_every == 0:
            estimation_window = returns.iloc[i - lookback:i]

            valid_assets = estimation_window.columns[
                estimation_window.notna().all()
            ]

            estimation_window = estimation_window[valid_assets]

            if len(valid_assets) == 0:
                continue

            if method == "min_variance":
                weights = _min_variance_weights(estimation_window)

            elif method == "max_sharpe":
                weights = _max_sharpe_weights(estimation_window)

            else:
                raise ValueError(f"Unknown method: {method}")

            current_weights = pd.Series(
                weights,
                index=valid_assets,
            )

            record = {
                "date": returns.index[i],
                "method": method,
            }

            record.update(current_weights.to_dict())
            weight_records.append(record)

        day_returns = returns.iloc[i][current_weights.index].dropna()

        if day_returns.empty:
            continue

        day_weights = current_weights[day_returns.index]
        day_weights = day_weights / day_weights.sum()

        portfolio_returns.iloc[i] = (
            day_returns @ day_weights
        )

    portfolio_returns = portfolio_returns.dropna()
    portfolio_returns.name = method

    growth_of_1 = (1 + portfolio_returns).cumprod()

    weights_over_time = pd.DataFrame(weight_records)

    metrics = performance_metrics(
        portfolio_returns,
        periods_per_year=252,
    )

    return {
        "daily_returns": portfolio_returns,
        "weights": weights_over_time,
        "growth_of_1": growth_of_1,
        "metrics": metrics,
    }


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """TODO: annualised return, annualised volatility, Sharpe, and max drawdown."""
    daily_returns = daily_returns.dropna()

    growth = (1 + daily_returns).cumprod()

    years = len(daily_returns) / periods_per_year

    annualised_return = (
        growth.iloc[-1] ** (1 / years) - 1
    )

    annualised_volatility = (
        daily_returns.std() * np.sqrt(periods_per_year)
    )

    sharpe = (
        annualised_return / annualised_volatility
        if annualised_volatility != 0
        else np.nan
    )

    drawdown = growth / growth.cummax() - 1

    return {
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
    }

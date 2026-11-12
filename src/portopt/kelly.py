"""Kelly criterion sizing for a portfolio of assets (continuous-time approximation)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_weights(mu: pd.Series, cov: pd.DataFrame, rf=0.0, fraction=1.0, max_leverage=None,
                  long_only=False) -> pd.Series:
    """Growth-optimal weights  f* = fraction * Sigma^{-1} (mu - rf).

    Weights need not sum to one: the remainder (1 - sum) is held in cash (or borrowed).
    ``max_leverage`` caps gross exposure sum(|w|) by scaling the whole vector.
    """
    f = fraction * np.linalg.solve(cov.values, (mu.loc[cov.index] - rf).values)
    f = pd.Series(f, index=cov.index)
    if long_only:
        f = f.clip(lower=0)
    if max_leverage is not None and f.abs().sum() > max_leverage:
        f *= max_leverage / f.abs().sum()
    return f


def single_asset_kelly(mu, sigma, rf=0.0):
    return (mu - rf) / sigma**2


def expected_growth(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf=0.0):
    """g(w) = rf + w'(mu - rf) - 0.5 w' Sigma w (annualised log-growth)."""
    w = w.loc[cov.index]
    return float(rf + w @ (mu.loc[cov.index] - rf) - 0.5 * w @ cov @ w)

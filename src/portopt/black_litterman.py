"""Black-Litterman (1992) model, in the He & Litterman (1999) formulation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_returns(cov: pd.DataFrame, market_weights: pd.Series, risk_aversion=2.5, rf=0.0) -> pd.Series:
    """Reverse-optimised equilibrium excess returns  pi = delta * Sigma * w_mkt."""
    return rf + risk_aversion * cov @ market_weights.loc[cov.index]


def risk_aversion_from_market(market_returns: pd.Series, rf=0.0, periods=252):
    excess = market_returns.mean() * periods - rf
    return excess / (market_returns.var() * periods)


def posterior(cov: pd.DataFrame, pi: pd.Series, P: pd.DataFrame, Q: pd.Series,
              omega: pd.DataFrame | None = None, tau=0.05, view_confidence=None):
    """Blend equilibrium returns with investor views.

    P : (k x n) pick matrix, rows are views, columns are assets
    Q : (k,) expected view returns
    omega : view uncertainty; default is the proportional prior tau * P Sigma P'
    view_confidence : optional (k,) values in (0, 1]; scales omega (Idzorek-style shortcut)

    Returns (posterior_mean, posterior_cov).
    """
    S, p = cov.values, pi.loc[cov.index].values
    Pm, q = P[cov.index].values, Q.values
    if omega is None:
        om = np.diag(np.diag(tau * Pm @ S @ Pm.T))
        if view_confidence is not None:
            c = np.asarray(view_confidence, float)
            om = om * np.diag((1 - c) / c)
    else:
        om = np.asarray(omega)
    tS_inv = np.linalg.inv(tau * S)
    om_inv = np.linalg.inv(om)
    M = np.linalg.inv(tS_inv + Pm.T @ om_inv @ Pm)
    mu_bl = M @ (tS_inv @ p + Pm.T @ om_inv @ q)
    return pd.Series(mu_bl, cov.index), pd.DataFrame(S + M, cov.index, cov.columns)


def make_views(assets, views):
    """Build (P, Q) from a list like [("AAPL", 0.10), (("JPM", "BAC"), 0.02)].

    A single ticker is an absolute view; a pair (long, short) is a relative view.
    """
    P = pd.DataFrame(0.0, index=range(len(views)), columns=assets)
    Q = pd.Series(0.0, index=range(len(views)))
    for i, (who, val) in enumerate(views):
        if isinstance(who, tuple):
            P.loc[i, who[0]], P.loc[i, who[1]] = 1.0, -1.0
        else:
            P.loc[i, who] = 1.0
        Q[i] = val
    return P, Q

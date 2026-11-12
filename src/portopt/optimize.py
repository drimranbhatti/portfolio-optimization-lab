"""Portfolio optimisers. All return a pandas Series of weights summing to 1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


def _as_series(w, names):
    w = np.where(np.abs(w) < 1e-10, 0.0, w)
    return pd.Series(w / w.sum(), index=names)


def _solve(objective, n, long_only=True, max_weight=1.0, extra_cons=()):
    bounds = [(0.0 if long_only else -max_weight, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra_cons]
    res = minimize(objective, np.full(n, 1 / n), method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def equal_weight(cov: pd.DataFrame) -> pd.Series:
    return pd.Series(1 / len(cov), index=cov.index)


def min_variance(cov: pd.DataFrame, long_only=True, max_weight=1.0) -> pd.Series:
    S = cov.values
    return _as_series(_solve(lambda w: w @ S @ w, len(S), long_only, max_weight), cov.index)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf=0.0, long_only=True, max_weight=1.0) -> pd.Series:
    S, m = cov.values, mu.loc[cov.index].values - rf
    return _as_series(_solve(lambda w: -(w @ m) / np.sqrt(w @ S @ w), len(S), long_only, max_weight), cov.index)


def target_return(mu, cov, target, long_only=True, max_weight=1.0):
    S, m = cov.values, mu.loc[cov.index].values
    con = ({"type": "eq", "fun": lambda w: w @ m - target},)
    return _as_series(_solve(lambda w: w @ S @ w, len(S), long_only, max_weight, con), cov.index)


def efficient_frontier(mu, cov, n_points=40, long_only=True):
    """Return DataFrame of (vol, ret) points along the long-only frontier."""
    w_min = min_variance(cov, long_only)
    lo = float(w_min @ mu.loc[cov.index])
    hi = float(mu.max()) * 0.999
    rows = []
    for t in np.linspace(lo, hi, n_points):
        try:
            w = target_return(mu, cov, t, long_only)
            rows.append((np.sqrt(w @ cov @ w), float(w @ mu.loc[cov.index])))
        except RuntimeError:
            continue
    return pd.DataFrame(rows, columns=["vol", "ret"])


def risk_parity(cov: pd.DataFrame) -> pd.Series:
    """Equal-risk-contribution portfolio (Maillard, Roncalli & Teiletche, 2010)."""
    S = cov.values
    n = len(S)

    def obj(w):
        rc = w * (S @ w)
        return np.sum((rc[:, None] - rc[None, :]) ** 2)

    return _as_series(_solve(obj, n), cov.index)


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    rc = w * (cov @ w)
    return rc / rc.sum()


# ----- Hierarchical Risk Parity (Lopez de Prado, 2016) -----

def _ivp(cov):
    ivp = 1 / np.diag(cov)
    return ivp / ivp.sum()


def _cluster_var(cov, items):
    sub = cov.iloc[items, items].values
    w = _ivp(sub)
    return w @ sub @ w


def hrp(cov: pd.DataFrame, corr: pd.DataFrame | None = None, method="single") -> pd.Series:
    """Hierarchical Risk Parity weights.

    ``corr`` may be a denoised/detoned correlation matrix used only for clustering.
    """
    if corr is None:
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
    dist = np.sqrt(np.clip((1 - np.asarray(corr)) / 2.0, 0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method=method)
    order = list(leaves_list(link))

    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[j:k] for c in clusters for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_l, v_r = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1 - alpha
    w.index = cov.index[w.index]
    return w.loc[cov.index] / w.sum()


def quasi_diag_order(corr: pd.DataFrame, method="single"):
    """Asset order produced by hierarchical clustering (for plotting seriated matrices)."""
    dist = np.sqrt(np.clip((1 - corr.values) / 2.0, 0, None))
    np.fill_diagonal(dist, 0.0)
    return list(corr.index[leaves_list(linkage(squareform(dist, checks=False), method=method))])


def portfolio_stats(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf=0.0):
    ret, vol = float(w @ mu.loc[w.index]), float(np.sqrt(w @ cov.loc[w.index, w.index] @ w))
    return {"return": ret, "volatility": vol, "sharpe": (ret - rf) / vol}

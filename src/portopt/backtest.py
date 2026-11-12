"""Walk-forward backtesting with periodic re-estimation and transaction costs."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def walk_forward(returns: pd.DataFrame, strategy: Callable[[pd.DataFrame], pd.Series],
                 lookback=504, rebalance="ME", cost_bps=5.0, cash_rate=0.0) -> pd.DataFrame:
    """Run ``strategy(history) -> weights`` on a rolling window and hold until next rebalance.

    Weights drift with prices between rebalances. Any weight not invested (1 - sum w)
    earns ``cash_rate``; negative cash means borrowing at the same rate.

    Returns a DataFrame with columns ``ret`` (net daily return) and ``turnover``.
    """
    dates = returns.index
    rebal_dates = returns.resample(rebalance).last().index
    rebal_idx = sorted({dates.searchsorted(d, side="right") - 1 for d in rebal_dates})
    rebal_idx = [i for i in rebal_idx if i >= lookback]
    daily_cash = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1

    out_ret, out_turn, out_dates = [], [], []
    w = pd.Series(0.0, index=returns.columns)
    next_rebal = iter(rebal_idx + [len(dates)])
    nxt = next(next_rebal)
    for i in range(rebal_idx[0], len(dates) - 1):
        turnover = 0.0
        if i == nxt:
            target = strategy(returns.iloc[i - lookback + 1:i + 1]).reindex(returns.columns).fillna(0.0)
            turnover = float((target - w).abs().sum())
            w = target
            nxt = next(next_rebal)
        r = returns.iloc[i + 1]
        cash = 1 - w.sum()
        gross = float(w @ r + cash * daily_cash)
        net = gross - turnover * cost_bps / 1e4
        out_ret.append(net); out_turn.append(turnover); out_dates.append(dates[i + 1])
        # drift weights with realised returns
        grown = w * (1 + r)
        w = grown / (1 + gross) if (1 + gross) != 0 else grown
    return pd.DataFrame({"ret": out_ret, "turnover": out_turn}, index=pd.DatetimeIndex(out_dates))


def performance(returns: pd.Series, rf=0.0) -> dict:
    r = returns.dropna()
    wealth = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = wealth.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    dd = wealth / wealth.cummax() - 1
    excess = r.mean() * TRADING_DAYS - rf
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": excess / vol if vol > 0 else np.nan,
        "Sortino": excess / downside if downside > 0 else np.nan,
        "Max drawdown": dd.min(),
        "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
    }


def summary(results: dict[str, pd.DataFrame], rf=0.0) -> pd.DataFrame:
    rows = {}
    for name, df in results.items():
        stats = performance(df["ret"], rf)
        stats["Avg turnover / rebalance"] = df.loc[df.turnover > 0, "turnover"].mean()
        rows[name] = stats
    return pd.DataFrame(rows).T

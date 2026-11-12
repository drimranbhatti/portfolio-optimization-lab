import numpy as np
import pandas as pd
import pytest

from portopt import backtest, black_litterman as bl, covariance, kelly, optimize

rng = np.random.default_rng(0)
N, T = 8, 1500
names = [f"A{i}" for i in range(N)]
factor = rng.standard_normal((T, 1)) * 0.01
rets = pd.DataFrame(factor @ rng.uniform(0.5, 1.5, (1, N)) + rng.standard_normal((T, N)) * 0.01,
                    columns=names, index=pd.bdate_range("2018-01-01", periods=T))
cov = covariance.sample(rets)
mu = pd.Series(np.linspace(0.04, 0.12, N), index=names)


def test_weights_sum_to_one_and_long_only():
    for w in (optimize.min_variance(cov), optimize.max_sharpe(mu, cov), optimize.risk_parity(cov), optimize.hrp(cov)):
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-9).all()


def test_min_variance_matches_closed_form_when_unconstrained():
    w = optimize.min_variance(cov, long_only=False, max_weight=10)
    inv = np.linalg.solve(cov.values, np.ones(N)); closed = inv / inv.sum()
    assert np.allclose(w.values, closed, atol=1e-4)


def test_risk_parity_equalises_contributions():
    rc = optimize.risk_contributions(optimize.risk_parity(cov), cov)
    assert rc.max() - rc.min() < 1e-3


def test_hrp_diagonal_cov_is_inverse_variance():
    d = pd.DataFrame(np.diag([0.01, 0.04, 0.09, 0.16]), index=list("abcd"), columns=list("abcd"))
    ivp = 1 / np.diag(d); ivp /= ivp.sum()
    assert np.allclose(optimize.hrp(d).values, ivp)


def test_denoising_keeps_diagonal_and_is_psd():
    dn, n_facts, _ = covariance.denoise(cov, n_obs=T)
    assert np.allclose(np.diag(dn), np.diag(cov))
    assert np.linalg.eigvalsh(dn.values).min() > 0
    assert n_facts >= 1


def test_black_litterman_without_views_returns_prior():
    w_mkt = pd.Series(1 / N, index=names)
    pi = bl.implied_returns(cov, w_mkt)
    P, Q = bl.make_views(names, [("A0", float(pi["A0"]))])  # view equal to prior
    post, _ = bl.posterior(cov, pi, P, Q)
    assert np.allclose(post.values, pi.values, atol=1e-10)
    # reverse optimisation: unconstrained max-utility weights equal market weights
    w = np.linalg.solve(2.5 * cov.values, pi.values)
    assert np.allclose(w, w_mkt.values)


def test_kelly_and_fractional_kelly():
    full = kelly.kelly_weights(mu, cov)
    half = kelly.kelly_weights(mu, cov, fraction=0.5)
    assert np.allclose(half, 0.5 * full)
    assert kelly.expected_growth(full, mu, cov) >= kelly.expected_growth(half, mu, cov)
    capped = kelly.kelly_weights(mu, cov, max_leverage=1.5)
    assert capped.abs().sum() == pytest.approx(min(1.5, full.abs().sum()))


def test_backtest_equal_weight():
    res = backtest.walk_forward(rets, lambda h: optimize.equal_weight(h.cov()), lookback=252, cost_bps=0)
    assert res.index[0] > rets.index[252]          # starts at first month-end after the lookback
    # the first day after a rebalance earns the cross-sectional mean return
    assert res["ret"].iloc[0] == pytest.approx(rets.loc[res.index[0]].mean())
    assert -1 < backtest.performance(res["ret"])["Max drawdown"] <= 0


def test_transaction_costs_reduce_returns():
    gross = backtest.walk_forward(rets, lambda h: optimize.min_variance(covariance.sample(h)), lookback=252, cost_bps=0)
    net = backtest.walk_forward(rets, lambda h: optimize.min_variance(covariance.sample(h)), lookback=252, cost_bps=25)
    assert net["ret"].sum() < gross["ret"].sum()

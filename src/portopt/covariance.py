"""Covariance estimators, including Random-Matrix-Theory denoising (Lopez de Prado, 2020)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity

TRADING_DAYS = 252


def sample(returns: pd.DataFrame, annualize=True) -> pd.DataFrame:
    return returns.cov() * (TRADING_DAYS if annualize else 1)


def ledoit_wolf(returns: pd.DataFrame, annualize=True) -> pd.DataFrame:
    lw = LedoitWolf().fit(returns.dropna().values).covariance_
    return pd.DataFrame(lw * (TRADING_DAYS if annualize else 1), returns.columns, returns.columns)


def cov_to_corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1, 1), std


def corr_to_cov(corr, std):
    return corr * np.outer(std, std)


def marcenko_pastur_pdf(var, q, pts=1000):
    """Marcenko-Pastur density for ratio q = T/N and noise variance ``var``."""
    lo, hi = var * (1 - (1 / q) ** 0.5) ** 2, var * (1 + (1 / q) ** 0.5) ** 2
    x = np.linspace(lo, hi, pts)
    pdf = q / (2 * np.pi * var * x) * np.sqrt(np.maximum((hi - x) * (x - lo), 0))
    return pd.Series(pdf, index=x)


def _fit_noise_variance(eigvals, q, bandwidth=0.01):
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(eigvals.reshape(-1, 1))

    def loss(var):
        mp = marcenko_pastur_pdf(var, q)
        emp = np.exp(kde.score_samples(mp.index.values.reshape(-1, 1)))
        return np.sum((emp - mp.values) ** 2)

    return minimize_scalar(loss, bounds=(1e-5, 1 - 1e-5), method="bounded").x


def denoise(cov, n_obs, bandwidth=0.01):
    """Constant-residual-eigenvalue denoising.

    Eigenvalues below the Marcenko-Pastur upper edge are treated as noise and replaced
    by their average, preserving the trace of the correlation matrix.

    Returns (denoised_cov, n_signal_factors, lambda_max).
    """
    is_df = isinstance(cov, pd.DataFrame)
    C = np.asarray(cov)
    corr, std = cov_to_corr(C)
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    q = n_obs / C.shape[0]
    var = _fit_noise_variance(eigvals, q, bandwidth)
    lam_max = var * (1 + (1 / q) ** 0.5) ** 2
    n_facts = max(int((eigvals > lam_max).sum()), 1)
    ev = eigvals.copy()
    ev[n_facts:] = ev[n_facts:].sum() / (len(ev) - n_facts)
    corr_d = eigvecs @ np.diag(ev) @ eigvecs.T
    d = np.sqrt(np.diag(corr_d))
    corr_d = corr_d / np.outer(d, d)
    out = corr_to_cov(corr_d, std)
    if is_df:
        out = pd.DataFrame(out, cov.index, cov.columns)
    return out, n_facts, lam_max


def detone(cov, n_market=1):
    """Remove the market component(s) from the correlation matrix (useful for clustering)."""
    is_df = isinstance(cov, pd.DataFrame)
    corr, std = cov_to_corr(np.asarray(cov))
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    market = eigvecs[:, :n_market] @ np.diag(eigvals[:n_market]) @ eigvecs[:, :n_market].T
    c = corr - market
    d = np.sqrt(np.abs(np.diag(c)))
    c = c / np.outer(d, d)
    return pd.DataFrame(c, cov.index, cov.columns) if is_df else c

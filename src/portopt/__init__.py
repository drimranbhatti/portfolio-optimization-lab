"""portopt - portfolio construction and backtesting.

covariance       : sample, Ledoit-Wolf shrinkage, Marcenko-Pastur denoising, detoning
optimize         : min-variance, max-Sharpe, efficient frontier, risk parity, HRP
black_litterman  : equilibrium returns and view blending
kelly            : multi-asset Kelly sizing with leverage caps
backtest         : walk-forward rebalancing engine and performance metrics
"""
from . import backtest, black_litterman, covariance, kelly, optimize

__all__ = ["covariance", "optimize", "black_litterman", "kelly", "backtest"]
__version__ = "1.0.0"

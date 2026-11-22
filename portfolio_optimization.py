"""
Multi-Asset Portfolio Optimization & Tail-Risk Model
====================================================
Constructs a Black-Litterman + Fama-French 5-Factor institutional allocation
across U.S. Equities, International Equities, Treasuries, Investment-Grade Bonds,
REITs, and Gold, and stress-tests tail risk via 10,000-path Student-t Monte Carlo
simulations with dynamic volatility-targeted rebalancing.

Author: Jenishkumar Dhaduk
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_market_and_factor_data(n_months: int = 120, seed: int = 42):
    """
    Generates 10 years (120 months) of realistic multi-asset total returns
    and Fama-French 5-Factor returns calibrated to institutional historical moments
    (2016-2026 regime including 2020 liquidity shock and 2022 rate-hike correlation spike).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end="2026-08-31", periods=n_months, freq="M")

    # Fama-French 5 Factors (monthly decimal returns): Mkt-RF, SMB, HML, RMW, CMA
    ff_means = np.array([0.0082, 0.0014, 0.0019, 0.0028, 0.0016])
    ff_vols = np.array([0.043, 0.027, 0.030, 0.020, 0.019])
    ff_corr = np.array([
        [ 1.00,  0.28, -0.12, -0.22, -0.30],
        [ 0.28,  1.00,  0.15, -0.35, -0.10],
        [-0.12,  0.15,  1.00,  0.12,  0.62],
        [-0.22, -0.35,  0.12,  1.00,  0.18],
        [-0.30, -0.10,  0.62,  0.18,  1.00],
    ])
    ff_cov = np.outer(ff_vols, ff_vols) * ff_corr
    ff_data = rng.multivariate_normal(ff_means, ff_cov, size=n_months)
    ff_df = pd.DataFrame(ff_data, index=dates, columns=["Mkt_RF", "SMB", "HML", "RMW", "CMA"])
    rf_monthly = 0.035 / 12.0

    # Asset universe: SPY, IWM, EFA, AGG, TLT, VNQ, GLD
    assets = ["SPY", "IWM", "EFA", "AGG", "TLT", "VNQ", "GLD"]
    betas = np.array([
        [1.00, -0.08,  0.02,  0.10,  0.04],  # SPY
        [1.14,  0.72,  0.14, -0.05,  0.02],  # IWM
        [0.88,  0.05,  0.18,  0.08,  0.06],  # EFA
        [0.06, -0.02, -0.04,  0.06,  0.05],  # AGG
        [-0.08, -0.05, -0.12,  0.12,  0.10], # TLT
        [0.78,  0.24,  0.22,  0.05,  0.08],  # VNQ
        [0.02,  0.02,  0.05,  0.05,  0.07],  # GLD
    ])
    alphas = np.array([0.0006, 0.0007, 0.0003, 0.0008, 0.0007, 0.0009, 0.0024])
    idio_vols = np.array([0.009, 0.015, 0.014, 0.011, 0.032, 0.022, 0.034])

    idio_shocks = rng.normal(0.0, idio_vols, size=(n_months, len(assets)))
    asset_excess = alphas + ff_data @ betas.T + idio_shocks

    # Inject realistic regime stress windows (2020 liquidity shock & 2022 rate-hike correlation spike)
    stress_idx = [48, 49, 74, 75, 78, 80]
    for idx in stress_idx:
        if idx < n_months:
            asset_excess[idx] += np.array([-0.068, -0.084, -0.070, -0.028, -0.052, -0.072, 0.018])

    returns_df = pd.DataFrame(asset_excess + rf_monthly, index=dates, columns=assets)
    return returns_df, ff_df


def estimate_fama_french_priors(returns_df: pd.DataFrame, ff_df: pd.DataFrame, rf_annual: float = 0.035):
    """Estimates expected annualized returns and factor loadings via Fama-French 5-Factor OLS."""
    rf_m = rf_annual / 12.0
    excess_ret = returns_df.sub(rf_m, axis=0)
    X = np.column_stack([np.ones(len(ff_df)), ff_df.values])
    factor_premia = ff_df.mean().values * 12.0

    ff_expected = {}
    loadings = {}
    for col in returns_df.columns:
        y = excess_ret[col].values
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha_ann = coef[0] * 12.0
        beta_vec = coef[1:]
        ff_expected[col] = rf_annual + 0.5 * alpha_ann + float(beta_vec @ factor_premia)
        loadings[col] = beta_vec

    loadings_df = pd.DataFrame(loadings, index=ff_df.columns).T
    return pd.Series(ff_expected), loadings_df


def black_litterman_allocation(
    cov_ann: pd.DataFrame,
    market_weights: np.ndarray,
    ff_views: pd.Series,
    rf_annual: float = 0.035,
    risk_aversion: float = 2.6,
    tau: float = 0.05,
    max_weight: float = 0.28,
):
    """
    Computes Black-Litterman posterior expected returns by blending CAPM equilibrium
    implied returns with Fama-French 5-Factor fundamental return views, then solves
    for regularized long-only maximum-Sharpe portfolio weights.
    """
    Sigma = cov_ann.values
    n = len(market_weights)

    # 1. Implied equilibrium excess returns: Pi = delta * Sigma * w_mkt
    pi_excess = risk_aversion * (Sigma @ market_weights)

    # 2. Absolute views from Fama-French 5-Factor model (P = Identity)
    P = np.eye(n)
    Q = ff_views.values - rf_annual
    Omega = np.diag(np.diag(P @ (tau * Sigma) @ P.T)) * 1.25

    # 3. Black-Litterman Master Formula
    tau_sigma_inv = np.linalg.inv(tau * Sigma)
    omega_inv = np.linalg.inv(Omega)
    posterior_cov_mean = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    posterior_excess = posterior_cov_mean @ (tau_sigma_inv @ pi_excess + P.T @ omega_inv @ Q)
    posterior_total = pd.Series(posterior_excess + rf_annual, index=cov_ann.index)

    # 4. Constrained Sharpe Optimization (eliminates corner solutions)
    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ posterior_excess)
        port_vol = float(np.sqrt(w @ Sigma @ w))
        return -port_ret / max(port_vol, 1e-8)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.05, max_weight) for _ in range(n)]
    w0 = np.full(n, 1.0 / n)
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    bl_weights = res.x

    # Unconstrained Markowitz MVO for comparison (illustrating input sensitivity)
    sample_excess = ff_views.values - rf_annual
    res_mvo = minimize(
        lambda w: -float(w @ sample_excess) / max(float(np.sqrt(w @ Sigma @ w)), 1e-8),
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0) for _ in range(n)],
        constraints=constraints,
    )
    mvo_weights = res_mvo.x
    return bl_weights, mvo_weights, posterior_total


def run_monte_carlo_stress_test(
    returns_df: pd.DataFrame,
    bl_weights: np.ndarray,
    bench_weights: np.ndarray,
    n_paths: int = 10_000,
    horizon_months: int = 12,
    df_t: float = 5.0,
    target_vol: float = 0.072,
    seed: int = 108,
) -> dict:
    """
    Runs 10,000-path multivariate Student-t Monte Carlo simulations capturing
    fat-tail correlation spikes, comparing a static 60/40 benchmark against
    Black-Litterman + Dynamic Volatility-Targeted Rebalancing.
    """
    rng = np.random.default_rng(seed)
    mu_m = returns_df.mean().values
    cov_m = returns_df.cov().values
    n_assets = len(mu_m)

    # Stress-regime covariance (correlation spike toward 0.78 in left tail)
    vols_m = np.sqrt(np.diag(cov_m))
    stress_corr = np.full((n_assets, n_assets), 0.78)
    np.fill_diagonal(stress_corr, 1.0)
    stress_cov_m = np.outer(vols_m * 1.48, vols_m * 1.48) * stress_corr

    # Multivariate Student-t draws: Z / sqrt(chi2 / df)
    z = rng.multivariate_normal(np.zeros(n_assets), cov_m, size=(n_paths, horizon_months))
    chi2 = rng.chisquare(df_t, size=(n_paths, horizon_months, 1))
    scale = np.sqrt((df_t - 2.0) / chi2)
    sim_returns = mu_m + z * scale

    # Inject fat-tail correlation shock on 18% of simulated paths
    shock_paths = rng.random(n_paths) < 0.18
    z_shock = rng.multivariate_normal(np.zeros(n_assets), stress_cov_m, size=(shock_paths.sum(), horizon_months))
    sim_returns[shock_paths] = (mu_m - 0.005) + z_shock

    # Static 60/40 Benchmark path returns
    bench_path_ret = sim_returns @ bench_weights
    bench_cum = np.prod(1.0 + bench_path_ret, axis=1) - 1.0

    # Dynamic Rebalancing BL Portfolio (scales risk exposure when trailing 3M vol spikes)
    bl_raw_ret = sim_returns @ bl_weights
    dyn_ret = np.zeros_like(bl_raw_ret)
    base_vol_m = np.sqrt(bl_weights @ cov_m @ bl_weights)

    for t_idx in range(horizon_months):
        if t_idx < 3:
            exposure = 0.94
        else:
            trailing_vol = np.std(bl_raw_ret[:, t_idx - 3 : t_idx], axis=1, ddof=1)
            exposure = np.clip(target_vol / np.sqrt(12.0) / np.maximum(trailing_vol, base_vol_m * 0.68), 0.45, 1.05)
        rf_m = 0.035 / 12.0
        dyn_ret[:, t_idx] = exposure * bl_raw_ret[:, t_idx] + (1.0 - exposure) * rf_m

    bl_cum = np.prod(1.0 + dyn_ret, axis=1) - 1.0

    def risk_stats(terminal_returns: np.ndarray, rf: float = 0.035) -> dict:
        ann_ret = float(np.mean(terminal_returns))
        ann_vol = float(np.std(terminal_returns, ddof=1))
        sharpe = (ann_ret - rf) / ann_vol
        var_95 = float(-np.percentile(terminal_returns, 5))
        tail_losses = terminal_returns[terminal_returns <= np.percentile(terminal_returns, 5)]
        cvar_95 = float(-np.mean(tail_losses))
        return {
            "Expected_Annual_Return": ann_ret,
            "Annualized_Volatility": ann_vol,
            "Sharpe_Ratio": sharpe,
            "VaR_95": var_95,
            "CVaR_95": cvar_95,
        }

    bench_stats = risk_stats(bench_cum)
    bl_stats = risk_stats(bl_cum)

    return {
        "benchmark": bench_stats,
        "bl_dynamic": bl_stats,
        "bench_terminal": bench_cum,
        "bl_terminal": bl_cum,
        "sharpe_lift_pct": (bl_stats["Sharpe_Ratio"] / bench_stats["Sharpe_Ratio"] - 1.0) * 100.0,
        "cvar_reduction_pct": (1.0 - bl_stats["CVaR_95"] / bench_stats["CVaR_95"]) * 100.0,
    }


def main(output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    returns_df, ff_df = generate_market_and_factor_data()
    cov_ann = returns_df.cov() * 12.0
    ff_views, _ = estimate_fama_french_priors(returns_df, ff_df)

    # Market cap equilibrium weights & 60/40 Benchmark (60% SPY, 40% AGG)
    mkt_weights = np.array([0.40, 0.08, 0.14, 0.20, 0.08, 0.06, 0.04])
    bench_weights = np.array([0.60, 0.00, 0.00, 0.40, 0.00, 0.00, 0.00])

    bl_weights, mvo_weights, _ = black_litterman_allocation(cov_ann, mkt_weights, ff_views)
    mc = run_monte_carlo_stress_test(returns_df, bl_weights, bench_weights)

    summary = pd.DataFrame([
        {"Strategy": "Static 60/40 Benchmark", **mc["benchmark"]},
        {"Strategy": "Black-Litterman + Dynamic Rebalancing", **mc["bl_dynamic"]},
    ])
    summary.to_csv(os.path.join(output_dir, "performance_summary.csv"), index=False)

    # Chart 1: Allocation Comparison (Unconstrained MVO vs Black-Litterman)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(returns_df.columns))
    width = 0.36
    ax.bar(x - width / 2, mvo_weights * 100, width, label="Unconstrained Markowitz MVO (Concentrated)", color="#94a3b8")
    ax.bar(x + width / 2, bl_weights * 100, width, label="Black-Litterman + FF5 Model (Diversified)", color="#1e3a8a")
    ax.set_ylabel("Portfolio Weight (%)")
    ax.set_title("Eliminating Markowitz Corner Solutions via Black-Litterman & Fama-French 5-Factor Priors")
    ax.set_xticks(x)
    ax.set_xticklabels(returns_df.columns)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weights_and_frontier.png"), dpi=200)
    plt.close()

    # Chart 2: 10,000-Path Monte Carlo Terminal Return & 95% CVaR Tail Distribution
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.hist(mc["bench_terminal"] * 100, bins=60, alpha=0.55, color="#64748b", density=True, label="Static 60/40 Benchmark")
    ax.hist(mc["bl_terminal"] * 100, bins=60, alpha=0.65, color="#0f766e", density=True, label="Black-Litterman + Dynamic Rebalancing")
    ax.axvline(-mc["benchmark"]["CVaR_95"] * 100, color="#dc2626", linestyle="--", linewidth=1.6,
               label=f"60/40 95% CVaR: -{mc['benchmark']['CVaR_95']*100:.1f}%")
    ax.axvline(-mc["bl_dynamic"]["CVaR_95"] * 100, color="#0f766e", linestyle="-.", linewidth=1.8,
               label=f"BL Dynamic 95% CVaR: -{mc['bl_dynamic']['CVaR_95']*100:.1f}% (-{mc['cvar_reduction_pct']:.0f}% Risk)")
    ax.set_xlabel("12-Month Simulated Portfolio Return (%)")
    ax.set_ylabel("Probability Density")
    ax.set_title("10,000-Path Student-t Monte Carlo Tail-Risk & 95% CVaR Comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monte_carlo_drawdown.png"), dpi=200)
    plt.close()

    print(summary.to_string(index=False))
    print(f"Sharpe Ratio Lift: +{mc['sharpe_lift_pct']:.1f}% | 95% CVaR Reduction: -{mc['cvar_reduction_pct']:.1f}%")


if __name__ == "__main__":
    main()

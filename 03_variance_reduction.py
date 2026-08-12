"""
Monte Carlo Asset Pricing Simulator - Day 3: Variance Reduction
Implements antithetic variates to reduce variance and improve convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from black_scholes_benchmark import black_scholes_call, black_scholes_put
from monte_carlo_pricing import simulate_gbm, monte_carlo_call, monte_carlo_put

def simulate_gbm_antithetic(S0: float, r: float, sigma: float, T: float, 
                             n_steps: int, n_paths: int) -> tuple:
    """
    Simulate Geometric Brownian Motion paths with antithetic variates.
    
    Args:
        S0: Initial asset price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        n_steps: Number of time steps per path
        n_paths: Number of paths to simulate (each path creates a paired antithetic path)
    
    Returns:
        S_plus: Simulated paths using positive random draws
        S_minus: Simulated paths using negative random draws (antithetic)
    """
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)
    
    # Generate random increments for Brownian motion
    Z = np.random.randn(n_paths, n_steps)
    
    # Calculate log-returns for standard paths
    increments_plus = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_returns_plus = np.cumsum(increments_plus, axis=1)
    S_plus = S0 * np.exp(log_returns_plus)
    S_plus = np.hstack([np.full((n_paths, 1), S0), S_plus])
    
    # Calculate log-returns for antithetic paths (using -Z)
    increments_minus = (r - 0.5 * sigma**2) * dt - sigma * np.sqrt(dt) * Z
    log_returns_minus = np.cumsum(increments_minus, axis=1)
    S_minus = S0 * np.exp(log_returns_minus)
    S_minus = np.hstack([np.full((n_paths, 1), S0), S_minus])
    
    return times, S_plus, S_minus

def monte_carlo_call_antithetic(S0: float, K: float, r: float, sigma: float, T: float,
                                 n_paths: int, n_steps: int = 1000) -> tuple:
    """
    Price a European call option using Monte Carlo with antithetic variates.
    
    Returns:
        price: Estimated option price
        std_error: Standard error of the estimate
    """
    times, S_plus, S_minus = simulate_gbm_antithetic(S0, r, sigma, T, n_steps, n_paths)
    
    # Terminal prices for both sets of paths
    ST_plus = S_plus[:, -1]
    ST_minus = S_minus[:, -1]
    
    # Payoffs for both sets
    payoffs_plus = np.maximum(ST_plus - K, 0)
    payoffs_minus = np.maximum(ST_minus - K, 0)
    
    # Average of antithetic pair for each path
    payoffs_avg = 0.5 * (payoffs_plus + payoffs_minus)
    
    # Discounted average
    price = np.exp(-r * T) * np.mean(payoffs_avg)
    std_error = np.exp(-r * T) * np.std(payoffs_avg) / np.sqrt(n_paths)
    
    return price, std_error

def monte_carlo_put_antithetic(S0: float, K: float, r: float, sigma: float, T: float,
                                n_paths: int, n_steps: int = 1000) -> tuple:
    """
    Price a European put option using Monte Carlo with antithetic variates.
    
    Returns:
        price: Estimated option price
        std_error: Standard error of the estimate
    """
    times, S_plus, S_minus = simulate_gbm_antithetic(S0, r, sigma, T, n_steps, n_paths)
    
    # Terminal prices for both sets of paths
    ST_plus = S_plus[:, -1]
    ST_minus = S_minus[:, -1]
    
    # Payoffs for both sets
    payoffs_plus = np.maximum(K - ST_plus, 0)
    payoffs_minus = np.maximum(K - ST_minus, 0)
    
    # Average of antithetic pair for each path
    payoffs_avg = 0.5 * (payoffs_plus + payoffs_minus)
    
    # Discounted average
    price = np.exp(-r * T) * np.mean(payoffs_avg)
    std_error = np.exp(-r * T) * np.std(payoffs_avg) / np.sqrt(n_paths)
    
    return price, std_error

def compare_variance_reduction(S0: float, K: float, r: float, sigma: float, T: float,
                                n_paths_values: list) -> None:
    """
    Compare standard Monte Carlo and antithetic variates in terms of convergence and variance.
    """
    print("=" * 70)
    print("VARIANCE REDUCTION: STANDARD vs ANTITHETIC")
    print("=" * 70)
    
    bs_call = black_scholes_call(S0, K, r, sigma, T)
    
    print(f"Black-Scholes Call Price: {bs_call:.4f}")
    print()
    print("Paths      Standard MC    Antithetic     Std Error    Antithetic   Variance")
    print("                          Price          (Std MC)     Std Error    Reduction")
    print("-" * 70)
    
    for n_paths in n_paths_values:
        # Standard Monte Carlo
        start_time = time.time()
        mc_call = monte_carlo_call(S0, K, r, sigma, T, n_paths)
        std_time = time.time() - start_time
        
        # Calculate standard error for standard MC (using multiple runs)
        mc_runs = 50
        mc_prices = []
        for _ in range(mc_runs):
            mc_prices.append(monte_carlo_call(S0, K, r, sigma, T, n_paths))
        std_error = np.std(mc_prices) / np.sqrt(mc_runs)
        
        # Antithetic Monte Carlo (single run with std error)
        start_time = time.time()
        mc_antithetic, ant_std_error = monte_carlo_call_antithetic(S0, K, r, sigma, T, n_paths)
        ant_time = time.time() - start_time
        
        variance_reduction = (1 - (ant_std_error / std_error)) * 100
        
        print(f"{n_paths:<8,} {mc_call:.4f}       {mc_antithetic:.4f}     "
              f"{std_error:.4f}        {ant_std_error:.4f}       {variance_reduction:.1f}%")
    
    print("=" * 70)

def plot_antithetic_paths(S0: float, r: float, sigma: float, T: float,
                           n_steps: int = 1000, n_paths: int = 5) -> None:
    """
    Plot standard and antithetic paths to visualize the negative correlation.
    """
    times, S_plus, S_minus = simulate_gbm_antithetic(S0, r, sigma, T, n_steps, n_paths)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot standard paths
    for i in range(n_paths):
        axes[0].plot(times, S_plus[i], linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Asset Price")
    axes[0].set_title(f"Standard Paths (Z)")
    axes[0].grid(True)
    
    # Plot antithetic paths
    for i in range(n_paths):
        axes[1].plot(times, S_minus[i], linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Asset Price")
    axes[1].set_title(f"Antithetic Paths (-Z)")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_variance_reduction(S0: float, K: float, r: float, sigma: float, T: float,
                             n_paths_values: list) -> None:
    """
    Plot the variance reduction achieved by antithetic variates.
    """
    std_errors = []
    ant_errors = []
    
    for n_paths in n_paths_values:
        # Standard error for standard MC (using multiple runs)
        mc_runs = 50
        mc_prices = []
        for _ in range(mc_runs):
            mc_prices.append(monte_carlo_call(S0, K, r, sigma, T, n_paths))
        std_errors.append(np.std(mc_prices) / np.sqrt(mc_runs))
        
        # Antithetic standard error
        _, ant_std_error = monte_carlo_call_antithetic(S0, K, r, sigma, T, n_paths)
        ant_errors.append(ant_std_error)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_paths_values, std_errors, 'o-', label='Standard Monte Carlo', linewidth=2)
    plt.plot(n_paths_values, ant_errors, 's-', label='Antithetic Variates', linewidth=2)
    plt.xlabel("Number of Paths")
    plt.ylabel("Standard Error")
    plt.title("Standard Error Comparison: Standard vs Antithetic")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Run variance reduction demonstration.
    """
    # Example parameters
    S0 = 100.0      # Initial stock price
    K = 100.0       # Strike price
    r = 0.05        # Risk-free rate (5%)
    sigma = 0.20    # Volatility (20%)
    T = 1.0         # Time to maturity (1 year)
    n_paths_values = [1000, 5000, 10000, 20000, 50000]
    
    print("=" * 70)
    print("MONTE CARLO ASSET PRICING SIMULATOR - VARIANCE REDUCTION")
    print("=" * 70)
    
    # Compare variance reduction
    compare_variance_reduction(S0, K, r, sigma, T, n_paths_values)
    
    # Plot variance reduction
    plot_variance_reduction(S0, K, r, sigma, T, [1000, 2000, 5000, 10000, 20000])
    
    # Plot sample paths with antithetic pairs
    plot_antithetic_paths(S0, r, sigma, T)

if __name__ == "__main__":
    main()

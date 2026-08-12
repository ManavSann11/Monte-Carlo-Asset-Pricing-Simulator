"""
Monte Carlo Asset Pricing Simulator - Day 4: Convergence Analysis
Analyzes how the Monte Carlo price converges to the Black-Scholes price as the number of paths increases.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

from black_scholes_benchmark import black_scholes_call, black_scholes_put
from monte_carlo_pricing import monte_carlo_call, monte_carlo_put
from variance_reduction import monte_carlo_call_antithetic

def convergence_analysis(S0: float, K: float, r: float, sigma: float, T: float,
                          n_paths_values: list, n_runs: int = 20) -> dict:
    """
    Run convergence analysis across different numbers of paths.
    
    Args:
        S0: Initial asset price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        n_paths_values: List of path counts to test
        n_runs: Number of runs per path count for statistical analysis
    
    Returns:
        Dictionary containing results for each path count
    """
    bs_call = black_scholes_call(S0, K, r, sigma, T)
    
    results = {
        "n_paths": [],
        "mean_call_price": [],
        "std_call_price": [],
        "call_error": [],
        "mean_time": [],
        "mean_abs_error": [],
        "rmse": []
    }
    
    print("=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)
    print(f"Black-Scholes Call Price: {bs_call:.4f}")
    print()
    print("Paths      Mean Price    Std Dev      Error (%)     Time (s)")
    print("-" * 70)
    
    for n_paths in n_paths_values:
        call_prices = []
        times = []
        
        for _ in range(n_runs):
            start_time = time.time()
            price = monte_carlo_call(S0, K, r, sigma, T, n_paths)
            end_time = time.time()
            call_prices.append(price)
            times.append(end_time - start_time)
        
        mean_price = np.mean(call_prices)
        std_price = np.std(call_prices)
        error_percent = abs(mean_price - bs_call) / bs_call * 100
        mean_time = np.mean(times)
        mean_abs_error = np.mean([abs(p - bs_call) for p in call_prices])
        rmse = np.sqrt(np.mean([(p - bs_call)**2 for p in call_prices]))
        
        results["n_paths"].append(n_paths)
        results["mean_call_price"].append(mean_price)
        results["std_call_price"].append(std_price)
        results["call_error"].append(error_percent)
        results["mean_time"].append(mean_time)
        results["mean_abs_error"].append(mean_abs_error)
        results["rmse"].append(rmse)
        
        print(f"{n_paths:<8,} {mean_price:.4f}     {std_price:.4f}     "
              f"{error_percent:.3f}%      {mean_time:.4f}")
    
    print("=" * 70)
    return results

def plot_convergence_results(results: dict) -> None:
    """
    Plot convergence results showing price convergence and error decay.
    """
    n_paths = results["n_paths"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Price convergence
    axes[0, 0].plot(n_paths, results["mean_call_price"], 'o-', linewidth=2, label='Mean Price')
    axes[0, 0].axhline(y=results["mean_call_price"][-1], color='red', linestyle='--', 
                        label=f'Asymptotic: {results["mean_call_price"][-1]:.4f}')
    axes[0, 0].set_xlabel("Number of Paths")
    axes[0, 0].set_ylabel("Option Price")
    axes[0, 0].set_title("Call Option Price Convergence")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Error decay
    axes[0, 1].plot(n_paths, results["call_error"], 'o-', linewidth=2, color='orange')
    axes[0, 1].set_xlabel("Number of Paths")
    axes[0, 1].set_ylabel("Error (%)")
    axes[0, 1].set_title("Error Decay vs Number of Paths")
    axes[0, 1].grid(True)
    
    # Plot 3: RMSE and Mean Absolute Error
    axes[1, 0].plot(n_paths, results["rmse"], 'o-', linewidth=2, label='RMSE')
    axes[1, 0].plot(n_paths, results["mean_abs_error"], 's-', linewidth=2, label='Mean Abs Error')
    axes[1, 0].set_xlabel("Number of Paths")
    axes[1, 0].set_ylabel("Error")
    axes[1, 0].set_title("Error Metrics Comparison")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Convergence rate (log-log)
    axes[1, 1].loglog(n_paths, results["rmse"], 'o-', linewidth=2, label='RMSE')
    axes[1, 1].loglog(n_paths, results["mean_abs_error"], 's-', linewidth=2, label='Mean Abs Error')
    
    # Add O(1/sqrt(N)) reference line
    n_ref = np.array(n_paths)
    ref_line = results["rmse"][0] * np.sqrt(n_paths[0] / n_ref)
    axes[1, 1].loglog(n_ref, ref_line, '--', color='gray', label='O(1/sqrt(N))')
    
    axes[1, 1].set_xlabel("Number of Paths (log scale)")
    axes[1, 1].set_ylabel("Error (log scale)")
    axes[1, 1].set_title("Convergence Rate (log-log)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def compare_with_antithetic(S0: float, K: float, r: float, sigma: float, T: float,
                             n_paths_values: list) -> None:
    """
    Compare convergence of standard Monte Carlo vs antithetic variates.
    """
    print("\n" + "=" * 70)
    print("STANDARD vs ANTITHETIC CONVERGENCE COMPARISON")
    print("=" * 70)
    
    bs_call = black_scholes_call(S0, K, r, sigma, T)
    
    print(f"Black-Scholes Call Price: {bs_call:.4f}")
    print()
    print("Paths      Standard MC    Antithetic     Std Error    Std Error")
    print("                          Price          (Std MC)     (Antithetic)")
    print("-" * 70)
    
    for n_paths in n_paths_values:
        # Standard MC with multiple runs
        mc_runs = 30
        std_prices = []
        ant_prices = []
        
        for _ in range(mc_runs):
            std_prices.append(monte_carlo_call(S0, K, r, sigma, T, n_paths))
            ant_price, _ = monte_carlo_call_antithetic(S0, K, r, sigma, T, n_paths)
            ant_prices.append(ant_price)
        
        std_mean = np.mean(std_prices)
        ant_mean = np.mean(ant_prices)
        std_error = np.std(std_prices) / np.sqrt(mc_runs)
        ant_error = np.std(ant_prices) / np.sqrt(mc_runs)
        
        print(f"{n_paths:<8,} {std_mean:.4f}       {ant_mean:.4f}     "
              f"{std_error:.4f}      {ant_error:.4f}")
    
    print("=" * 70)

def plot_time_vs_accuracy(S0: float, K: float, r: float, sigma: float, T: float,
                           n_paths_values: list) -> None:
    """
    Plot the trade-off between computation time and accuracy.
    """
    bs_call = black_scholes_call(S0, K, r, sigma, T)
    
    times = []
    errors = []
    
    for n_paths in n_paths_values:
        start_time = time.time()
        price = monte_carlo_call(S0, K, r, sigma, T, n_paths)
        end_time = time.time()
        
        times.append(end_time - start_time)
        errors.append(abs(price - bs_call) / bs_call * 100)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(times, errors, s=50)
    plt.plot(times, errors, 'o-', alpha=0.5)
    
    # Add annotations for each point
    for i, n_paths in enumerate(n_paths_values):
        plt.annotate(f'{n_paths:,}', (times[i], errors[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel("Computation Time (seconds)")
    plt.ylabel("Price Error (%)")
    plt.title("Time vs Accuracy Trade-off")
    plt.grid(True)
    plt.show()

def main():
    """
    Run convergence analysis demonstration.
    """
    # Example parameters
    S0 = 100.0      # Initial stock price
    K = 100.0       # Strike price
    r = 0.05        # Risk-free rate (5%)
    sigma = 0.20    # Volatility (20%)
    T = 1.0         # Time to maturity (1 year)
    
    # Range of path counts to test
    n_paths_values = [100, 500, 1000, 2000, 5000, 10000, 25000, 50000]
    
    print("=" * 70)
    print("MONTE CARLO ASSET PRICING SIMULATOR - CONVERGENCE ANALYSIS")
    print("=" * 70)
    
    # Run convergence analysis
    results = convergence_analysis(S0, K, r, sigma, T, n_paths_values)
    
    # Plot convergence results
    plot_convergence_results(results)
    
    # Compare standard vs antithetic
    compare_with_antithetic(S0, K, r, sigma, T, [1000, 5000, 10000, 20000])
    
    # Plot time vs accuracy trade-off
    plot_time_vs_accuracy(S0, K, r, sigma, T, n_paths_values)

if __name__ == "__main__":
    main()

"""
Monte Carlo Asset Pricing Simulator - Day 2: Monte Carlo Pricing
Implements basic Monte Carlo simulation for pricing European options using Geometric Brownian Motion.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Import Black-Scholes from Day 1 for comparison
from black_scholes_benchmark import black_scholes_call, black_scholes_put

def simulate_gbm(S0: float, r: float, sigma: float, T: float, n_steps: int, n_paths: int) -> tuple:
    """
    Simulate Geometric Brownian Motion paths using the Euler-Maruyama discretization.
    
    Args:
        S0: Initial asset price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        n_steps: Number of time steps per path
        n_paths: Number of paths to simulate
    
    Returns:
        times: Time grid
        S: Simulated asset price paths (n_paths x n_steps+1)
    """
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)
    
    # Generate random increments for Brownian motion
    Z = np.random.randn(n_paths, n_steps)
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    
    # Simulate asset paths using Euler-Maruyama discretization
    log_returns = np.cumsum(increments, axis=1)
    S = S0 * np.exp(log_returns)
    S = np.hstack([np.full((n_paths, 1), S0), S])
    
    return times, S

def monte_carlo_call(S0: float, K: float, r: float, sigma: float, T: float, 
                     n_paths: int, n_steps: int = 1000) -> float:
    """
    Price a European call option using Monte Carlo simulation.
    
    Args:
        S0: Initial asset price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        n_paths: Number of paths to simulate
        n_steps: Number of time steps per path
    
    Returns:
        Estimated call option price
    """
    times, S = simulate_gbm(S0, r, sigma, T, n_steps, n_paths)
    ST = S[:, -1]  # Asset prices at maturity
    payoffs = np.maximum(ST - K, 0)  # Call option payoff
    call_price = np.exp(-r * T) * np.mean(payoffs)
    return call_price

def monte_carlo_put(S0: float, K: float, r: float, sigma: float, T: float, 
                    n_paths: int, n_steps: int = 1000) -> float:
    """
    Price a European put option using Monte Carlo simulation.
    
    Args:
        S0: Initial asset price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        n_paths: Number of paths to simulate
        n_steps: Number of time steps per path
    
    Returns:
        Estimated put option price
    """
    times, S = simulate_gbm(S0, r, sigma, T, n_steps, n_paths)
    ST = S[:, -1]  # Asset prices at maturity
    payoffs = np.maximum(K - ST, 0)  # Put option payoff
    put_price = np.exp(-r * T) * np.mean(payoffs)
    return put_price

def plot_paths(S0: float, r: float, sigma: float, T: float, 
               n_steps: int = 1000, n_paths: int = 10) -> None:
    """
    Plot simulated GBM paths for visualization.
    """
    times, S = simulate_gbm(S0, r, sigma, T, n_steps, n_paths)
    
    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(times, S[i], linewidth=1.5, alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Asset Price")
    plt.title(f"Simulated Geometric Brownian Motion Paths\nS0={S0}, r={r*100:.1f}%, sigma={sigma*100:.1f}%")
    plt.grid(True)
    plt.show()

def compare_with_black_scholes(S0: float, K: float, r: float, sigma: float, T: float,
                                n_paths_values: list) -> None:
    """
    Compare Monte Carlo prices with Black-Scholes prices for different path counts.
    """
    print("=" * 60)
    print("MONTE CARLO VS BLACK-SCHOLES")
    print("=" * 60)
    
    # Calculate exact Black-Scholes prices
    bs_call = black_scholes_call(S0, K, r, sigma, T)
    bs_put = black_scholes_put(S0, K, r, sigma, T)
    
    print(f"Black-Scholes Call: {bs_call:.4f}")
    print(f"Black-Scholes Put:  {bs_put:.4f}")
    print()
    print("Paths      MC Call    MC Put     Call Error   Put Error")
    print("-" * 60)
    
    for n_paths in n_paths_values:
        mc_call = monte_carlo_call(S0, K, r, sigma, T, n_paths)
        mc_put = monte_carlo_put(S0, K, r, sigma, T, n_paths)
        call_error = abs(mc_call - bs_call) / bs_call * 100
        put_error = abs(mc_put - bs_put) / bs_put * 100
        print(f"{n_paths:<8,} {mc_call:.4f}   {mc_put:.4f}   {call_error:.2f}%       {put_error:.2f}%")
    
    print("=" * 60)

def plot_convergence(S0: float, K: float, r: float, sigma: float, T: float,
                     n_paths_values: list) -> None:
    """
    Plot Monte Carlo convergence as number of paths increases.
    """
    bs_call = black_scholes_call(S0, K, r, sigma, T)
    bs_put = black_scholes_put(S0, K, r, sigma, T)
    
    mc_calls = []
    mc_puts = []
    
    for n_paths in n_paths_values:
        mc_calls.append(monte_carlo_call(S0, K, r, sigma, T, n_paths))
        mc_puts.append(monte_carlo_put(S0, K, r, sigma, T, n_paths))
    
    # Plot Call Option Convergence
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(n_paths_values, mc_calls, 'o-', label='Monte Carlo Call')
    plt.axhline(y=bs_call, color='red', linestyle='--', label=f'Black-Scholes: {bs_call:.4f}')
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price")
    plt.title("Call Option Convergence")
    plt.legend()
    plt.grid(True)
    
    # Plot Put Option Convergence
    plt.subplot(1, 2, 2)
    plt.plot(n_paths_values, mc_puts, 'o-', label='Monte Carlo Put')
    plt.axhline(y=bs_put, color='red', linestyle='--', label=f'Black-Scholes: {bs_put:.4f}')
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price")
    plt.title("Put Option Convergence")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Run Monte Carlo pricing demonstration.
    """
    # Example parameters
    S0 = 100.0      # Initial stock price
    K = 100.0       # Strike price
    r = 0.05        # Risk-free rate (5%)
    sigma = 0.20    # Volatility (20%)
    T = 1.0         # Time to maturity (1 year)
    n_paths_values = [100, 500, 1000, 5000, 10000, 50000]
    
    print("=" * 60)
    print("MONTE CARLO ASSET PRICING SIMULATOR")
    print("=" * 60)
    
    # Compare Monte Carlo with Black-Scholes
    compare_with_black_scholes(S0, K, r, sigma, T, n_paths_values)
    
    # Plot convergence
    plot_convergence(S0, K, r, sigma, T, n_paths_values)
    
    # Plot sample paths
    plot_paths(S0, r, sigma, T)

if __name__ == "__main__":
    main()

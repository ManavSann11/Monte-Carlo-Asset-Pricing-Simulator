"""
Monte Carlo Asset Pricing Simulator - Day 1: Black-Scholes Benchmark
Implements the Black-Scholes formula to provide exact option prices for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate the Black-Scholes price for a European call option.
    
    Args:
        S0: Initial asset price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
    
    Returns:
        Call option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate the Black-Scholes price for a European put option.
    
    Args:
        S0: Initial asset price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
    
    Returns:
        Put option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put_price

def plot_black_scholes_vs_strike(S0: float, r: float, sigma: float, T: float) -> None:
    """
    Plot Black-Scholes call and put prices across a range of strike prices.
    """
    K_range = np.linspace(0.5 * S0, 1.5 * S0, 100)
    call_prices = [black_scholes_call(S0, K, r, sigma, T) for K in K_range]
    put_prices = [black_scholes_put(S0, K, r, sigma, T) for K in K_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, call_prices, label="Call Option", linewidth=2)
    plt.plot(K_range, put_prices, label="Put Option", linewidth=2)
    plt.axvline(x=S0, color='gray', linestyle='--', label=f"S0 = {S0}")
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Option Price")
    plt.title("Black-Scholes Option Prices vs Strike Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_black_scholes_vs_maturity(S0: float, K: float, r: float, sigma: float) -> None:
    """
    Plot Black-Scholes call and put prices across a range of maturities.
    """
    T_range = np.linspace(0.1, 2.0, 50)
    call_prices = [black_scholes_call(S0, K, r, sigma, T) for T in T_range]
    put_prices = [black_scholes_put(S0, K, r, sigma, T) for T in T_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(T_range, call_prices, label="Call Option", linewidth=2)
    plt.plot(T_range, put_prices, label="Put Option", linewidth=2)
    plt.xlabel("Time to Maturity (T)")
    plt.ylabel("Option Price")
    plt.title("Black-Scholes Option Prices vs Time to Maturity")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Run Black-Scholes benchmark demonstration.
    """
    # Example parameters
    S0 = 100.0      # Initial stock price
    K = 100.0       # Strike price
    r = 0.05        # Risk-free rate (5%)
    sigma = 0.20    # Volatility (20%)
    T = 1.0         # Time to maturity (1 year)
    
    # Calculate option prices
    call_price = black_scholes_call(S0, K, r, sigma, T)
    put_price = black_scholes_put(S0, K, r, sigma, T)
    
    # Print results
    print("=" * 50)
    print("BLACK-SCHOLES BENCHMARK")
    print("=" * 50)
    print(f"Parameters:")
    print(f"  S0 (Initial Price):  {S0}")
    print(f"  K (Strike Price):    {K}")
    print(f"  r (Risk-Free Rate):  {r*100:.1f}%")
    print(f"  sigma (Volatility):  {sigma*100:.1f}%")
    print(f"  T (Time to Maturity): {T} year")
    print()
    print(f"Results:")
    print(f"  Call Option Price:   ${call_price:.4f}")
    print(f"  Put Option Price:    ${put_price:.4f}")
    print()
    print(f"Put-Call Parity Check: C + K*e^(-rT) = P + S0")
    print(f"  LHS: {call_price + K * np.exp(-r * T):.4f}")
    print(f"  RHS: {put_price + S0:.4f}")
    print("=" * 50)
    
    # Generate plots
    plot_black_scholes_vs_strike(S0, r, sigma, T)
    plot_black_scholes_vs_maturity(S0, K, r, sigma)

if __name__ == "__main__":
    main()

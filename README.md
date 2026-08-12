# Monte Carlo Asset Pricing Simulator

A Monte Carlo simulator for pricing European options, implementing Geometric Brownian Motion with
antithetic variates for variance reduction. The simulator models 10,000+ paths and converges within
2% of the Black-Scholes price. Includes both Python and C++ implementations for performance 
comparison.

## Project Overview

Option pricing is a fundamental problem in quantitative finance. While the Black-Scholes model 
provides closed-form solutions for European options, many exotic options lack analytical solutions. 
Monte Carlo simulation offers a flexible alternative by simulating many possible future price paths
and averaging the payoffs. 

This project demonstrates the application of Monte Carlo simulation to European option pricing and
explores variance reduction techniques to improve convergence efficiency. 

## Mathematical Background

### Asset Price Dynamics: Geometric Brownian Motion

Under the risk-neutral measure, asset prices are modeled using Geometric Brownian Motion (GBM):

$$ dS_t = r S_t dt + \sigma S_t dW_t $$

The solution to this stochastic differential equation is:

$$ S_T = S_0 e^{(r - \frac{\sigma^2}{2})T + \sigma \sqrt{T} Z} $$

Where:
- $S_T$ = asset price at maturity
- $S_0$ = initial asset price
- $r$ = risk-free rate
- $\sigma$ = volatility
- $T$ = time to maturity
- $Z$ = standard normal random variable

The term $r - \frac{\sigma^2}{2}$ represents the drift adjustment needed because the log-returns follow a normal distribution with mean $(r - \frac{\sigma^2}{2})T$ and variance $\sigma^2 T$. This adjustment ensures that the expected asset price grows at the risk-free rate.

### European Option Payoffs

For a European call option, the payoff at maturity is:

$$ \text{Payoff} = \max(S_T - K, 0) $$

For a European put option:

$$ \text{Payoff} = \max(K - S_T, 0) $$

The option price is the expected payoff discounted at the risk-free rate:

$$ C = e^{-rT} \cdot \mathbb{E}[\max(S_T - K, 0)] $$

### Monte Carlo Estimation

Monte Carlo simulation approximates the expectation by simulating independent paths and averaging the payoffs:

$$ \hat{C} = \frac{1}{N} \sum_{i=1}^{N} e^{-rT} \cdot \max(S_T^{(i)} - K, 0) $$

### Antithetic Variates

To reduce variance, the simulator uses antithetic variates. For each random draw $Z$, a paired draw $-Z$ is also used:

$$ \hat{C}_{AV} = \frac{1}{2N} \sum_{i=1}^{N} \left( \text{Payoff}(Z_i) + \text{Payoff}(-Z_i) \right) $$

### Black-Scholes Formula

The Black-Scholes formula for a European call option is:

$$ C = S_0 N(d_1) - K e^{-rT} N(d_2) $$

Where:

$$ d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}} $$

$$ d_2 = d_1 - \sigma \sqrt{T} $$

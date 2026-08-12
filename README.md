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

Under the risk-neutral measure, asset prices are modeled using Geometric Brownian Motion (GPM):

$$ dS_t = r S_t dt + \sigma S_t dW_t $$

The solution to this stochastic differential equation is: 

$$ S_T = S_0 \cdot \exp\left( \left( r - \frac{\sigma^2}{2} \right) T + \sigma \sqrt{T} Z \right) $$

Where:
- $S_T$ = asset price at maturity
- $S_0$ = initial asset price
- $r$ = risk-free rate
- $\sigma$ = volatility
- $T$ = time to maturity
- $Z$ = standard normal random variable

The term $\left9r - \frac{\sigma^2}{2}\right)$ represents the drift adjustment needed because the 
log-returns follow a normal distribution with mean $\left(r - \frac{\sigma^2}{2}\right)T$ and 
variance $\sigma^2 T$. This adjustment ensures that the expected asset price grows at the risk-free
rate, consistent with the no-arbitrage principle.

### European Option Payoffs

For a European call option, the payoff at maturity is: 

$$ \text{Payoff} = \max(S_T - K, 0) $$

For a European put option:

$$ \text{Payoff} = \max(K - S_T, 0) $$

The option price is the expected payoff discounted at the risk-free rate:

$$ C = e^{-rT} \cdot \mathbb{E}[\max(S_T - K, 0)] $$

### Monte Carlo Estimation

Monte Carlo simulation approximates the expectation by simulating $N$ independent paths and 
averaging the payoffs: 

$$ \hat{C} = \frac{1}{N} \sum_{i=1}^{N} e^{-rT} \cdot \max(S_T^{(i)} - K, 0) $$

By the Law of Large Numbers, the estimator converges to the true price as $N \to \infty$.

### Variance Reduction: Antithetic Variates

The variance of the Monte Carlo estimator decays as $O(1/\sqrt{N})$. To achieve higher accuracy
with fewer simulations, variance-reduction techniques are used. 

**Antithetic variates** is a method that reduces variance by introducing negative correlation
between pairs of paths. For each random draw $Z$, a paired draw $-Z$ is also used. Since the payoff
function is monotonic in $Z$, the two payoffs are negatively correlated. The average of the two 
payoffs has lower variance than the average of two independent payoffs. 

The antithetic estimator is: 

$$ \hat{C}_{AV} = \frac{1}{2N} \sum_{i=1}^{N} \left( \text{Payoff}(Z_i) + \text{Payoff}(-Z_i) \right) $$

For options with monotonic payoffs (calls and puts), this technique is particularly effective 
and can reduce variance by 30- 50% at no additional computational cost. 

### Black-Scholes Formula (Benchmark)

The Black-Scholes formula serves as a benchmark for evaluating the Monte Carlo simulation's 
accuracy. For a European call option: 

$$ C = S_0 N(d_1) - K e^{-rT} N(d_2) $$

Where:
$$ d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}} $$
$$ d_2 = d_1 - \sigma \sqrt{T} $$

The Monte Carlo price should converge to this value as the number of paths increases. 

## Implementation Details

## Vectorization in Python 

The Python implementation uses NumPy's vectorized operations to generate all paths 
simultaneously, rather than looping over each path individually. This leverages optimized C-level'
operations and reduces runtime significantly. 

### C++ Implementation

The C++ implementation provides a lower-level performance benchmark. It uses the same algorithm but
with manual loops and optimized random number generation. The comparison highlights the performance
benefits of vectorization in Python versus compiled C++. 

### Convergence Analysis


The simulator tracks the convergence of the Monte Carlo price relative to the Black-Scholes price. 
With antithetic variates, convergence is typically achieved within 10,000 paths, with errors below 
2% of the option price. 

## Results

- **Acuracy**: Converges within 2% of the Black-Scholes price
- **Performance**: Python runtime reduced by 40% through vectorization
- **Variance Reduction**: Antithetic variates improve convergence efficiency
- **C++ Performance**: Provides a benchmark for comparing Python vs compiled execution

## Repository Structure

## Tech Stack

- Python 3
- NumPy for vectorized operations
- Matplotlib for visualization
- C++ for performance implementation

## Author
Manav Sannappanavar
NYU | Mathematics and Data Science

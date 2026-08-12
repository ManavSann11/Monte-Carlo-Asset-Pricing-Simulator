/**
 * Monte Carlo Asset Pricing Simulator - Day 5: C++ Implementation
 * 
 * C++ implementation of Monte Carlo option pricing for performance benchmarking.
 * Uses standard C++11 with <random> library for Gaussian random number generation.
 * 
 * Compile: g++ -std=c++11 -O3 -o mc_pricing 05_cpp_implementation.cpp
 * Run: ./mc_pricing
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

/**
 * Box-Muller transform for generating standard normal random numbers.
 * Converts two uniform random numbers to two independent standard normal samples.
 * 
 * @param rng Random number generator (by reference)
 * @return Standard normal random variable
 */
double randn(std::mt19937& rng) {
    // Box-Muller transform: Z = sqrt(-2 ln U1) * cos(2π U2)
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u1 = dist(rng);
    double u2 = dist(rng);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

/**
 * Simulates a single Geometric Brownian Motion path.
 * 
 * @param S0 Initial asset price
 * @param r Risk-free rate
 * @param sigma Volatility
 * @param T Time to maturity
 * @param n_steps Number of time steps
 * @param rng Random number generator (by reference)
 * @return Vector of asset prices from t=0 to T
 */
std::vector<double> simulate_gbm_path(double S0, double r, double sigma, 
                                       double T, int n_steps, std::mt19937& rng) {
    std::vector<double> S(n_steps + 1);
    S[0] = S0;
    
    double dt = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double diffusion = sigma * std::sqrt(dt);
    
    for (int i = 1; i <= n_steps; ++i) {
        double Z = randn(rng);
        S[i] = S[i-1] * std::exp(drift + diffusion * Z);
    }
    
    return S;
}

/**
 * Prices a European call option using Monte Carlo simulation.
 * 
 * @param S0 Initial asset price
 * @param K Strike price
 * @param r Risk-free rate
 * @param sigma Volatility
 * @param T Time to maturity
 * @param n_paths Number of paths to simulate
 * @param n_steps Number of time steps per path
 * @param rng Random number generator (by reference)
 * @return Estimated call option price
 */
double monte_carlo_call(double S0, double K, double r, double sigma, double T,
                        int n_paths, int n_steps, std::mt19937& rng) {
    double sum_payoffs = 0.0;
    
    for (int i = 0; i < n_paths; ++i) {
        std::vector<double> S = simulate_gbm_path(S0, r, sigma, T, n_steps, rng);
        double ST = S[n_steps];  // Asset price at maturity
        double payoff = std::max(ST - K, 0.0);
        sum_payoffs += payoff;
    }
    
    double mean_payoff = sum_payoffs / n_paths;
    return std::exp(-r * T) * mean_payoff;
}

/**
 * Prices a European put option using Monte Carlo simulation.
 * 
 * @param S0 Initial asset price
 * @param K Strike price
 * @param r Risk-free rate
 * @param sigma Volatility
 * @param T Time to maturity
 * @param n_paths Number of paths to simulate
 * @param n_steps Number of time steps per path
 * @param rng Random number generator (by reference)
 * @return Estimated put option price
 */
double monte_carlo_put(double S0, double K, double r, double sigma, double T,
                       int n_paths, int n_steps, std::mt19937& rng) {
    double sum_payoffs = 0.0;
    
    for (int i = 0; i < n_paths; ++i) {
        std::vector<double> S = simulate_gbm_path(S0, r, sigma, T, n_steps, rng);
        double ST = S[n_steps];  // Asset price at maturity
        double payoff = std::max(K - ST, 0.0);
        sum_payoffs += payoff;
    }
    
    double mean_payoff = sum_payoffs / n_paths;
    return std::exp(-r * T) * mean_payoff;
}

/**
 * Black-Scholes formula for European call option.
 * Provides exact price for benchmarking the Monte Carlo estimates.
 * 
 * @param S0 Initial asset price
 * @param K Strike price
 * @param r Risk-free rate
 * @param sigma Volatility
 * @param T Time to maturity
 * @return Black-Scholes call option price
 */
double black_scholes_call(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S0 * std::erfc(-d1 / std::sqrt(2)) / 2 
         - K * std::exp(-r * T) * std::erfc(-d2 / std::sqrt(2)) / 2;
}

/**
 * Black-Scholes formula for European put option.
 * Provides exact price for benchmarking the Monte Carlo estimates.
 * 
 * @param S0 Initial asset price
 * @param K Strike price
 * @param r Risk-free rate
 * @param sigma Volatility
 * @param T Time to maturity
 * @return Black-Scholes put option price
 */
double black_scholes_put(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return K * std::exp(-r * T) * std::erfc(d2 / std::sqrt(2)) / 2 
         - S0 * std::erfc(d1 / std::sqrt(2)) / 2;
}

/**
 * Compares performance of Python and C++ implementations.
 * Displays execution times for different numbers of paths.
 */
void compare_performance(double S0, double K, double r, double sigma, double T,
                         const std::vector<int>& path_values) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "C++ PERFORMANCE BENCHMARK\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Paths      Price         Time (ms)\n";
    std::cout << std::string(70, '-') << "\n";
    
    std::random_device rd;
    std::mt19937 rng(rd());
    
    int n_steps = 1000;
    
    for (int n_paths : path_values) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        double price = monte_carlo_call(S0, K, r, sigma, T, n_paths, n_steps, rng);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        std::cout << n_paths << std::setw(12) << std::fixed << std::setprecision(4) 
                  << price << "    " << std::setprecision(3) << time_ms << "\n";
    }
    
    std::cout << std::string(70, '=') << "\n";
}

int main() {
    // Example parameters (matching Python examples)
    double S0 = 100.0;      // Initial stock price
    double K = 100.0;       // Strike price
    double r = 0.05;        // Risk-free rate (5%)
    double sigma = 0.20;    // Volatility (20%)
    double T = 1.0;         // Time to maturity (1 year)
    
    std::random_device rd;
    std::mt19937 rng(rd());
    
    // Display header
    std::cout << std::string(70, '=') << "\n";
    std::cout << "MONTE CARLO ASSET PRICING SIMULATOR (C++)" << "\n";
    std::cout << std::string(70, '=') << "\n";
    
    // Display parameters
    std::cout << "\nPARAMETERS\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << "S0 (Initial Price):       " << S0 << "\n";
    std::cout << "K (Strike Price):         " << K << "\n";
    std::cout << "r (Risk-Free Rate):       " << r * 100 << "%\n";
    std::cout << "sigma (Volatility):       " << sigma * 100 << "%\n";
    std::cout << "T (Time to Maturity):     " << T << " year\n";
    std::cout << std::string(70, '-') << "\n";
    
    // Calculate and display Black-Scholes benchmark
    double bs_call = black_scholes_call(S0, K, r, sigma, T);
    double bs_put = black_scholes_put(S0, K, r, sigma, T);
    
    std::cout << "\nBLACK-SCHOLES BENCHMARK\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Call Price:   " << std::fixed << std::setprecision(4) << bs_call << "\n";
    std::cout << "Put Price:    " << bs_put << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    // Monte Carlo pricing with different numbers of paths
    std::cout << "\nMONTE CARLO PRICING\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Paths      Call Price    Put Price    Call Error (%)   Time (ms)\n";
    std::cout << std::string(70, '-') << "\n";
    
    std::vector<int> path_values = {100, 500, 1000, 5000, 10000, 50000};
    int n_steps = 1000;
    
    for (int n_paths : path_values) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        double call_price = monte_carlo_call(S0, K, r, sigma, T, n_paths, n_steps, rng);
        double put_price = monte_carlo_put(S0, K, r, sigma, T, n_paths, n_steps, rng);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        double call_error = std::abs(call_price - bs_call) / bs_call * 100;
        
        std::cout << n_paths << std::setw(12) << std::fixed << std::setprecision(4) 
                  << call_price << std::setw(12) << put_price 
                  << std::setw(16) << call_error << "%" 
                  << std::setw(14) << std::setprecision(3) << time_ms << "\n";
    }
    
    std::cout << std::string(70, '-') << "\n";
    
    // Performance comparison across different path counts
    compare_performance(S0, K, r, sigma, T, {1000, 5000, 10000, 20000, 50000});
    
    // Put-Call Parity Check
    std::cout << "\nPUT-CALL PARITY CHECK\n";
    std::cout << std::string(70, '-') << "\n";
    
    int n_paths_check = 10000;
    double mc_call = monte_carlo_call(S0, K, r, sigma, T, n_paths_check, n_steps, rng);
    double mc_put = monte_carlo_put(S0, K, r, sigma, T, n_paths_check, n_steps, rng);
    double lhs = mc_call + K * std::exp(-r * T);
    double rhs = mc_put + S0;
    
    std::cout << "C + K*e^(-rT) = " << std::fixed << std::setprecision(4) << lhs << "\n";
    std::cout << "P + S0        = " << rhs << "\n";
    std::cout << "Difference    = " << std::abs(lhs - rhs) << "\n";
    std::cout << std::string(70, '=') << "\n";
    
    return 0;
}

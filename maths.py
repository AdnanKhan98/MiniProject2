import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Step 1: Load multiple CSV files
data_dir = './stocks/stocks/'  # Path to the directory containing CSV files
file_list = os.listdir(data_dir)

# Combine all stocks' data into one DataFrame
all_data = pd.DataFrame()

for file in file_list:
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_dir, file))
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[['Close']].rename(columns={'Close': file.replace('.csv', '')})
        all_data = pd.concat([all_data, df], axis=1)

# Drop rows with missing values
all_data.dropna(inplace=True)

# Step 2: Feature Engineering
# Calculate daily returns
returns = all_data.pct_change().dropna()

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Step 3: Apply K-Means Clustering (Machine Learning Approach)
# Use mean returns and volatility (standard deviation) for clustering
stock_features = pd.DataFrame({
    'mean_return': mean_returns,
    'volatility': returns.std()
})

# Normalize the features
stock_features_scaled = (stock_features - stock_features.mean()) / stock_features.std()

# K-Means clustering
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(stock_features_scaled)
stock_features['Cluster'] = clusters



# Step 4: Portfolio Optimization with Simulated Annealing (Machine Learning Approach)
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return returns, volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_returns, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (p_returns - risk_free_rate) / p_volatility

def check_sum(weights):
    return np.sum(weights) - 1

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': check_sum})
bounds = tuple((0, 1) for _ in range(len(mean_returns)))

# Initial guess for weights
initial_weights = np.array(len(mean_returns) * [1. / len(mean_returns)])

# Optimization
optimized_ml = minimize(
    negative_sharpe_ratio, 
    initial_weights, 
    args=(mean_returns, cov_matrix),
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

optimized_weights_ml = optimized_ml.x

# Display the optimized portfolio (Machine Learning Approach)
p_returns_ml, p_volatility_ml = portfolio_performance(optimized_weights_ml, mean_returns, cov_matrix)
sharpe_ratio_ml = (p_returns_ml - 0.01) / p_volatility_ml


# Print the optimal weights in terms of percentages, excluding zero allocations
print("Optimal Weights (Percentages):")
for i, col in enumerate(all_data.columns):
    if optimized_weights_ml[i] > 0:
        print(f"{col}: {optimized_weights_ml[i] * 100:.2f}%")

# Step 5: Portfolio Optimization with Markowitz Mean-Variance Model (Mathematical Approach)
def minimize_volatility(weights, mean_returns, cov_matrix, target_return):
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return volatility

def get_portfolio_return(weights, mean_returns):
    return np.sum(mean_returns * weights) * 252

# Target return for optimization
target_return = p_returns_ml

# Constraints and bounds
constraints_mvo = (
    {'type': 'eq', 'fun': check_sum},
    {'type': 'eq', 'fun': lambda w: get_portfolio_return(w, mean_returns) - target_return}
)

# Optimization
optimized_mvo = minimize(
    minimize_volatility, 
    initial_weights, 
    args=(mean_returns, cov_matrix, target_return),
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints_mvo
)

optimized_weights_mvo = optimized_mvo.x

# Display the optimized portfolio (Mathematical Approach)
p_returns_mvo, p_volatility_mvo = portfolio_performance(optimized_weights_mvo, mean_returns, cov_matrix)
sharpe_ratio_mvo = (p_returns_mvo - 0.01) / p_volatility_mvo

print("\nOptimized Portfolio (Mathematical Approach):")
print(f"Return: {p_returns_mvo}")
print(f"Volatility: {p_volatility_mvo}")
print(f"Sharpe Ratio: {sharpe_ratio_mvo}")

# Print the optimal weights in terms of percentages, excluding zero allocations
print("Optimal Weights (Percentages):")
for i, col in enumerate(all_data.columns):
    if optimized_weights_mvo[i] > 0:
        print(f"{col}: {optimized_weights_mvo[i] * 100:.2f}%")

# Optional: Plot the efficient frontier
def simulate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate=0.01):
    results = np.zeros((num_portfolios, 3))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        
        p_returns, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe_ratio = (p_returns - risk_free_rate) / p_volatility
        
        results[i, 0] = p_returns
        results[i, 1] = p_volatility
        results[i, 2] = sharpe_ratio
    return results

num_portfolios = 10000
results = simulate_random_portfolios(num_portfolios, mean_returns, cov_matrix)


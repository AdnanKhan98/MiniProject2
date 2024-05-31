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

# Step 3: Apply K-Means Clustering
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

# Display clustering result
print(stock_features)

# Step 4: Portfolio Optimization with Simulated Annealing
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
optimized = minimize(
    negative_sharpe_ratio, 
    initial_weights, 
    args=(mean_returns, cov_matrix),
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

optimized_weights = optimized.x

# Step 5: Display the optimized portfolio
p_returns, p_volatility = portfolio_performance(optimized_weights, mean_returns, cov_matrix)
sharpe_ratio = (p_returns - 0.01) / p_volatility

print(f"Optimized Portfolio Return: {p_returns}")
print(f"Optimized Portfolio Volatility: {p_volatility}")
print(f"Optimized Portfolio Sharpe Ratio: {sharpe_ratio}")

# Print the optimal weights in terms of percentages, excluding zero allocations
print("Optimal Weights (Percentages):")
for i, col in enumerate(all_data.columns):
    if optimized_weights[i] > 0:
        print(f"{col}: {optimized_weights[i] * 100:.2f}%")

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

plt.figure(figsize=(10, 6))
plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], cmap='YlGnBu', marker='o')
plt.title('Simulated Portfolio Optimization based on Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(p_volatility, p_returns, marker='*', color='r', s=500, label='Optimized Portfolio')
plt.legend(labelspacing=0.8)
plt.show()

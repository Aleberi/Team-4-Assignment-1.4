# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Loading the dataset
data = pd.read_csv('top_50_stocks_data_formatted.csv')

# Calculating annualized return
data['Daily Return'] = data.groupby('Ticker')['Adjusted Close'].pct_change()
annualized_return = data.groupby('Ticker')['Daily Return'].mean() * 252

# Calculating annualized volatility
annualized_volatility = data.groupby('Ticker')['Daily Return'].std() * (252 ** 0.5)

# Assuming there's risk-free rate of 0.01 (1%)
risk_free_rate = 0.01

# Calculating Sharpe ratio
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

# Create a dataframe to hold these metrics
metrics = pd.DataFrame({
    'Annualized Return': annualized_return,
    'Annualized Volatility': annualized_volatility,
    'Sharpe Ratio': sharpe_ratio
})
# # Display the first 5 rows of annualized return, annualized volatility, and Sharpe ratio
# metrics.head(5)

# Calculate and visualize the correlation matrix
correlation_matrix = metrics.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Stock Metrics')
plt.show()

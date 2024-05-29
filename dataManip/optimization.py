import numpy as np
from scipy.optimize import minimize

def stock_optimization(total_stocks_needed, current_stocks, n_days):
    """
    Optimize the stock set for each day to be near the average stock set.

    Args:
        total_stocks_needed (int): Total stocks needed for the week.
        current_stocks (list): List of current stocks for each day.
        n_days (int): Number of days in a week.

    Returns:
        np.ndarray: Optimized stock set for each day.
        np.ndarray: Amount of stocks actually used each day.
        np.ndarray: Amount of stocks left in the system for each day.
        np.ndarray: Amount of stock shortage for each day.
    """
    # Calculate the daily average stock set
    daily_avg_stock_set = total_stocks_needed / n_days

    # Define the objective function
    def objective(x):
        return np.sum((x - daily_avg_stock_set) ** 2)

    # Define the constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - total_stocks_needed}
    ]

    # Define the bounds
    bounds = [(0, None) for _ in range(n_days)]

    # Initial guess
    x0 = np.full(n_days, daily_avg_stock_set)

    # Solve the optimization problem
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_stock_set = result.x

    # Initialize arrays
    total_stocks_in_system = 0
    stocks_used = np.zeros(n_days)
    stocks_left = np.zeros(n_days)
    stock_shortages = np.zeros(n_days)

    # Compute stocks used, left, and shortages
    for i in range(n_days):
        total_stocks_in_system += current_stocks[i]
        stocks_used[i] = min(optimized_stock_set[i], total_stocks_in_system)
        stocks_left[i] = total_stocks_in_system - stocks_used[i]
        stock_shortages[i] = max(0, optimized_stock_set[i] - stocks_used[i])
        total_stocks_in_system -= stocks_used[i]

    return optimized_stock_set, stocks_used, stocks_left, stock_shortages

# Example usage
total_stocks_needed = 1200
current_stocks = [1000, 100, 100, 100, 100, 100, 100]
n_days = 7

optimized_stock_set, stocks_used, stocks_left, stock_shortages = stock_optimization(total_stocks_needed, current_stocks, n_days)

# Calculate the total stocks used
total_stocks_used = np.sum(stocks_used)

# Set print options for better readability
np.set_printoptions(suppress=True, precision=2)

print("Current Stock Set:")
print(current_stocks)
print("Optimized Stock Set:")
print(optimized_stock_set)
print("\nStocks Used:")
print(stocks_used)
print("\nStocks Left:")
print(stocks_left)
print("\nStock Shortages:")
print(stock_shortages)

# Comparison
print("\nTotal Stocks Needed: ", total_stocks_needed)
print("Total Stocks Used: ", total_stocks_used)
print("Difference: ", total_stocks_needed - total_stocks_used)

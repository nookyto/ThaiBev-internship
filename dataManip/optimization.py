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
    """
    # Calculate the daily average stock set
    daily_avg_stock_set = total_stocks_needed / n_days

    # Define the objective function
    def objective(x):
        return np.sum((x - daily_avg_stock_set) ** 2)

    # Define the constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - total_stocks_needed},
        {'type': 'ineq', 'fun': lambda x: x - np.array(current_stocks)}
    ]

    # Define the bounds
    bounds = [(0, None)] * n_days

    # Initial guess
    x0 = np.full(n_days, daily_avg_stock_set)

    # Solve the optimization problem
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

# Example usage
total_stocks_needed = 1200
current_stocks = [100, 150, 200, 150, 100, 150, 150]
n_days = 7

optimized_stock_set = stock_optimization(total_stocks_needed, current_stocks, n_days)
print(optimized_stock_set)
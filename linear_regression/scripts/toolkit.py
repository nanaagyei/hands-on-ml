"""
A toolkit to help with machine learning models and other machine learning tasks.
"""

import numpy as np
import matplotlib.pyplot as plt


def find_learning_rate(X, y, model, lr_min=1e-5, lr_max=10, n_steps=50):
    """
    Find a good learning rate by testing a logarithmic range.
    
    Strategy: Increase LR until cost starts rising or explodes.
    The best LR is typically ~1 order of magnitude below the explosion point.
    
    Parameters:
        X (ndarray): Training data, shape (n_samples, n_features)
        y (ndarray): Target values, shape (n_samples,)
        model (object): Model with learning_rate attribute and fit method
        lr_min (float): Minimum learning rate to try
        lr_max (float): Maximum learning rate to try
        n_steps (int): Number of learning rates to test
        
    Returns:
        best_lr (float): Recommended learning rate
        learning_rates (ndarray): All tested learning rates
        costs (list): Final cost for each learning rate
    """
    learning_rates = np.geomspace(lr_min, lr_max, n_steps)
    costs = []
    tested_lrs = []
    
    for lr in learning_rates:
        model.learning_rate = lr
        model.fit(X, y)
        
        final_cost = model.cost_history_[-1]
        
        # Check for numerical instability
        if np.isnan(final_cost) or np.isinf(final_cost):
            print(f"LR={lr:.6f}: Cost exploded (NaN/Inf), stopping search")
            break
        
        costs.append(final_cost)
        tested_lrs.append(lr)
        
        # Stop if cost is increasing (LR too high)
        if len(costs) > 1 and costs[-1] > costs[-2]:
            print(f"LR={lr:.6f}: Cost increasing, stopping search")
            break
    
    # Best LR: the one with lowest cost
    # Or more conservatively: one step before cost started rising
    best_idx = np.argmin(costs)
    best_lr = tested_lrs[best_idx]
    
    return best_lr, np.array(tested_lrs), costs


def plot_lr_finder(learning_rates, costs):
    """Visualize the learning rate finder results."""
    plt.figure(figsize=(10, 5))
    plt.semilogx(learning_rates, costs, 'b-o', linewidth=2, markersize=6)
    
    # Mark the best
    best_idx = np.argmin(costs)
    plt.scatter([learning_rates[best_idx]], [costs[best_idx]], 
                color='red', s=200, zorder=5, label=f'Best LR: {learning_rates[best_idx]:.4f}')
    
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Final Cost')
    plt.title('Learning Rate Finder')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
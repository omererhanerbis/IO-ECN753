import sys
import os
# Add the rust_1987 directory to sys.path
module_path = os.path.join(os.path.dirname(__file__), 'rust_1987')
sys.path.append(module_path)
import numpy as np

# Utility Functions

# maintenance cost
def c(x, linear_cost_param):
    """
    Calculates the maintenance cost based on mileage and a linear cost parameter.

    This function computes the maintenance cost as a product of the mileage (`x`), 
    scaled by a linear cost parameter, with a factor of 0.001 applied. 

    Args:
        x (float or np.array): The mileage or usage level of the product.
        linear_cost_param (float): The linear cost parameter that scales the maintenance cost.

    Returns:
        float or np.array: The calculated maintenance cost, scaled by 0.001.
    """
    
    maintenance_cost = 0.001 * linear_cost_param * x
    
    return maintenance_cost







# flow utility
def u(y, i, params):
    """
    Calculates the flow utility for a given state and decision.

    This function computes the flow utility for maintaining or replacing an asset based on its state and parameters.
    Following the structure from Rust (1987), the utility incorporates maintenance costs and replacement costs,
    where replacement occurs if `i = 1` and maintenance cost applies if `i = 0`.

    Args:
        y (float or np.array): The current state variable (e.g., mileage) of the asset.
        i (int): The decision indicator, where 0 represents "maintain" and 1 represents "replace."
        params (tuple): A tuple containing model parameters in the order `(beta, theta_11, theta_3, RC)`:
            - params[1] (float): `theta_11`, the linear cost parameter for maintenance cost.
            - params[3] (float): `RC`, the replacement cost.

    Returns:
        float or np.array: The calculated flow utility for the specified state and decision.
    """
    
    RC, theta_11 = params[3], params[1]
    flow_utility = -c(y * (1 - i), theta_11) - i * RC
    return flow_utility








#  flow utility matrix
def find_u(grid, params, state_size):
    """
    Constructs the flow utility matrix for all states and decisions.

    This function calculates the flow utility for both the "maintain" and "replace" decisions across all states.
    It creates a 3D array `U` where the third dimension represents the two possible decisions:
    - `U[:,:,0]` represents the utility for maintaining (i=0).
    - `U[:,:,1]` represents the utility for replacing (i=1).

    Args:
        grid (tuple of np.array): A meshgrid tuple representing the state space. Typically, `grid[0]` contains
                                  the values for the state variable (e.g., mileage).
        params (tuple): Model parameters, following the structure `(beta, theta_11, theta_3, RC)`:
            - params[1] (float): `theta_11`, the linear cost parameter for maintenance.
            - params[3] (float): `RC`, the replacement cost.
        state_size (int, optional): The number of discrete states, defaulting to `N`.
        tol (float, optional): Tolerance level for convergence, although it is not used in this function.

    Returns:
        np.array: A 3D array of shape `(state_size, state_size, 2)` where each entry represents the flow utility
                  for a given state and decision.
    """
    
    U = np.zeros((state_size, state_size, 2), np.float64)

    y = grid[0]
    U[:, :, 0] = u(y, 0, params)  # Utility for maintaining
    U[:, :, 1] = u(y, 1, params)  # Utility for replacing
    
    return U
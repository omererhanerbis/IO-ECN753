import sys
import os
# Add the rust_1987 directory to sys.path
module_path = os.path.join(os.path.dirname(__file__), 'rust_1987')
sys.path.append(module_path)
import numpy as np
from utility_functions import c, u, find_u
from probability_functions import g, find_P


# Value Function related Functions

# Transition matrix update
def update(EV_val, U_mat, P_mat, params):
    """
    Updates the value function using Bellman’s equation for dynamic programming.

    This function computes the expected future utility by combining current utilities with the expected 
    continuation values. It calculates the log-sum of utilities for each state, which represents the 
    choice-specific value function. Transition probabilities are then used to update the expected values.

    Args:
        EV_val (np.array): A `(state_size, 2)` array representing the expected value function for each 
                           state and decision (maintain or replace).
        U_mat (np.array): A `(state_size, state_size, 2)` array representing the flow utility matrix,
                          where the third dimension represents the two decision choices.
        P_mat (np.array): A `(state_size, state_size, 2)` array of transition probabilities for each 
                          decision choice.
        params (tuple): Model parameters structured as `(beta, theta_11, theta_3, RC)`:
            - params[0] (float): `beta`, the discount factor for future utility.
        state_size (int, optional): The number of discrete states, defining the dimensions of matrices.
        tol (float, optional): Tolerance level for convergence, though not directly used in this function.

    Returns:
        np.array: A `(state_size, 2)` array where each row contains the updated expected values for each 
                  decision choice (maintain or replace) across all states.
    """
    
    U1 = U_mat[1, :, :]
    lifetime_utility = U1 + params[0] * EV_val  # Uses beta (params[0]) as the discount factor
    cons = lifetime_utility[0, 1]
    
    # Calculate the log-sum-exp for numerical stability
    log_sum_utils = np.log(np.sum(np.exp(lifetime_utility - cons), axis=1)) + cons
    
    # Transition matrices for maintain and replace decisions
    P0 = P_mat[:, :, 0]
    P1 = P_mat[:, :, 1]
    t1 = P0 @ log_sum_utils
    t2 = P1 @ log_sum_utils
    
    return np.column_stack((t1, t2))








# Value function iteration for expected value calculation

def value_function_iteration(params, state_size, tol):
    """
    Performs value function iteration to calculate the expected value function for each state.

    This function iteratively updates the expected value function using Bellman’s equation until convergence
    is reached. It initializes the expected values with zeros, then applies the update function iteratively,
    checking for convergence based on the tolerance level.

    Args:
        params (tuple): Model parameters structured as `(beta, theta_11, theta_3, RC)`:
            - params[0] (float): `beta`, the discount factor for future utility.
            - params[1] (float): `theta_11`, the linear cost parameter for maintenance cost.
            - params[2] (tuple): `theta_3`, transition probabilities for different state changes.
            - params[3] (float): `RC`, the replacement cost.
        state_size (int, optional): The number of discrete states, defining the dimensions of matrices.
        tol (float, optional): Tolerance level for convergence.

    Returns:
        tuple: A tuple containing:
            - np.array: A `(state_size, 2)` array representing the final expected value function for 
              each state and decision (maintain or replace).
            - float: The final error after convergence.
            - np.array: The flow utility matrix `U`, used in subsequent calculations.
    """
    
    state = np.arange(state_size)
    space = np.meshgrid(state, state)
    U = find_u(space, params, state_size)
    P = find_P(space, params[2], state_size)
    
    # Initialize expected value function and error
    EV = np.zeros((state_size, 2))
    error = 1
    
    # Iteratively update the expected value function until convergence
    while error > tol:
        EV_prime = update(EV, U, P, params)
        error = np.linalg.norm(EV_prime - EV)
        EV = EV_prime
    
    return EV, error, U
# This code is going to try to estimate Rust 1987 paper.
# We are required to estimate RC and theta_11

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from time import time

# Prelimimary Step: We need calibrated beta, which is given to be beta = 0.9999, and number of states is 90
beta        =   0.9999
N           =   90

# First stage is already done and the estimates are provided
# Use the given transition parameter estimates
theta_30    =   0.3919
theta_31    =   0.5953
# Initial values for estimates (RC, theta_11)
theta_11    =   1
RC          =   1
estimates = (theta_11, RC)
# Create state transition parameter estimates tuple
theta_3     =   (theta_30, theta_31)
# Create the whole parameter vector, (beta, theta_1, theta_3), beware the order
theta       =   (beta, theta_11, theta_3, RC)



# Set the tolerance level
tolerance   =   10**-4
Comp_parameters=(N,tolerance)

# Read data
data        =   pd.read_csv("rust_1987/group_4.csv")



















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
def find_u(grid, params, state_size=N, tol=tolerance):
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

















# Probability Matrix Functions


# State transition matrix calculator through parameterized transition function g()
def g(delta_x, transition_params, state_size):
    """
    Calculates the state transition matrix for given state changes and transition parameters.

    This function constructs the state transition matrix `P`, where each row represents the transition 
    probabilities from a given state to other states based on changes in state (`delta_x`). The transition 
    probabilities are assigned based on the parameterized values in `transition_params`, with any remaining 
    probability assigned to a third category.

    Args:
        delta_x (np.array): A 2D array representing the changes in states between each pair of states.
                            The values in `delta_x` should indicate state transitions (e.g., 0, 1, 2).
        transition_params (tuple): A tuple containing transition parameters:
            - transition_params[0] (float): Probability for `delta_x == 0`.
            - transition_params[1] (float): Probability for `delta_x == 1`.
        state_size (int): The number of discrete states, defining the dimensions of the transition matrix.

    Returns:
        np.array: A `(state_size, state_size)` transition matrix where each row sums to 1, representing the 
                  probabilities of transitioning from each state to other states.
    """
    
    P = np.zeros((state_size, state_size))

    P[delta_x == 0] = transition_params[0]
    P[delta_x == 1] = transition_params[1]
    P[delta_x == 2] = 1 - sum(transition_params)
    P1p = np.sum(P, axis=1).reshape((state_size, 1))
    P = P / P1p
    
    return P






def find_P(grid, transition_params=theta_3, state_size=N, tol=tolerance):
    """
    Constructs the probability transition matrix for both decision choices across all states.

    This function computes a 3D probability transition matrix `P`, where:
    - `P[:, :, 0]` represents the transition probabilities when the decision is to maintain (no change in state).
    - `P[:, :, 1]` represents the transition probabilities when the decision is to replace (state resets).

    Args:
        grid (tuple of np.array): A meshgrid tuple representing the state space. Typically, `grid[0]` 
                                  contains the values for the state variable (e.g., mileage), and 
                                  `grid[1]` represents other relevant state dimensions.
        transition_params (tuple, optional): Transition parameters defining probabilities of state changes:
            - transition_params[0] (float): Probability for no change (`delta_x == 0`).
            - transition_params[1] (float): Probability for a state change (`delta_x == 1`).
        state_size (int, optional): The number of discrete states, defining the dimensions of the transition matrix.
        tol (float, optional): Tolerance level for convergence, though not used directly in this function.

    Returns:
        np.array: A 3D array of shape `(state_size, state_size, 2)` where:
                  - `P[:, :, 0]` represents transition probabilities for maintaining,
                  - `P[:, :, 1]` represents transition probabilities for replacing.
    """
    
    x = grid[1]
    y = grid[0]
    P = np.zeros((state_size, state_size, 2))
    P[:, :, 0] = g(y - x, transition_params, state_size)  # Transition for maintaining
    P[:, :, 1] = g(y, transition_params, state_size)      # Transition for replacing
    
    return P
























# Value Function related Functions


# Transition matrix update
def update(EV_val, U_mat, P_mat, params=theta, state_size=N, tol=tolerance):
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

def value_function_iteration(params, state_size=N, tol=tolerance):
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
    U = find_u(space, params=params)
    P = find_P(space, transition_params=params[2])
    
    # Initialize expected value function and error
    EV = np.zeros((state_size, 2))
    error = 1
    
    # Iteratively update the expected value function until convergence
    while error > tol:
        EV_prime = update(EV, U, P, params=params)
        error = np.linalg.norm(EV_prime - EV)
        EV = EV_prime
    
    return EV, error, U







# Log-likelihood function
def log_likelihood(EV_val, U_mat, params, df=data, state_size=N, tol=tolerance):
    """
    Calculates the log-likelihood of observed choices given the expected value function.

    This function computes the log-likelihood based on the observed states and decisions in the data, 
    comparing them to the predicted probabilities from the model. It uses the choice-specific utilities 
    to calculate the probability of each observed choice, then sums the log-probabilities.

    Args:
        EV_val (np.array): A `(state_size, 2)` array representing the expected value function for each 
                           state and decision (maintain or replace).
        U_mat (np.array): A `(state_size, state_size, 2)` array representing the flow utility matrix,
                          where the third dimension represents the two decision choices.
        params (tuple): Model parameters structured as `(beta, theta_11, theta_3, RC)`:
            - params[0] (float): `beta`, the discount factor for future utility.
        df (pd.DataFrame, optional): The dataset containing observed states and decisions. The columns 
                                     should include `state` and `decision`.
        state_size (int, optional): The number of discrete states, defining the dimensions of matrices.
        tol (float, optional): Tolerance level for convergence, though not used directly in this function.

    Returns:
        float: The log-likelihood value for the observed data given the model predictions.
    """
    
    # Extract observed states and decisions from the data
    x = np.array(df["state"])
    i = np.array(df["decision"])
    
    # Calculate choice-specific value with EV and discount factor beta (params[0])
    P1 = U_mat[0, :, :] + params[0] * EV_val
    P2 = np.exp(P1 - P1.max())  # Applying log-sum-exp trick for numerical stability
    Pr_bot = np.sum(P2, axis=1).reshape((state_size, 1))
    Pr = P2 / Pr_bot
    
    # Calculate log-likelihood by summing log of predicted probabilities for observed choices
    return np.sum(np.log(Pr[x, i]))





def objective_function(estimates, beta_par=beta, theta_3_par=theta_3, state_size=N, tol=tolerance):
    """
    Computes the negative log-likelihood for given parameter estimates.

    This function evaluates the model's fit to the observed data by calculating the log-likelihood 
    and then returning its negative, which is suitable for minimization in optimization routines.
    The expected value function is first calculated using value function iteration, and then the 
    log-likelihood is computed based on observed choices.

    Args:
        estimates (tuple): A tuple containing the parameters to be estimated:
            - estimates[0] (float): `RC`, the replacement cost.
            - estimates[1] (float): `theta_11`, the linear cost parameter for maintenance.
        beta_par (float, optional): The discount factor, `beta`, for future utility. Defaults to the predefined `beta`.
        theta_3_par (tuple, optional): Transition parameters, `theta_3`, for state changes. Defaults to the predefined `theta_3`.
        state_size (int, optional): The number of discrete states, defining the dimensions of matrices.
        tol (float, optional): Tolerance level for convergence in value function iteration.

    Returns:
        float: The negative log-likelihood value for the observed data given the model predictions.
    """
    
    RC, theta_11 = estimates
    theta_hat = (beta_par, theta_11, theta_3_par, RC)
    EV, error, U = value_function_iteration(theta_hat, state_size, tol)
    log_likelihood_value = log_likelihood(EV, U, theta_hat, data, state_size, tol)
    
    return -log_likelihood_value







# Start time
start_time = time()

search = minimize(objective_function, estimates, method='L-BFGS-B' )

# End time
end_time = time()
# Calculate and print runtime
runtime = end_time - start_time
print(f"Optimization completed in {runtime:.2f} seconds")


RC2, theta_112=(search.x[0],search.x[1])
Theta2=(beta,theta_112,theta_3,RC2)
EV2,err2,U2=value_function_iteration(Theta2)
loglike2=log_likelihood(EV2,U2,Theta2)
print(Theta2)
print(loglike2)
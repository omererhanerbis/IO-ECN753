# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 18:16:13 2024

@author: Andres w/ help from Erhan
"""
####################################################################################
# IMPORTANT: Part (a) takes 46 min to run because it compares multiple optimizers
####################################################################################


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from time import time


# Import data
data = pd.read_csv('Data/group_4.csv')

##############################################################################
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Part a #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

"""
I tried to code this from scratch by myself, but I was getting the wrong results. The results I initially found were:
    RC = 7.4
    theta_11 = 3.5

I tried many things to fix the code, but I couldn't manage to fix it.

I want to thank Erhan for sharing his code. This way I'm getting the right values

I'm comparing multiple optimizers to check for stability, so the code may take a while to run.
"""
# Preliminary setup
beta = 0.9999
N = 90
theta_30 = 0.3919
theta_31 = 0.5953
theta_11 = 1                                                                    # Initial guess for cost parameter
RC = 1                                                                          # Initial guess for replacement cost
estimates = (theta_11, RC)
theta_3 = (theta_30, theta_31)
theta = (beta, theta_11, theta_3, RC)
tolerance = 1e-4


# Maintenance cost function
def cost_function(x, linear_cost_param):
    maintenance_cost = 0.001 * linear_cost_param * x
    return maintenance_cost

# Flow utility function
def u(y, i, params):
    RC, theta_11 = params[3], params[1]
    if i == 0:
        flow_utility = -cost_function(y, theta_11)
    else:
        flow_utility = -cost_function(0, theta_11) - RC
    return flow_utility

# Flow utility matrix
def find_u(grid, params, state_size=N, tol=tolerance):
    U = np.zeros((state_size, state_size, 2), np.float64)
    y = grid[0]
    U[:, :, 0] = u(y, 0, params)  # Utility for maintaining
    U[:, :, 1] = u(y, 1, params)  # Utility for replacing
    return U

# State transition matrix calculation
def g(delta_x, transition_params, state_size):
    P = np.zeros((state_size, state_size))
    P[delta_x == 0] = transition_params[0]
    P[delta_x == 1] = transition_params[1]
    P[delta_x == 2] = 1 - sum(transition_params)
    P1p = np.sum(P, axis=1).reshape((state_size, 1))
    P = P / P1p
    return P

# Transition probability matrix for both maintain and replace decisions
def find_P(grid, transition_params=theta_3, state_size=N, tol=tolerance):
    x = grid[1]
    y = grid[0]
    P = np.zeros((state_size, state_size, 2))
    P[:, :, 0] = g(y - x, transition_params, state_size)  # Transition for maintaining
    P[:, :, 1] = g(y, transition_params, state_size)      # Transition for replacing
    return P

# Transition matrix update using Bellman equation
def update(EV_val, U_mat, P_mat, params=theta, state_size=N, tol=tolerance):
    U1 = U_mat[1, :, :]
    lifetime_utility = U1 + params[0] * EV_val  # Uses beta as discount factor
    cons = lifetime_utility[0, 1]
    
    # Log-sum-exp for stability
    log_sum_utils = np.log(np.sum(np.exp(lifetime_utility - cons), axis=1)) + cons
    P0 = P_mat[:, :, 0]
    P1 = P_mat[:, :, 1]
    t1 = P0 @ log_sum_utils
    t2 = P1 @ log_sum_utils
    return np.column_stack((t1, t2))

# Value function iteration with expected value calculation
def value_function_iteration(params, state_size=N, tol=tolerance):
    state = np.arange(state_size)
    space = np.meshgrid(state, state)
    U = find_u(space, params=params)
    P = find_P(space, transition_params=params[2])
    
    # Initialize expected value function and error
    EV = np.zeros((state_size, 2))
    error = 1
    
    # Iteratively update until convergence
    while error > tol:
        EV_prime = update(EV, U, P, params=params)
        error = np.linalg.norm(EV_prime - EV)
        EV = EV_prime
    
    return EV, error, U

# Log-likelihood function
def log_likelihood(EV_val, U_mat, params, df=data, state_size=N, tol=tolerance):
    x = np.array(df["state"])
    i = np.array(df["decision"])
    
    # Choice-specific value with EV and beta
    P1 = U_mat[0, :, :] + params[0] * EV_val
    P2 = np.exp(P1 - P1.max())
    Pr_bot = np.sum(P2, axis=1).reshape((state_size, 1))
    Pr = P2 / Pr_bot
    
    # Log-likelihood
    return np.sum(np.log(Pr[x, i]))

# Objective function for optimization
def objective_function(estimates, beta_par=beta, theta_3_par=theta_3, state_size=N, tol=tolerance):
    RC, theta_11 = estimates
    theta_hat = (beta_par, theta_11, theta_3_par, RC)
    EV, error, U = value_function_iteration(theta_hat, state_size, tol)
    log_likelihood_value = log_likelihood(EV, U, theta_hat, data, state_size, tol)
    return -log_likelihood_value

# Optimization procedure
start_time = time()
results = {}
for method in ['L-BFGS-B', 'Nelder-Mead', 'Powell', 'TNC', 'COBYLA', 'SLSQP']:
    print(f"Testing method: {method}")
    result = minimize(objective_function, estimates, method=method, options={'disp': True})
    results[method] = result
    print(f"Method: {method}, Result: {result.fun}, Parameters: {result.x}")
    print("-" * 30)
end_time = time()
runtime = (end_time - start_time)/60
print(f"Optimization completed in {runtime:.2f} minutes")




##############################################################################################

"""
# Model specifications
beta = 0.9999
num_states = 90  # Dimension of state space
initial_params = [10, 1.0]  # Initial guesses for [RC, theta_11]

# Transition parameters (given in Rust 1987)
theta_30 = 0.3919
theta_31 = 0.5953
transition_params = (theta_30, theta_31)

# Cost function (unchanged)
def cost_function(x, theta_11):
    return 0.001 * theta_11 * x

# Parameterized transition matrix function `g`
def g(delta_x, transition_params, state_size):
    """
  #  Calculates the state transition matrix for given state changes and transition parameters.

 #   This function constructs the state transition matrix `P`, where each row represents the transition 
  #  probabilities from a given state to other states based on changes in state (`delta_x`). The transition 
   # probabilities are assigned based on the parameterized values in `transition_params`.
    """
    P = np.zeros((state_size, state_size))

    # Set probabilities based on delta_x (state changes)
    P[delta_x == 0] = transition_params[0]  # Probability of staying in the same state
    P[delta_x == 1] = transition_params[1]  # Probability of moving to the next state
    P[delta_x == 2] = 1 - sum(transition_params)  # Probability of other transitions

    # Normalize rows to ensure they sum to 1
    P1p = np.sum(P, axis=1).reshape((state_size, 1))
    P = P / P1p
    
    return P

# Create a full transition matrix `find_P` based on g()
def find_P(state_size, transition_params=transition_params):
    """
  #  Constructs a probability transition matrix for maintain and replace decisions.
    """
    state_grid = np.arange(state_size)
    grid = np.meshgrid(state_grid, state_grid)

    x = grid[1]
    y = grid[0]
    P = np.zeros((state_size, state_size, 2))

    # Maintain decision
    P[:, :, 0] = g(y - x, transition_params, state_size)  # Transition matrix for maintaining

    # Replace decision
    P[:, :, 1] = g(y, transition_params, state_size)      # Transition matrix for replacing (reset state)

    return P

# Bellman equation solution using transition matrix `P`
def compute_value_function(params, beta, P):
    RC, theta_11 = params
    V = np.zeros(num_states)  # Initialize value function
    tolerance = 1e-4  # Convergence tolerance

    for _ in range(500):  # Limit number of iterations
        V_old = V.copy()
        
        # Calculate replacement and continuation values for each state
        value_replace = -RC * np.ones(num_states)  # Replacement cost for all states
        cost = cost_function(np.arange(num_states), theta_11)  # Maintenance cost for each state
        expected_value_continue = P[:, :, 0] @ V  # Expected continuation values for maintaining
        value_continue = -cost + beta * expected_value_continue  # Value of continuing in each state
        
        # Update value function by taking the max of replace and continue
        V = np.maximum(value_replace, value_continue)

        # Check for convergence
        if np.max(np.abs(V - V_old)) < tolerance:
            break
    return V

# Log-likelihood function (unchanged)
def log_likelihood(params, data, beta, P):
    RC, theta_11 = params
    V = compute_value_function(params, beta, P)
    log_likelihood_value = 0

    for _, row in data.iterrows():
        x = int(row['state'])
        decision = row['decision']

        cost = cost_function(x, theta_11)
        value_replace = -RC
        expected_value_continue = P[x, :, 0] @ V  # Expected continuation value for state x
        value_continue = -cost + beta * expected_value_continue

        # Calculate choice probabilities with normalization to avoid overflow
        c = max(value_replace, value_continue)
        prob_replace = np.exp(value_replace - c) / (np.exp(value_replace - c) + np.exp(value_continue - c))
        prob_continue = 1 - prob_replace

        # Log-likelihood contribution
        if decision == 1:
            log_likelihood_value += np.log(prob_replace + 1e-10)  # Avoid log(0)
        else:
            log_likelihood_value += np.log(prob_continue + 1e-10)

    return -log_likelihood_value  # Negative for minimization

# Optimize parameters using maximum likelihood
P = find_P(num_states, transition_params)
result = minimize(log_likelihood, initial_params, args=(data, beta, P), method='L-BFGS-B')
estimated_params = result.x
print("Estimated RC and theta_11:", estimated_params)

"""

##############################################################################
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Part b #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


# Estimate the Conditional Choice Probabilities (CCPs)
# Logistic regression model for replacement probability as a function of mileage
logit_model = LogisticRegression()
X = data[['mileage']].values  # Mileage is the explanatory variable
y = data['decision'].values  # Replacement decision (1 if replaced, 0 otherwise)
logit_model.fit(X, y)
data['ccp_replace'] = logit_model.predict_proba(X)[:, 1]  # Predicted CCP for replacement

# Model parameters to be estimated
initial_params = [2, 2]  # Initial guesses for parameters [theta, c_replace]

# Discount factor
beta = 0.9999

# Define the log-likelihood function using the CCP estimates
def log_likelihood_ccp(params, data):
    theta, c_replace = params
    log_likelihood_value = 0
    
    # Iterate over each bus
    for _, row in data.iterrows():
        mileage = row['mileage']
        replacement_decision = row['decision']
        ccp_replace = row['ccp_replace']
        
        # Calculate costs and values based on mileage
        maint_cost = 0.001 * theta * mileage
        replace_cost = c_replace
        
        # Approximate value function using CCP
        prob_replace = ccp_replace
        prob_continue = 1 - prob_replace
        
        # Add small constant to avoid log(0)
        prob_replace = max(prob_replace, 1e-10)
        prob_continue = max(prob_continue, 1e-10)
        
        # Calculate log-likelihood contribution based on the observed decision
        if replacement_decision == 1:
            log_likelihood_value += np.log(prob_replace)
        else:
            log_likelihood_value += np.log(prob_continue)
    
    # Debugging output
    print(f"Params: theta={theta}, c_replace={c_replace}, Log-likelihood={-log_likelihood_value}")
    
    return -log_likelihood_value  # Negative log-likelihood for minimization

# Estimate the parameters using maximum likelihood with CCPs
result = minimize(log_likelihood_ccp, initial_params, args=(data,), method='Powell', options={'disp': True})
estimated_params = result.x
print("Estimated parameters using CCP method:", estimated_params)
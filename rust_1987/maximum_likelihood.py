import sys
import os
# Add the rust_1987 directory to sys.path
module_path = os.path.join(os.path.dirname(__file__), 'rust_1987')
sys.path.append(module_path)
import numpy as np
from value_iteration import update, value_function_iteration


# Log-likelihood function

def log_likelihood(EV_val, U_mat, params, df, state_size):
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





def objective_function(estimates, df, beta_par, theta_3_par, state_size, tol):
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
    #print(f"Evaluating objective_function with RC={RC}, theta_11={theta_11}")
    theta_hat = (beta_par, theta_11, theta_3_par, RC)
    EV, error, U = value_function_iteration(theta_hat, state_size, tol)
    #print(f"Value function iteration completed with error={error}")
    log_likelihood_value = log_likelihood(EV, U, theta_hat, df, state_size)
    #print(f"Log-likelihood value: {log_likelihood_value}")
    return -log_likelihood_value

import os 
import numpy as np
import pandas as pd
from scipy.optimize import minimize

θ_30 = 0.3919
θ_31 = 0.5953
def cost(theta,x):
    return 0.001*theta*x
def state_space(length):
    lst = []
    for i in range(length):
        lst.append(i)
    return lst
def cal_utility(theta1, rc,states):
    u0 = list(cost(theta1,np.array(states)))
    u1 = rc
    return [u0,u1]

def transition(S,theta_3_0, theta_3_1):
    """
    construct transition matrix for the Rust model with empirical probabilities
    of the three possible state transitions (delta = 0, 5000, or 10000)

    inputs:
        S, number of possible states
        
    output:
        P, S x S matrix with entries [i, j] containing the probability of 
            transitioning to state j from state i
    """

    theta_3_2 = 1 - theta_3_0 - theta_3_1
    P = np.zeros((S, S))

    # fill off-diagonals of transition matrix
    P[np.arange(0, S), np.arange(0, S)] = theta_3_0
    P[np.arange(0, S-1), np.arange(1, S)] = theta_3_1
    P[np.arange(0, S-2), np.arange(2, S)] = theta_3_2

    # adjust absorbing states to sum to 1
    P[S-1, S-1] = 1
    P[S-2, S-1] = 1 - theta_3_0
    return P

def compute_EV(states, P, theta1,rc, beta, tol):
    """
    inputs:
        states, state space vector
        P, S x S transition matrix
        theta, vector of parameters associated with the utility/cost
            functions
        beta, discount factor
        tol, tolerance at which to stop the iteration

    output:
        EV, length S vector encoding the expected value function for each
            state in x at the given parameters theta
    """

    def B(EV):
        """
        Bellman operator to iterate on

        input:
            EV, length S vector encoding the expected value function

        output:
            B, length S vector encoding the value B(EV)
        """
        
        # utility and value from continuing (without the error term)
        U = cal_utility(theta1, rc,states)
        u_0 = np.array(U[0])
        v_0 = u_0 + beta * EV
        
        # utility and value from replacing (without the error term)
        u_1 = np.array(U[1])
        v_1 = u_1 + beta * EV[0]

        # subtract and re-add EV to avoid overflow issues
        G = np.exp(v_0 - EV) + np.exp(v_1 - EV) # social surplus function
        B = P @ (np.log(G) + EV) # Bellman operator

        return B

    EV_old = EV = np.zeros(P.shape[0]) # initial EV guess
    error = 1

    while error > tol:
        EV_old = EV
        EV = B(EV_old)
        error = np.max(np.abs(EV - EV_old))

    return EV

def choice_prob(states,theta1,rc,beta,EV):
    """
    compute dynamic logit choice probabilities conditional on 
    state variables

    inputs:
        states, length S vector of state variables
        theta1, rc, parameters in the utility function
        beta, discount factor
        EV, length S vector of expected values
    
    output:
        Pr, S x 2 array whose entries [i, j] are the probabilities
            of choosing actions j = 0, 1 conditional on state i
    """
    U = cal_utility(theta1, rc,states)
    u_0 = np.array(U[0])
    v_0 = u_0 + beta * EV

    # utility and value from replacing (without the error term)
    u_1 = np.array(U[1])
    v_1 = u_1 + beta * EV[0]

    # dynamic logit choice probabilities
    # subract max(EV) from exponents to avoid overflow
    Pr_0 = np.exp(v_0 - max(EV)) / (np.exp(v_0 - max(EV)) + np.exp(v_1 - max(EV)))
    Pr_1 = 1 - Pr_0

    Pr = np.transpose(np.array((Pr_0, Pr_1)))
    return Pr

def objective(params, states, P, beta, tol, X, I):
    """
    compute partial log-likelihood objective function

    inputs:
        params, vector of parameters associated with the utility
        states, length S vector of state variables
        P, S x S transition matrix
        beta, discount factor
        tol, tolerance at which to stop the EV iteration
        X, vector of observed states in data
        I, vector of observed decisions in data

    output:
        LL, partial log-likelihood evaluated at theta
    """
    theta1,rc = params
    # solve for EV and conditional choice probabilities at theta
    EV = compute_EV(states,P,theta1,rc,beta,tol)
    Pr = choice_prob(states,theta1,rc,beta,EV)

    # compute partial likelihood function
    LL = 0
    for x_t, i_t in zip(X, I):
        LL += np.log(Pr[x_t, i_t])

    return -LL
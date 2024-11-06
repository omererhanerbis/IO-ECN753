import sys
import os
# Add the rust_1987 directory to sys.path
module_path = os.path.join(os.path.dirname(__file__), 'rust_1987')
sys.path.append(module_path)
import numpy as np

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






def find_P(grid, transition_params, state_size):
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
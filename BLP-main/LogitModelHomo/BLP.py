import numpy as np 
import pandas as pd
import os
from scipy.optimize import minimize
from processing_data import df
from processing_data import agent_data
import matplotlib.pyplot as plt
def initialize_random_delta(num_products):
    # Initialize random shares for each product in a market
    # We will normalize these random values to sum to 1 (as shares should sum to 1)
    random_shares = np.random.rand(num_products)
    return random_shares / random_shares.sum()

def compute_utilities(beta_vector, alpha_nu, product_data, agent_data):
    """
    Compute utility matrix for a market using vectorized operations.
    
    Parameters:
    - beta_vector: The [beta_0, beta_sugar, beta_price] vector
    - alpha_nu: The scaling factor for noise
    - product_data: DataFrame with product info (sugar, price) for a market
    - agent_data: DataFrame with agent info (income, noise) for a market
    
    Returns:
    - Utility matrix where rows are products and columns are agents
    """
    # Extract product characteristics: [1, sugar, price] for each product
    product_features = np.column_stack((np.ones(len(product_data)), 
                                        product_data['sugar'].values, 
                                        product_data['prices'].values))
    
    # Extract agent characteristics: income and noise for each agent
    incomes = agent_data['income'].values.reshape(1, -1)  # Shape (1, num_agents)
    noises = agent_data['nodes0'].values.reshape(1, -1)    # Shape (1, num_agents)
    
    # Compute the agent-specific terms: (beta_vector * income_i + alpha_nu * noise_i)
    agent_effects = beta_vector[:, np.newaxis] * incomes + alpha_nu * noises
    
    # Vectorized computation of the utility matrix
    # Utility matrix = product_features @ agent_effects
    utility_matrix = product_features @ agent_effects
    
    return utility_matrix
def calculate_market_shares(delta_market, utility_matrices, agent_data, product_data):
    """
    Calculate estimated market shares for each product in each market.
    
    Parameters:
    - delta_market: Dictionary containing delta (random market shares) for each market
    - utility_matrices: Dictionary of utility matrices for each market (rows: products, columns: agents)
    - agent_data: DataFrame containing agent information (including weights) for each market
    - product_data: DataFrame containing product data (used to get the list of products per market)
    
    Returns:
    - market_shares: Dictionary containing estimated market shares for each market and product
    """
    market_shares = {}
    
    # Loop over each market
    for market in delta_market.keys():
        # Get the delta values (random market shares) for this market
        delta_t = delta_market[market]
        
        # Get the utility matrix for this market
        utility_matrix = utility_matrices[market]  # Shape: (num_products, num_agents)
        
        # Get the agent data (weights) for this market
        market_agents = agent_data[agent_data['market_ids'] == market]
        weights = market_agents['weights'].values  # Shape: (num_agents,)
        
        # Number of products and agents in this market
        num_products = utility_matrix.shape[0]
        num_agents = utility_matrix.shape[1]
        
        # Initialize array to store estimated market shares for each product in this market
        product_shares = np.zeros(num_products)
        
        # Loop over each product
        for j in range(num_products):
            # Numerator: w_it * exp(delta_jt + u_ijt)
            numerator =  np.exp(delta_t[j] + utility_matrix[j, :])
            
            # Denominator: 1 + sum_k exp(delta_kt + u_ikt) for each agent i
            exp_term = np.exp(delta_t[:, np.newaxis] + utility_matrix)  # Shape: (num_products, num_agents)
            denominator = 1 + np.sum(exp_term, axis=0)  # Sum over products for each agent
            
            # Compute market share for product j in market t
            product_shares[j] = np.mean(numerator / denominator)
        
        # Store the market shares for this market
        market_shares[market] = product_shares
    
    return market_shares

def update_delta(delta_market, product_data, estimated_market_shares):
    """
    Update the delta vector for each market based on the observed and estimated market shares.
    
    Parameters:
    - delta_market: Dictionary containing current delta (random market shares) for each market.
    - product_data: DataFrame containing product data (used to get observed shares).
    - estimated_market_shares: Dictionary containing estimated market shares for each market.
    
    Returns:
    - updated_delta_market: Dictionary containing updated delta for each market.
    """
    updated_delta_market = {}
    
    # Loop through each market
    for market in delta_market.keys():
        # Get current delta for this market
        delta_t = delta_market[market]
        
        # Get observed market shares (s_t) from the product data for this market
        market_product_data = product_data[product_data['market_ids'] == market]
        observed_shares = market_product_data['shares'].values
        
        # Get estimated market shares (q_t) for this market
        estimated_shares = estimated_market_shares[market]
        
        # Update delta vector using the rule: delta_new = delta_current + ln(s_t) - ln(q_t)
        updated_delta_t = delta_t + np.log(observed_shares) - np.log(estimated_shares)
        
        # Store the updated delta vector for this market
        updated_delta_market[market] = updated_delta_t
    
    return updated_delta_market
def iterate_until_convergence(delta_market, utility_matrices, agent_data, product_data, tolerance=1e-12):
    """
    Iteratively update delta vectors and estimate market shares until convergence.
    
    Parameters:
    - delta_market: Initial delta vectors (randomly initialized).
    - utility_matrices: Dictionary of utility matrices for each market.
    - agent_data: DataFrame containing agent information (weights) for each market.
    - product_data: DataFrame containing product data (observed shares).
    - tolerance: Convergence tolerance level.
    
    Returns:
    - final_delta_market: The final delta vectors after convergence.
    - final_market_shares: The estimated market shares at convergence.
    """
    # Initialize
    iteration = 0
    converged_markets = set()  # Track which markets have converged
    
    # Track previous delta for each market
    previous_delta_market = delta_market.copy()
    
    while True:  # No iteration limit, will stop only when all markets have converged
#         print(f"remaining {len(previous_delta_market) - len(converged_markets)}")
        
        # Calculate estimated market shares for the current iteration
        estimated_market_shares = calculate_market_shares(previous_delta_market, utility_matrices, agent_data, product_data)
        
        # Update delta for markets that haven't converged
        updated_delta_market = {}
        for market in previous_delta_market.keys():
            if market in converged_markets:
                # Skip markets that have already converged
                updated_delta_market[market] = previous_delta_market[market]
                continue
            
            # Update delta for this market
            updated_delta_t = previous_delta_market[market] + np.log(product_data[product_data['market_ids'] == market]['shares'].values) - np.log(estimated_market_shares[market])
            updated_delta_market[market] = updated_delta_t
            
            # Check for convergence: norm(delta_{r+1} - delta_r) < tolerance
            delta_diff_norm = np.linalg.norm(updated_delta_t - previous_delta_market[market])
            if delta_diff_norm < tolerance:
#                 print(f"Market {market} has converged.")
                converged_markets.add(market)  # Mark this market as converged
        
        # Update the delta_market for the next iteration
        previous_delta_market = updated_delta_market
        
        # Check if all markets have converged
        if len(converged_markets) == len(previous_delta_market):
#             print("All markets have converged.")
            break
        
        iteration += 1
    
    # Once finished, return the final delta vectors and market shares
    final_delta_market = previous_delta_market
    final_market_shares = calculate_market_shares(final_delta_market, utility_matrices, agent_data, product_data)
    
    return final_delta_market, final_market_shares

def get_instruments_all_markets(product_data):
    # Create a list of the instrument columns (demand_instruments0 to demand_instruments19)
    instrument_columns = [f'demand_instruments{i}' for i in range(20)]
    
    # Group the data by market_id
    markets = product_data.groupby('market_ids')
    
    # Create a dictionary to store instrument data for each market
    instruments_all_markets = {}
    
    # Loop through each market and extract instruments
    for market_id, group in markets:
        # Extract the instrument columns for this market and convert to NumPy array
        instruments_all_markets[market_id] = group[instrument_columns].values  # Array of instruments for each product
    
    return instruments_all_markets

def get_price_all_markets(product_data):
    # Group the data by market_id
    markets = product_data.groupby('market_ids')
    
    # Create a dictionary to store price data for each market
    price_all_markets = {}
    
    # Loop through each market and extract prices
    for market_id, group in markets:
        price_all_markets[market_id] = group['prices'].values  # Extract prices for each product in the market
    
    return price_all_markets

def stack_instruments_all_markets(instruments_all_markets):
    """
    Stack the instrument matrices for all markets into one large instrument matrix.
    
    Parameters:
    - instruments_all_markets: Dictionary of instrument matrices for each market.
    
    Returns:
    - Z_all: Stacked instrument matrix for all products across all markets.
    """
    Z_all = []
    
    # Loop over all markets
    for market in instruments_all_markets.keys():
        # Get the instrument matrix for the current market
        Z = instruments_all_markets[market]
        
        # Append to the list
        Z_all.append(Z)
    
    # Stack instrument matrices from all markets
    Z_all = np.vstack(Z_all)
    return Z_all

def gmm_objective_to_minimize(params, product_data, agent_data):
    """
    GMM objective function to minimize.
    
    Parameters:
    - params: A list [beta_0, beta_sugar, beta_price,alpha_nu,beta_cons,alpha_0] that contains the parameters to optimize.
    - product_data
    - agent data
    Returns:
    - gmm_value: The GMM objective value to minimize.
    """
    beta_0,beta_sugar,beta_price = params[0],params[1],params[2]
    beta_vector = np.array([beta_0, beta_sugar, beta_price])  
    alpha_nu = params[3]  # Noise scaling factor
    beta_cons = params[4]
    alpha_0 = params[5]
    markets = product_data['market_ids'].unique()
    delta_market = {}
    for market in markets:
        # Get the number of products in this market
        num_products = product_data[product_data['market_ids'] == market].shape[0]
        # Initialize random delta (market shares)
        delta_market[market] = initialize_random_delta(num_products)
    # Loop through markets and compute utility matrix for each market
    utility_matrices = {}
    for market in markets:
        # Filter product and agent data for the market
        market_product_data = product_data[product_data['market_ids'] == market]
        market_agent_data = agent_data[agent_data['market_ids'] == market]
        # Compute the utility matrix using vectorized operations
        utility_matrix = compute_utilities(beta_vector, alpha_nu, market_product_data, market_agent_data)
        # Store the result
        utility_matrices[market] = utility_matrix
    # Now, run the function to compute estimated market shares
    estimated_market_shares = calculate_market_shares(delta_market, utility_matrices, agent_data, product_data)
    # Now, update the delta vectors using the observed and estimated market shares
    updated_delta_market = update_delta(delta_market, product_data, estimated_market_shares)
    # Run the iterative process without a limit on the number of iterations
    final_delta_market, final_market_shares = iterate_until_convergence(delta_market, utility_matrices, agent_data, product_data, tolerance=1e-12)
    instruments_all_markets = get_instruments_all_markets(product_data)
    price_all_markets = get_price_all_markets(product_data)
    Z_all = stack_instruments_all_markets(instruments_all_markets)
    delta_all_markets = final_delta_market
    
    # Step 1: Calculate residuals (xi) for all markets
    xi_all = []

    # Loop over all markets
    for market in delta_all_markets.keys():
        # Calculate xi for the current market
        delta = delta_all_markets[market]
        price = price_all_markets[market]
        xi = delta - beta_cons - alpha_0 * price

        # Append the residuals to the list
        xi_all.append(xi)

    # Stack residuals from all markets into a single vector
    xi_all = np.concatenate(xi_all)
    # Step 2: Compute the GMM objective value: xi' Z Z' xi
    ZZ_transpose = Z_all @ Z_all.T

    # Calculate the GMM objective xi' Z Z' xi
    gmm_value = xi_all.T @ ZZ_transpose @ xi_all
    return gmm_value



product_data = df
initial_guess = [0.1,0.1,0.1,0.1,0.1,0.1]

# Define the arguments for the GMM objective function
args = ( product_data,agent_data)

# Use scipy's minimize function to minimize the GMM objective
result = minimize(gmm_objective_to_minimize, initial_guess, args=args, method='BFGS')

betav = result.x[:3]
alphan = result.x[3]
bt0,al0 = result.x[4], result.x[5]
utilities = compute_utilities(betav, alphan, product_data[product_data['market_ids'] == 'C01Q1'], agent_data[agent_data['market_ids'] == 'C01Q1'])
utility = {"C01Q1":utilities}
deltas = []
for p in product_data[product_data['market_ids'] == 'C01Q1']['prices']:
    deltas.append(bt0 + al0*p)
deltas = np.array(deltas)
dm1 = {'C01Q1':np.array(deltas)}
alphas = []
for i,j in zip(agent_data[agent_data['market_ids'] == 'C01Q1']['income'], agent_data[agent_data['market_ids'] == 'C01Q1']['nodes0']):
    alphas.append(al0 + betav[0]*i + alphan*j)
num_products = utilities.shape[0]
num_agents = utilities.shape[1]

# Initialize array to store estimated market shares for each product in this market
product_shares = np.zeros(num_products)

# Loop over each product
for j in range(num_products):
    # Numerator: w_it * exp(delta_jt + u_ijt)
    numerator =  np.exp(deltas[j] + utilities[j, :])

    # Denominator: 1 + sum_k exp(delta_kt + u_ikt) for each agent i
    exp_term = np.exp(deltas[:, np.newaxis] + utilities)  # Shape: (num_products, num_agents)
    denominator = 1 + np.sum(exp_term, axis=0)  # Sum over products for each agent

    # Compute market share for product j in market t
    product_shares[j] = np.mean(np.array(alphas) * numerator / denominator * (1 - numerator / denominator))

elasticity = []
for i,j,z in zip(product_data[product_data['market_ids'] == 'C01Q1']['prices'],product_data[product_data['market_ids'] == 'C01Q1']['shares'],product_shares):
    elasticity.append(-i/j*z)

plt.scatter(product_data[product_data['market_ids'] == 'C01Q1']['prices'],elasticity)
plt.xlabel('price')
plt.ylabel('Own Price Elasticity')
plt.title('Own Price Elasticity v.s. Price for C01Q1 (BLP)')
plt.savefig('Elasticity_BLP',dpi = 400)
plt.show()
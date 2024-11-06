#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:41:53 2024

@author: elanieloiswatson
"""

import pyblp
import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS
import statsmodels.formula.api as smf
import statsmodels.api as sm
pyblp.options.digits = 2
pyblp.options.verbose = False
pyblp.__version__
import matplotlib.pyplot as plt
from scipy.optimize import minimize

agent_data = pd.read_csv('/Users/elanieloiswatson/Downloads/agent_data.csv')
product_data = pd.read_csv('/Users/elanieloiswatson/Downloads/product_data.csv')
product_data.head()

#Q1
agent_data = pd.read_csv('/Users/elanieloiswatson/Downloads/agent_data.csv')
product_data = pd.read_csv('/Users/elanieloiswatson/Downloads/product_data.csv')
product_data.head()
product_data['market_share']= product_data.groupby('market_ids')['shares'].transform('sum')
product_data['non_market_share'] = 1- product_data['market_share']
product_data['log_market_share'] = np.log(product_data['shares'])
product_data['log_non_market_share'] = np.log(product_data['non_market_share'])
product_data['log_difference']=product_data['log_market_share'] - product_data['log_non_market_share']
results = smf.ols('log_difference ~ prices + sugar', data = product_data).fit()
print(results.summary())
dependent_var = product_data['log_difference']
exogenous_vars = sm.add_constant(product_data['sugar'])
endogenous_var = product_data['prices']
instruments = product_data[['demand_instruments0','demand_instruments1','demand_instruments2','demand_instruments3','demand_instruments4','demand_instruments5','demand_instruments6','demand_instruments7','demand_instruments8','demand_instruments9','demand_instruments10','demand_instruments11','demand_instruments12','demand_instruments13','demand_instruments14','demand_instruments15','demand_instruments16','demand_instruments17','demand_instruments18','demand_instruments19']]
model = IV2SLS(dependent_var, exogenous_vars, endogenous_var, instruments)
results2 = model.fit()
print(results2)

product_data['left'] = 1-product_data['shares']
product_data['product']=product_data['prices']*product_data['left']
product_data['elasticity'] = -11.291* product_data['product']


plt.figure(figsize=(8, 6))
for category in product_data['market_ids'].unique():
    subset = product_data[product_data['market_ids'] == 'C01Q1']
    plt.scatter(subset['prices'], subset['elasticity'], label=category)
plt.xlabel('prices')
plt.ylabel('own_elasticity')
plt.title('Scatter Plot for unique market')
plt.show()

#Q2
# construct initial delta matrix
rng = np.random.default_rng()
def initial_random_delta(num_products):
    random_shares = np.random.rand(num_products)
    return random_shares / random_shares.sum()
# define x_it, sigam, d_it, v_it, phi
def computed_utility(phi,sigma,product_data,agent_data):
    x_it = np.column_stack((np.ones(len(product_data)), 
                        product_data['sugar'].values, 
                        product_data['prices'].values))
    d_it = agent_data['income'].values.reshape(1, -1)
    v_it = agent_data['nodes0'].values.reshape(1, -1)
    utility_matrix = x_it @ (phi[:, np.newaxis] * d_it + sigma * v_it)
    return utility_matrix
def calculated_market_shares(delta_market, utility_vector, agent_data, product_data):
    market_shares = {}
    for market in delta_market.keys():
        delta_t = delta_market[market]
        utility_matrix = utility_vector[market]  
        market_agents = agent_data[agent_data['market_ids'] == market]
        weight = market_agents['weights'].values  
        num_products = utility_matrix.shape[0]
        num_agents = utility_matrix.shape[1]
        product_shares = np.zeros(num_products)
        for j in range(num_products):
            numerator =  np.exp(delta_t[j] + utility_matrix[j, :])
            exp_term = np.exp(delta_t[:, np.newaxis] + utility_matrix)  
            denominator = 1 + np.sum(exp_term, axis=0) 
            product_shares[j] = np.mean(numerator / denominator)
            market_shares[market] = product_shares
    return market_shares
def updated_delta(delta_market,estimated_market_shares,product_data):
    update_delta = {}
    for market in delta_market.keys():
        delta_t = delta_market[market]
        market_product_data = product_data[product_data['market_ids'] == market]
        observed_shares = market_product_data['shares'].values
        estimated_shares = estimated_market_shares[market]
        updated_delta_t = delta_t + np.log(observed_shares) - np.log(estimated_shares)
        updated_delta[market] = updated_delta_t
    return update_delta
# test for convergence
def iteration(difference,epsilon,delta_market,utility_market):
    i = 0
    converged_market_shares = {}
    previous_delta_market = delta_market.copy()
    while True:
        estimated_market_shares = calculated_market_shares(previous_delta_market, utility_market)
    updated_delta_market = {}
    for market in previous_delta_market.keys():
        if market in converged_market_shares:
           updated_delta_market[market] = previous_delta_market[market]
           continue
        updated_delta_t = previous_delta_market[market] + np.log(product_data[product_data['market_ids'] == market]['shares'].values) - np.log(estimated_market_shares[market])
        updated_delta_market[market] = updated_delta_t
        difference = np.linalg.norm(updated_delta_t - previous_delta_market[market])
        epsilon = 1e-12
        if difference < epsilon:
              converged_market_shares.add(market)
              previous_delta_market = updated_delta_market
        if len(converged_market_shares) == len(previous_delta_market):
            break
    i = i + 1
    
    final_delta_market = previous_delta_market
    final_market_shares = calculated_market_shares(final_delta_market, utility_market)
    return final_delta_market, final_market_shares
def get_instruments(product_data):
    instruments_columns = [f'demand_instruments{i}' for i in range(20)]
    market = product_data.groupby('market_ids')
    instruments = {}
    for market_ids, group in market:
        instruments[market_ids] = group[instruments_columns].values  
    
    return instruments
def get_prices(product_data):
    market = product_data.groupby('market_ids')
    prices = {}
    for market_ids, group in market:
        prices[market_ids] = group['prices'].values
    return prices
def Z(instruments):
    Z_aggregate = []
    for market in instruments.keys():
        Z = instruments[market]
        Z_aggregate.append(Z)
    Z_aggregate = np.vstack(Z_aggregate)
    return Z_aggregate
def GMM(params,product_data, agent_data):
    beta_0,beta_sugar,beta_price = params[0],params[1],params[2]
    phi = np.array([beta_0, beta_sugar, beta_price])
    sigma = params[3]
    beta_constant = params[4]
    alpha_constant = params[5]
    markets = product_data['market_ids'].unique()
    delta_market = {}
    for market in markets:
        num_products = product_data[product_data['market_ids'] == market].shape[0]
        delta_market[market] = initial_random_delta
    utility_vector = {}
    for market in markets:
        # Filter product and agent data for the market
        market_product_data = product_data[product_data['market_ids'] == market]
        market_agent_data = agent_data[agent_data['market_ids'] == market]
        utility_matrix = computed_utility(phi, sigma, market_product_data, market_agent_data)
        utility_vector[market] = utility_matrix
    estimated_market_shares = calculated_market_shares(delta_market, utility_vector,product_data,agent_data)
    updated_delta_market = updated_delta(delta_market, estimated_market_shares,product_data)
    epsilon = 1e-12
    final_delta_market, final_market_shares = iteration(delta_market, utility_vector,epsilon)
    instruments = get_instruments(product_data)
    prices = get_prices(product_data)
    Z_aggregate = Z(instruments)
    delta_all_markets = final_delta_market
    
    xi_aggregate = []
    for market in delta_all_markets.keys():
        delta = delta_all_markets[market]
        price = prices[market]
        xi = delta - beta_constant - alpha_constant * price
        xi_aggregate.append(xi)
    xi_aggregate = np.concatenate(xi_aggregate)
    ZZ_transpose = Z_aggregate @ Z_aggregate.T
    gmm_value = xi_aggregate.T @ ZZ_transpose @ xi_aggregate
    return gmm_value
initial_guess = [0.1,0.1,0.1,0.1,0.1,0.1]
args = ( product_data,agent_data )
result = minimize(GMM, initial_guess, args=args, method='BFGS')

    

            




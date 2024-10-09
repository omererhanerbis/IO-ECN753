# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

#import packages
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
#from numba import jit
from scipy.optimize import minimize
import time                                                                     # just to check how long it takes to run

# Load data
data_agents = pd.read_csv("Data/agent_data.csv")
data_products = pd.read_csv("Data/product_data.csv")



# Add constant
data_products["constant"] = 1 

# Set market and product ids as indices

data_products.head()                                                            # check original data
data_products = data_products.set_index("market_ids", append = False)           # set market id as index

enc = LabelEncoder()
enc.fit(data_products["product_ids"])                                           # create numeric values for product ids
data_products["product_ids"] = enc.transform(data_products["product_ids"])      # turn product id into integers to treat it as a fake time index

data_products.head()                                                            # check resulting data
data_products = data_products.set_index("product_ids", append = True)           # set product id as second index
data_products.head()                                                            # check resulting data


#Creating shares relative to outside option for each market:
i = data_products.index.get_level_values('product_ids')
aggregata_data_products = 1 - data_products[(i >= 0) & (i <= 23)].sum(level=0)
share_0 = aggregata_data_products["shares"]                                     # share of outside option in market t

cities = data_products.index.get_level_values('market_ids')
data_products['share_0'] = share_0.loc[cities].values                           # joining both datasets for easier computation
             
data_products["rel_shares"] = 0
for n in np.arange(0,len(data_products)):
    data_products.iloc[n, -1] = np.log(data_products.iloc[n, 1]) - np.log(data_products.iloc[n, -2])
    
     


##############################################################################
###### Problem 1
##############################################################################
######### Part 1.a
##############################################################################
m_ols = PanelOLS(dependent = data_products["rel_shares"], 
                    exog = data_products[["constant", "sugar", "prices"]],
                    entity_effects = False,
                    time_effects = False)

# m_ols = PanelOLS.from_formula("shares ~ constant + sugar + prices", data = data_products) # quivalent syntax

m_ols.fit()
print("================================================================================")
print("                                 Simple OLS                                     ")
print("================================================================================")
print(m_ols.fit())

##############################################################################
######### Part 1.b
##############################################################################
m_stg1 = PanelOLS(dependent = data_products["prices"],
                  exog = data_products[["constant",
                                        "demand_instruments0",
                                        "demand_instruments1",
                                        "demand_instruments2",
                                        "demand_instruments3",
                                        "demand_instruments4",
                                        "demand_instruments5",
                                        "demand_instruments6",
                                        "demand_instruments7",
                                        "demand_instruments8",
                                        "demand_instruments9",
                                        "demand_instruments10",
                                        "demand_instruments11",
                                        "demand_instruments12",
                                        "demand_instruments13",
                                        "demand_instruments14",
                                        "demand_instruments15",
                                        "demand_instruments16",
                                        "demand_instruments17",
                                        "demand_instruments18",
                                        "demand_instruments19"]],
                    entity_effects = False,
                    time_effects = False)

m_stg1.fit()
x_hat = m_stg1.fit().predict(fitted = True)

data_products["prices_hat"] = x_hat

m_stg2 = PanelOLS(dependent = data_products["rel_shares"], 
                    exog = data_products[["constant", "prices_hat", "sugar"]],
                    entity_effects = False,
                    time_effects = False)

m_stg2.fit()
print("================================================================================")
print("                                   2SLS OLS                                     ")
print("================================================================================")
print(m_stg2.fit())

betas = m_stg2.fit().params

##############################################################################
######### Part 1.c
##############################################################################

data_products["shares_hat"] = 0
for n in np.arange(0,len(data_products)):
    data_products.iloc[n, -1] = np.exp(data_products.iloc[n, -3] + np.log(data_products.iloc[n, -5]))

price_elast = betas[1]*data_products["prices"]*(1 - data_products["shares"])

mkt1_price_elast = price_elast.iloc[0:24]

figure, axes = plt.subplots()
axes.scatter(data_products.iloc[0:24, 2], mkt1_price_elast)
axes.set(title = "Prices vs estimated elasticities in market C01Q1", 
         xlabel = "Prices",
         ylabel = "Estimated elasticities")
plt.savefig("Figures/Problem1_Part_c.png")

##############################################################################
###### Problem 2
##############################################################################
######### Part 2.a
##############################################################################

# Load data
data_agents = pd.read_csv("Data/agent_data.csv")
data_products = pd.read_csv("Data/product_data.csv")




# Add constant and log_shares
data_products["constant"] = 1
data_products["log_shares"] = np.log(data_products["shares"]) 

# Take variables of interest
obs_var = np.array(data_products[["market_ids", "constant", "sugar", "prices"]])
obs_var_exo = np.array(data_products[["constant", "sugar"]])
dist = np.array(data_agents[["market_ids", "income", "nodes0"]])
shares = np.array(data_products[["market_ids", "shares"]])
log_shares = np.array(data_products[["market_ids", "log_shares"]]) 
#prices = np.array([data_products["market_ids", "prices"]])
instruments = np.array(data_products.iloc[:, 6:26])
instrumental_variables = np.concatenate((obs_var_exo, instruments), axis = 1)
index_mkt = np.array(pd.unique(data_products["market_ids"]))


# Classify parameters
K = obs_var_exo.shape[1]                                                        # Number of coefficients
def ut_params(params):
    params = np.array(params)
    lin_param = np.array([params[0], 0, params[1]]).reshape(K + 1, 1)
    demo_param = params[2:5].reshape(3, 1)
    unobs_param = np.diag(np.array([0, 0, params[5]]))
    return lin_param, demo_param, unobs_param


#Functions for predicting shares
def pred_shares_draw(delta, demo_param, unobs_param, obs_var, draw):
    delta = delta.reshape(delta.size, 1)
    demo_taste = demo_param*draw[0]
    unobs_taste = (unobs_param @ np.array([[0], [0], [draw[1]]])
                         ) # constructed numpy arrary with magic numbers because of model specification
    nonlin_taste = (np.array(obs_var) @ (demo_taste + unobs_taste))
    quantity = np.exp((delta + nonlin_taste).astype(float))
    mkt_size = 1 + sum(quantity)                                                # 1 comes from outside option
    shares = quantity/mkt_size
    return shares.flatten()                                                     # so shares are a 1-D array

def pred_shares(delta, demo_param, unobs_param, obs_var, dist):
    shares_draw = (lambda y: pred_shares_draw(delta, demo_param, unobs_param, obs_var, y))
    monte_carlo = np. apply_along_axis(shares_draw, 1, dist)
    mkt_shares = np.mean(monte_carlo, axis = 0)
    return mkt_shares
    

#@jit
def mkt_inv(demo_param, unobs_param, obs_var, dist, log_shares, delta_guess=None, tolerance=1e-12, max_iteration=1000, error=1):
    # Initialize delta_init based on whether delta_guess is provided
    if delta_guess is None:                                              # If there is no guess at all on any delta
        J = obs_var.shape[0]
        delta_init = np.zeros((J, 1))  # Initialize with zeros
    else:
        delta_init = delta_guess

    # while loop for iteration
    iteration = 1
    while error > tolerance:
        
        pred_shares_loop = pred_shares(delta_init, demo_param, unobs_param, obs_var, dist)  
        log_shares_hat = np.log(pred_shares_loop).reshape(pred_shares_loop.size, 1)
        delta_loop = delta_init + log_shares - log_shares_hat 
        error = np.max(np.abs(delta_init - delta_loop))
        delta_init = delta_loop  # Update guess

        # Break if max iterations reached
        if iteration >= max_iteration:
            break                                                               
        
#        print("Iteration: " + str(iteration) + " | Error " + str(error))        #to make sure this thing is running b/c I was losing my mind when I did not have this
        iteration += 1  # Increase iteration

    return delta_loop.flatten()

#@jit
def inversion(demo_param, unobs_param, obs_var, dist, log_shares, index):
    inversion = np.array([])  # Initialize empty array
    prod_mkt = obs_var[:, 0]
    agent_mkt = dist[:, 0]

    for i in index:
        prod_submkt = (prod_mkt == i)
        agent_submkt = (agent_mkt == i)

        loop_prod = obs_var[prod_submkt][:, 1:]
        loop_dist = dist[agent_submkt][:, 1:]
        loop_log_shares = log_shares[prod_submkt][:, 1:]

        loop_inv = mkt_inv(demo_param, unobs_param, loop_prod, loop_dist, loop_log_shares)

        inversion = np.concatenate((inversion, loop_inv.flatten()), axis=0)  # Use flatten to ensure 1D concatenation

    return inversion


#GMM
def demand_shock(lin_param, demo_param, unobs_param, obs_var, dist, log_shares, index):
    delta = inversion(demo_param, unobs_param, obs_var, dist, log_shares, index)
    X = obs_var[:, 1:]
    lin_taste = X @ lin_param
    X_i = delta.reshape(delta.size, 1) - lin_taste.reshape(lin_taste.size, 1)
    return X_i

def proj_matrix(Z):
    return Z @ np.linalg.inv(Z.T @ Z) @ Z.T

def gmm_w_inst(lin_param, demo_param, unobs_param, obs_var, dist, log_shares, index, weights):
    X_i = demand_shock(lin_param, demo_param, unobs_param, obs_var, dist, log_shares, index)
    obj_fun = X_i.T @ weights @ X_i
    return float(obj_fun)
    
#Elasticities
def shares_price_deriv(lin_param, delta, demo_param, unobs_param, obs_var, dist):
    nonlin_p_param = np.array([[float(demo_param[2])], [unobs_param[2, 2]]])
    price_coef = lin_param[2] + dist @ nonlin_p_param
    shares_int = (lambda y: pred_shares_draw(delta, demo_param, unobs_param, obs_var, y))
    
    shares_grid = np.apply_along_axis(shares_int, 1, dist)
    shares_p_deriv_grid = price_coef*shares_grid*(1 - shares_grid)
    shares_p_deriv = np.mean(shares_p_deriv_grid, axis = 0)
    return shares_p_deriv


#####
# Computation
#####

weights = proj_matrix(instrumental_variables)

# [ -3.,  -2.,   4.,   0., -33.,   1.]
init_params = tuple([-3., -2., 4., 0., -33., 1.])                               # Thanks Chris

lin_param, demo_param, unobs_param = ut_params(init_params)

obj_fun = (lambda params: gmm_w_inst(*ut_params(params), 
                                     obs_var, 
                                     dist, 
                                     log_shares, 
                                     index_mkt, 
                                     weights))

tik = time.time()
results = minimize(obj_fun, 
                   init_params, 
                   method = "L-BFGS-B",
                   options = {"disp": True})    

tok = time.time()
print("run time: " + str((tok - tik)/60) + " minutes")

estimates_blp = results.x
print("================================================================================")
print("                                      BLP                                       ")
print("================================================================================")
print(results.x)

##############################################################################
######### Part 2.b
##############################################################################

    
# set up numpy arrays for market C01Q1
mkt_obs_var= np.array(data_products[data_products["market_ids"] == "C01Q1"][["constant", "sugar", "prices"]])
mkt_dist = np.array(data_agents[data_agents["market_ids"] == "C01Q1"][["income", "nodes0"]])
mkt_log_shares = np.array(data_products[data_products["market_ids"] == "C01Q1"]["log_shares"])
mkt_log_shares = mkt_log_shares.reshape(mkt_log_shares.size, 1)
mkt_p = np.array(data_products[data_products["market_ids"] == "C01Q1"]["prices"])
mkt_shares = np.array(data_products[data_products["market_ids"] == "C01Q1"]["shares"])
    
    # Organize estimates into linear, demographic, and unobserved parameters
K = mkt_obs_var.shape[1] - 1

lin_param, demo_param, unobs_param = ut_params(estimates_blp)

mkt_inv = mkt_inv(demo_param, unobs_param, mkt_obs_var, mkt_dist, mkt_log_shares)

mkt_delta = mkt_inv[0]
mkt_delta = mkt_delta.reshape(mkt_delta.size, 1)
share_p_deriv = shares_price_deriv(lin_param, mkt_delta, demo_param, unobs_param, mkt_obs_var, mkt_dist)

p_elast = share_p_deriv*(mkt_p/mkt_shares)


#Figures

figure, axes = plt.subplots()
axes.scatter(mkt_p, p_elast)
axes.set(title = "Prices vs estimated elasticities in market C01Q1", 
         xlabel = "Prices",
         ylabel = "Estimated elasticities")
plt.savefig("Figures/Problem2_Part_b_1.png")

print("Correlation between price and elasticities using BLP is " + str(np.corrcoef(mkt_p, p_elast)[0,1]))

figure, axes = plt.subplots()
plt.scatter(mkt_p, mkt1_price_elast, label = "2SLS")
axes.scatter(mkt_p, p_elast, label = "BLP")

axes.set(title = "Prices vs estimated elasticities in market C01Q1", 
         xlabel = "Prices",
         ylabel = "Estimated elasticities")
axes.legend()
plt.savefig("Figures/Problem2_Part_b_2.png")

fig, axes = plt.subplots()
scatter = plt.scatter(mkt1_price_elast, p_elast, c=mkt_p, cmap='jet', label='Elasticities')
min_limit = min(np.min(mkt1_price_elast), np.min(p_elast))
max_limit = max(np.max(mkt1_price_elast), np.max(p_elast))
axes.plot([min_limit, max_limit], [min_limit, max_limit], linestyle="--", color="k", label='45-degree line')
axes.text(-0.2, -0.1, "45°", fontsize=12, color="k", verticalalignment='bottom')
axes.set(title="2SLS elasticities vs BLP elasticities in market C01Q1", 
         xlabel="2SLS elasticities",
         ylabel="BLP elasticities")
cbar = plt.colorbar(scatter)
cbar.set_label('Price')
axes.set_xlim([-2.5, 0])
axes.set_ylim([min_limit, max_limit])
plt.savefig("Figures/Problem2_Part_b_3.png")



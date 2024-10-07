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
from numba import jit
import scipy as sp
from scipy.optimize import minimize

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

price_elast = betas[2]*data_products["prices"]*(1 - data_products["shares"])

plt.scatter(data_products.iloc[0:24, 2], price_elast.iloc[0:24])
plt.title("Prices vs estimated elasticities in market C01Q1")
plt.xlabel("Prices")
plt.ylabel("Estimated elasticities")
plt.show()


##############################################################################
###### Problem 2
##############################################################################
######### Part 2.a
##############################################################################

# Load data
data_agents = pd.read_csv("Data/agent_data.csv")
data_products = pd.read_csv("Data/product_data.csv")


#Matrices for product, demographic and instrumental variables
X_p = data_products[["prices", "sugar"]].values
X_d = data_agents[["income", "nodes0"]].values
Z = data_products[["demand_instruments0",
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
                  "demand_instruments19"]]

#Coefficients for product characteristics
np.random.seed(42)

theta = np.array([np.random.normal(), np.random.normal()])
sigma = np.array([np.random.normal(), np.random.normal()])

def utility(delta, X_p, X_d, theta, sigma):
    delta_expanded = np.repeat(delta, X_d.shape[0], axis = 0).reshape(len(delta), X_d.shape[0])
    mu_ijt = np.dot(X_p, theta)[:, np.newaxis] + np.dot(X_d, sigma)
    U = delta_expanded + mu_ijt
    return U

def comp_market_shares(delta, X_p, X_d, theta, sigma):
    U = utility(delta, X_p, X_d, theta, sigma)
    
    #Logit
    exp_U = np.exp(U)
    sum_exp_U = np.sum(exp_U, axis = 0)
    choice_prob = exp_U / sum_exp_U
    
    #Market shares for each product
    mkt_share = np.mean(choice_prob, axis = 1)
    return mkt_share

@jit
def contraction_mapping(delta, X_p, X_d, obs_shares, theta, sigma, tolerance = 10e-6, max_iterations = 1000):
    it = 0
    while it < max_iterations:
        predicted_shares = comp_market_shares(delta, X_p, X_d, theta, sigma)
        delta_new = delta + np.log(obs_shares / predicted_shares)
        
        if np.max(np.abs(delta_new - delta)) < tolerance:
            break
        
        delta = delta_new
        it += 1
    
    return delta

@jit
def gmm_w_inst(delta, X_p, X_d, Z, obs_shares, theta, sigma):
    predicted_shares = comp_market_shares(delta, X_p, X_d, theta, sigma)
    
    moment_error = obs_shares - predicted_shares
    weighted_moments = np.dot(moment_error.T, Z)
    
    return np.sum(weighted_moments**2)

#Compute delta
delta_init = np.zeros(X_p.shape[0])
results = minimize(gmm_w_inst, 
                   delta_init, 
                   args = (X_p, X_d, Z, data_products["shares"].values, theta, sigma),
                   method = "BFGS")

delta_estimates = results.x
                                            # deltas for first market

##############################################################################
######### Part 2.b
##############################################################################

def comp_elasticities(delta, X_p, X_d, theta, sigma, prices, pred_shares):
    
    n_prod = X_p.shape[0]

    theta_price = theta[0]

    elasticities = np.zeros((n_prod, n_prod))

    for j in range(n_prod):
        elasticities[j] = theta_price*prices[j] * (1 - pred_shares[j])

    for j in range(n_prod):
        elasticities[j] = elasticities[j]/pred_shares[j]
        
    return elasticities

prices = data_products["prices"].values

pred_mkt_shares = comp_market_shares(delta_estimates, X_p, X_d, theta, sigma)

elasticities = comp_elasticities(delta_estimates, X_p, X_d, theta, sigma, prices, pred_mkt_shares)

elasticities_C01Q1 = elasticities[0:24,0]

plt.scatter(data_products.iloc[0:24, 2], elasticities_C01Q1)
plt.title("Prices vs estimated elasticities in market C01Q1")
plt.xlabel("Prices")
plt.ylabel("Estimated elasticities")
plt.show()

plt.scatter(data_products.iloc[0:24, 2], price_elast.iloc[0:24], color = "blue", label = "2SLS")
plt.scatter(data_products.iloc[0:24, 2], elasticities_C01Q1, color = "red", label = "BLP")
plt.title("Prices vs estimated elasticities in market C01Q1")
plt.xlabel("Prices")
plt.ylabel("Estimated elasticities")
plt.legend()
plt.show()



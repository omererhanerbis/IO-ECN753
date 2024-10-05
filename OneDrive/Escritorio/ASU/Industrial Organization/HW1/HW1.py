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
m_stg1 = PanelOLS(dependent = data_products["sugar"],
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

data_products["sugar_hat"] = x_hat

m_stg2 = PanelOLS(dependent = data_products["rel_shares"], 
                    exog = data_products[["constant", "sugar_hat", "prices"]],
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


n_markets = data_products["market_ids"].unique().size
n_products = data_products["product_ids"].unique().size
n_agents = int(1/data_agents.iloc[0, 2])                                        # since equally weighted and all markets have same number of consumers

np.random.seed(42)
#Random coefficients
theta = np.random.normal(loc = 0, 
                               scale = 1, 
                               size = 3)                                        # price & sugar
sigma = np.random.normal(loc = 0,
                               scale = 1,
                               size = 3)                                        # income effect & taste_shock

#U = delta + random_coef*charac
def utility(delta, product_data, agent_data, random_coef):
    price, sugar = product_data["prices"], product_data["sugar"]
    income, taste_shock = agent_data["income"], agent_data["nodes0"]
    
    alpha = random_coef[3] + random_coef[4]*income + random_coef[5]*taste_shock
    beta_0 = random_coef[0] + random_coef[1]*income
    beta_sugar = random_coef[2] 
    return delta + beta_0 + beta_sugar*sugar + alpha*price

#Market shares calculation
@jit
def est_market_shares(delta, products, demographics, random_coef):
    market_ids = data_products["market_ids"].unique()
    mkt_shares = []
    
    for t in market_ids:
        market_products = products[products["market_ids"] == t]
        market_consumers = demographics[demographics["market_ids"] == t]
        
        #Utility for each consumer in market t
        utilities = []
        for i, product_data in market_products.iterrows():
            product_util = []
            for _, consumer_data in market_consumers.iterrows():
                product_util.append(utility(delta, product_data, consumer_data, random_coef))
            utilities.append(product_util)
            
        #From utilities to probabilities
        utilities = np.array(utilities)
        exp_utilities = np.exp(utilities)
        mkt_probs = exp_utilities/np.sum(exp_utilities, axis = 0)
        
        #Avg Mkt Share
        avg_mkt_share = np.mean(mkt_probs, axis = 1)
        mkt_shares.append(avg_mkt_share)
        
    return np.concatenate(mkt_shares)
    
#Objective function: GMM to minimize observed and sim shares
def gmm_obj(delta, products, demographics, observed_shares, random_coef):
    predicted_shares = est_market_shares(delta, products, demographics, random_coef)
    moment_error = observed_shares - predicted_shares
    return np.sum(moment_error**2)                                              # we want to minimize square error

#Estimate delta by minimizing the square error

# observed_shares = data_products.loc[data_products["market_ids"]=="C01Q1", "shares"].values
    
observed_shares = data_products["shares"].values
random_coef = np.concatenate([theta, sigma])

delta_init = np.zeros(n_markets)
result = minimize(gmm_obj, 
                  delta_init, 
                  args = (data_products, data_agents, observed_shares, random_coef), 
                  method = "BFGS") 

delta_est = result.x

#Contraction mapping
@jit
def contraction_mapping(delta, products, demographics, observed_shares, random_coef, tolerance = 10e-15, max_iterations = 1000):
    iter = 0
    while iter < max_iterations:
        predicted_shares = est_market_shares(delta, products, demographics, random_coef)
        delta_new = delta + np.log(observed_shares / predicted_shares)
        if np.max(np.abs(delta_new - delta)) < tolerance:
            break
        delta = delta_new
        iter += 1
    return delta

#Apply contraction mapping
delta_sol = contraction_mapping(delta_init, data_products, data_agents, observed_shares, random_coef)



                                
                                     
                                
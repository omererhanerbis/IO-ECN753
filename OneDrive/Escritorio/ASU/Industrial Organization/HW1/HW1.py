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
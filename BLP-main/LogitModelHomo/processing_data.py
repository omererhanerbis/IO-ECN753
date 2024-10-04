
import numpy as np 
import pandas as pd
import os

df = pd.read_csv(os.path.join(os.getcwd(),'LogitModelHomo\\product_data.csv'))
agent_data = pd.read_csv(os.path.join(os.getcwd(),'LogitModelHomo\\agent_data.csv'))
df['total_market_share'] = df.groupby('market_ids')['shares'].transform('sum')

# Step 2: Calculate "no market share"
df['no_market_share'] = 1 - df['total_market_share']

# Step 3: Calculate the log of product share and no market share
df['log_product_share'] = np.log(df['shares'])
df['log_no_market_share'] = np.log(df['no_market_share'])

# Step 4: Create the new column for log difference
df['log_difference'] = df['log_product_share'] - df['log_no_market_share']



import statsmodels.api as sm
from processing_data import df
import pandas as pd
import matplotlib.pyplot as plt
endog_var = df['prices']
exog_vars = df[['sugar']]
iv_vars = df[df.columns[6:26]] 

X_first_stage = pd.concat([exog_vars, iv_vars], axis=1)
X_first_stage = sm.add_constant(X_first_stage)  

# Use OLS for the first stage
first_stage_model = sm.OLS(endog_var, X_first_stage).fit()

# Get predicted values of price from the first stage
predicted_price = first_stage_model.fittedvalues

# Second Stage: Regress log_difference on exogenous variables + predicted price
X_second_stage = pd.concat([exog_vars, predicted_price], axis=1)
X_second_stage.columns = list(exog_vars.columns) + ['predicted_price']  # Rename the column
X_second_stage = sm.add_constant(X_second_stage)
y = df['log_difference']
# Perform the second stage regression (OLS)
second_stage_model = sm.OLS(y, X_second_stage).fit()

# Output the summary of the second-stage regression
print(second_stage_model.summary())

print(second_stage_model.params.values)
market1 = df[df['market_ids'] == 'C01Q1']
price_sensitivity = second_stage_model.params.values[-1]
price_elasticity = []
for i,j in zip(market1['prices'], market1['shares']):
    price_elasticity.append(price_sensitivity * i * (1 - j))
plt.scatter(market1['prices'], price_elasticity)
plt.xlabel('price')
plt.ylabel('Own Price Elasticity')
plt.title('Own Price Elasticity v.s. Price for C01Q1 (Homo)')
plt.savefig('Elasticity_homo',dpi = 400)
plt.show()
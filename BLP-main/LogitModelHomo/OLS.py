import statsmodels.api as sm
from processing_data import df

independent_vars = df[['prices','sugar']]

X = sm.add_constant(independent_vars)

# Step 3: Dependent variable (log_difference)
y = df['log_difference']

# Step 4: Run OLS regression
model = sm.OLS(y, X).fit()

# Step 5: Print the summary of the OLS regression results
print(model.summary())

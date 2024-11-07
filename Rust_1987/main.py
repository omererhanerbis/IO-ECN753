import os 
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from functions import *

data = pd.read_csv(os.path.join(os.getcwd(),'group_4.csv'))
data['delta'] = data.groupby('Bus_ID')['state'].diff().fillna(0)
df = data[['state','decision','delta']]
X = df.iloc[:, 0].values
I = df.iloc[:, 1].values

S = 90 # number of states
state = state_space(S) # state space vector
beta = 0.9999
tol = 0.0001
params = np.array([0.2,0.5])
# estimate transition probabilities theta_3 to build transition matrix
P = transition(S, θ_30,θ_31)

result = minimize(objective,params,args = (state, P, beta, tol, X, I),method = 'L-BFGS-B')
print(result.x)
paraname = ['theta11','rc']
values = result.x
estimates = pd.DataFrame({'name':paraname,'val':values})
estimates.to_csv('results.csv')
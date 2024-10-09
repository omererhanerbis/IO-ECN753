# In VSCode need to choose >interpreter, others, get to change the environment accordingly
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Read Data
start=datetime.now()

product_data = pd.read_csv("product_data.csv")
agent_data = pd.read_csv("agent_data.csv")

# Data manipulation

## Outside option market share
product_data["sum_product_shares"] = product_data.groupby("market_ids")["shares"].transform('sum')
product_data["outside_option_shares"] = 1 - product_data["sum_product_shares"]

## Log market shares
product_data["log_product_shares"] = np.log(product_data["shares"])
product_data["log_outside_option_shares"] = np.log(product_data["outside_option_shares"])
product_data["difference_log_shares"] = product_data["log_product_shares"] - product_data["log_outside_option_shares"]
y = product_data["difference_log_shares"].to_numpy()

## Add constant column to the data
product_data["constant"] = 1

# Model 1 - OLS
## OLS regressors
ols_regressors = product_data[["constant", "sugar", "prices"]]
## OLS regression 1
ols = sm.OLS(product_data["difference_log_shares"], ols_regressors).fit()
print(ols.summary())
## OLS regression 2
X = ols_regressors.to_numpy()
ols_estimates = np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y


# Model 2 - 2SLS

## 2SLS regression 1
### First stage
first_stage_regressors = product_data[["constant", "sugar"] + product_data.filter(like='demand_instruments').columns.tolist()]
first_stage = sm.OLS(product_data["prices"], first_stage_regressors).fit()
product_data["tsls_fitted_price"] = first_stage.fittedvalues
### Second stage
second_stage_regressors = sm.add_constant(product_data[["sugar", "tsls_fitted_price"]])
tsls = sm.OLS(product_data["log_product_shares"] - product_data["log_outside_option_shares"], second_stage_regressors).fit()
print(tsls.summary())
## 2SLS regression 2
### Z matrix of exogeneous variables and instruments
instruments_np = product_data.filter(like='demand_instruments').to_numpy()
tsls_exogeneous_regressors_np = product_data[["constant", "sugar"]].to_numpy()
Z = np.concatenate((tsls_exogeneous_regressors_np, instruments_np), axis=1)
# Matrix formulation calculation of 2SLS
tsls_estimates = np.linalg.inv(np.transpose(X)@Z@np.linalg.inv(np.transpose(Z)@Z)@np.transpose(Z)@X)@np.transpose(X)@Z@np.linalg.inv(np.transpose(Z)@Z)@np.transpose(Z)@y


# Price elasticities
C01Q1 = product_data[product_data["market_ids"] == "C01Q1"]
price_sensitivity = tsls.params.values[-1]
price_elasticity = []
for i,j in zip(C01Q1["prices"], C01Q1["shares"]):
    price_elasticity.append(price_sensitivity * i * (1 - j))


# Plot the Price elasticities
plt.scatter(C01Q1["prices"], price_elasticity)
plt.xlabel("Price")
plt.ylabel("Price Elasticity")
plt.title("2SLS - Price Elasticities for market C01Q1")
# plt.savefig("TSLS_Elasticities",dpi = 400)
plt.show()























# product_data.csv

# - market_ids: IDs for the different markets. 
#   The ID denotes a city-quarter. For example, 'C01Q1' captures a city (e.g., 'C01') in a quarter ('Q1').

# - product_ids: IDs for the different products.
#   A single code might correspond to, for example, 'Apple-Cinnamon Cheerios.'

# - shares: Precomputed market shares for the products.

# - prices: Prices of a cereal box (for each product).

# - sugar: Amount of sugar in each product.

# - demand_instruments0, ..., demand_instruments19: Precomputed demand instruments (e.g., 
#   demand_instruments used in the original paper).


# agent_data.csv

# market_ids: IDs for the different markets. 
# The ID denotes a city-quarter. For example, ‘C01Q1’ represents:
# - 'C01' as a city (e.g., city 1)
# - 'Q1' as a specific quarter (e.g., quarter 1).

# weights: The weight of each draw, which is set to 1/20.

# income: A draw from the income distribution in each market.
# This is an observed demographic variable and is specific to each market.

# nodes0: A draw from a standard normal distribution.
# This is used in the computation of the unobserved taste for characteristics.



# Data Dimensions

I = agent_data["market_ids"].value_counts().iloc[1] # number of individuals R
J = product_data["product_ids"].unique().size       # number of products
T = agent_data["market_ids"].unique().size          # number of markets
N = I * T                                           # number of observations for agent_data M
S = J * T                                           # number of observations for product_data
K = 2                                               # number of product features        - sugar and price
L = 1                                               # number of demographic variables   - income


# Skeleton for estimates
# Gamma       = np.zeros((K+1,L))
# Sigma       = np.zeros((K+1,K+1))
# cons        = np.zeros(K+1)

# Initiation values for estimates
theta       = np.ones(6)
# Gamma       = np.random.rand(K+1,L)
# Sigma[2,2]  = 1.5
# cons        = np.random.rand((K+1))

# Tha matrix of interaction term effects of income, K + 1 (sugar and price + constant)
D = np.tile(agent_data["income"].values, (K + 1, 1))

# Create the tuple of product features (including constant) by broadcasting to each consumer, for each good and for each market i.e. (x_jt, p_jt) matrix with i s.
A = X
A_0 = A[:,0]                                    # Assign the constant
A_0 = A_0.reshape((T,J))                        # Since constant is constant, create the sugar amount matrix
A_0_2 = np.broadcast_to(A_0,(I,)+A_0.shape)     # broadcast constant matrix for each consumer i, create the 3D array
A_1 = A[:,1]                                    # Assign the sugar amounts
A_1 = A_1.reshape((T,J))                        # Since all J product sugar values are same in any market, create the sugar amount matrix
A_1_2 = np.broadcast_to(A_1,(I,)+A_1.shape)     # broadcast each sugar amount matrix for each consumer i, create the 3D array
A_2 = A[:,2]                                    # Assign the prices
A_2 = A_2.reshape((T,J))                        # Since all J product prices are same in any market, create the price matrix
A_2_2 = np.broadcast_to(A_2,(I,)+A_2.shape)     # broadcast price matrix for each consumer i, create the 3D array

A_input = A_0_2,A_1_2,A_2_2                     # combine all matrices in a tuple









def predict_shares(gamma, sigma, D, delta, A_input, T, J, I):
    """
    Calculates predicted market shares given non-linear parameters using 3D-arrays and broadcasting.

    The function calculates the interaction of product characteristics and individual-specific taste variations, 
    then computes the predicted market shares. The method follows Nevo (2000, 2004) with first calculating mu_ijt matrix,
    then calculates the utilities with mean utility and these individual variations, and utilizing these calculates the shares.

    Args:
        gamma (np.array): A (K+1) x L matrix with coefficients of demographic variables.
        sigma (np.array): A (K+1) x (K+1) diagonal matrix representing the variance of unobserved taste interactions.
        D (np.array): A (K+1) x I matrix, where each row contains demographic information repeated across individuals.
        delta (np.array): A (T x J) matrix of mean utilities for each product in each market.
        A_input (tuple): A tuple of three (T x J) matrices representing product characteristics (constant, sugar, price).
        T (int): Number of markets.
        J (int): Number of products in each market.
        I (int): Number of individuals (agents).

    Returns:
        np.array: A (T x J) matrix representing the mean predicted market shares across individuals for each product in each market.
        np.array: A (I x T x J) matrix representing the individual-specific predicted market shares.
    """
    
    demographics_effects = gamma.transpose() * D.transpose()                    # this is the observed demographics unit effect matrix on utilities
    taste_effects = np.array((np.zeros_like(agent_data['nodes0']),
                              np.zeros_like(agent_data['nodes0']),
                              sigma[2,2]*agent_data['nodes0'])).transpose()     # this is the unobserved taste characteristics unit effect matrix on utilities
    B = demographics_effects + taste_effects                                    # this is the aggregate taste unit effect matrix
    
    # similar to A, following is for creating 3D arrays for these effects by broadcasting to each good, for each individual each market i.e. (Gamme * D_it +Sigma * nu_it) with j s.
    B_0 = B[:,0]
    B_0 = B_0.reshape(T,I).transpose()
    B_0_2 = np.broadcast_to(B_0[...,None],B_0.shape+(J,))
    B_1 = B[:,1]
    B_1 = B_1.reshape(T,I).transpose()
    B_1_2 = np.broadcast_to(B_1[...,None],B_1.shape+(J,))
    B_2 = B[:,2]
    B_2 = B_2.reshape(T,I).transpose()
    B_2_2 = np.broadcast_to(B_2[...,None],B_2.shape+(J,))
    
    
    # Calculate the mu_ijt i.e. consumer-level variation effects across mean utility (handout chapter 2)
    Z_0 = A_input[0] * B_0_2    # first product feature effect on each consumer - constant
    Z_1 = A_input[1] * B_1_2    # second product feature effect on each consumer - sugar
    Z_2 = A_input[2] * B_2_2    # third product feature effect on each consumer - price

    sum_Z = Z_1 + Z_2 + Z_0         # sum all these variation effects
    U = np.exp(sum_Z + delta)    # take the exponent of summed consumer-level variation and mean utilities

    denominator = 1 + np.sum(U, axis=2)
    denominator_exponential = np.broadcast_to(denominator[...,None],denominator.shape+(J,))
    estimated_shares = U / denominator_exponential
    #sigma is a T times J matrix
    return np.mean(estimated_shares,axis=0),estimated_shares






def predict_delta(delta_initial, gamma, sigma, D, A_input, T, J, I, tolerance = 10**-12):
    """
    Iteratively predicts mean utilities (delta) that rationalize market shares using a contraction mapping.

    The function updates the `delta` values until the predicted shares match the observed shares within a specified tolerance.

    Args:
        delta_initial (np.array): Initial guess for the mean utilities, a (T x J) matrix.
        gamma (np.array): A (K+1) x L matrix with coefficients of demographic variables.
        sigma (np.array): A (K+1) x (K+1) diagonal matrix representing the variance of unobserved taste interactions.
        D (np.array): A (K+1) x I matrix, where each row contains demographic information repeated across individuals.
        A_input (tuple): A tuple of three (T x J) matrices representing product characteristics (constant, sugar, price).
        T (int): Number of markets.
        J (int): Number of products in each market.
        I (int): Number of individuals (agents).
        tolerance (float, optional): The convergence criterion for the contraction mapping. Defaults to 10^-12.

    Returns:
        delta_prime (np.array): Updated mean utilities (T x J) that rationalize the shares.
        error (float): Maximum difference between consecutive guesses of delta (convergence error).
        i (int): Number of iterations until convergence is achieved.
        mean_predicted_shares (np.array): Mean predicted market shares (T x J) matrix.
        predicted_shares (np.array): Individual-specific predicted shares (I x T x J) matrix.
    """
    
    error = 2                                   # initial value of error
    delta = delta_initial.reshape((T,J))        # format the mean utilities
    
    # Convert the shares to the operable 3D array form
    log_shares = product_data["log_product_shares"].values.reshape((T,J))   # convert the shares to market by good matrix form
    lshare = np.broadcast_to(log_shares,(J,)+log_shares.shape).transpose()  # for each good broadcast the shares to 3D array of each good

    i = 0                                   # counter for loops
    while error > tolerance:
        i += 1
        
        delta_shaped = np.broadcast_to(delta, (I,) + delta.shape) # reshape the previous guess
        
        predicted_shares = predict_shares(gamma, sigma, D, delta_shaped, A_input, T, J, I) # get new predicted shares with previous delta guess
        mean_predicted_shares = predicted_shares[0] # assign mean estimated shares 
        delta_prime = delta + log_shares - np.log(mean_predicted_shares) # by proven contraction, update the delta prediction
        
        error = np.abs(delta_prime - delta).max() # get the supremum metric result of the caucy error
        delta = delta_prime # update the guess in case it fails      
        
    return delta_prime, error, i, predicted_shares[0], predicted_shares[1]














def gmm_objective(cons, delta, size):
    """
    Computes the GMM objective function, which is the value to be minimized in the estimation process.

    Args:
        cons (np.array): Linear parameters (coefficients) for the GMM model.
        delta (np.array): Mean utilities, a (T x J) matrix where T is the number of markets and J is the number of products.
        size (int): Total number of observations in the product data.

    Returns:
        float: The GMM objective function value to be minimized.
    """
    
    # Weighting matrix for gmm estimation, under homoskedasticity optimal weights are Z'Z which is what we are using
    W = np.linalg.inv(np.transpose(Z)@Z)
    
    # Calculate moments
    omega = delta - np.sum(A * cons, axis=1).reshape((T,J))
    omega = omega
    zeta_tj_1 = omega.reshape((size,1))
    moments = np.transpose(zeta_tj_1)@ Z
    
    
    return moments @ W @ np.transpose(moments)





def optimization(theta, D, A_input, T, J, I, S):
    """
    Optimizes the GMM objective function by computing the theta values and corresponding predicted delta values and passing it to the GMM objective.

    Args:
        theta (np.array): Parameter vector to be optimized, containing the coefficients for gamma and sigma.
        D (np.array): Demographic matrix, where each row contains demographic information repeated across individuals.
        A_input (tuple): Tuple of product characteristics (constant, sugar, price), each of shape (T x J).
        T (int): Number of markets.
        J (int): Number of products in each market.
        I (int): Number of individuals.
        S (int): Total number of observations in the product data.

    Returns:
        float: The GMM objective function value to be minimized.
    """
    
    gamma = np.zeros((K+1,L))
    sigma = np.zeros((K+1,K+1))
    cons = np.zeros(K+1)
    gamma[0] = theta[0]
    gamma[1] = theta[1]
    gamma[2] = theta[2]
    sigma[K,K] = theta[3]
    cons = theta[4],0,theta[5]
    
    # Initiate delta
    delta_initial = np.random.rand(S)
    
    predicted_delta = predict_delta(delta_initial, gamma, sigma, D, A_input, T, J, I)
    
    return gmm_objective(cons, predicted_delta[0], S)









gmm_results = minimize(optimization, theta, args=(D, A_input, T, J, I, S),method='L-BFGS-B',options={'disp': True})

gamma           = np.zeros((K+1,L))
sigma           = np.zeros((K+1,K+1))
cons            = np.zeros(K+1)
gamma[0]        = gmm_results.x[0]
gamma[1]        = gmm_results.x[1]
gamma[2]        = gmm_results.x[2]
sigma[K,K]      = gmm_results.x[3]
cons            = gmm_results.x[4],0,gmm_results.x[5]

mean_utilities  = predict_delta(np.random.rand(S), gamma, sigma, D, A_input, T, J, I)

    
# Calculate the elasticities
C01Q1_agent     = agent_data[agent_data['market_ids']=='C01Q1'].copy()

alpha_i1        = cons[2] + gamma[2] * np.array(C01Q1_agent['income']) + sigma[K,K] * np.array(C01Q1_agent['nodes0'])
alpha_i1.shape  = (I,1)
s_itj           = mean_utilities[4]
s_i1j           = s_itj[:,0,:]

elasticityq2_j1 =(np.array(C01Q1['prices']) / np.array(C01Q1['shares'])) * np.mean(alpha_i1 * s_i1j * (1 - s_i1j), axis = 0)


# Correlation
correlation_coefficient = np.corrcoef(np.array(C01Q1['prices']), elasticityq2_j1)[0, 1]
line = np.mean(elasticityq2_j1) + correlation_coefficient * (np.array(C01Q1['prices']) - np.mean(np.array(C01Q1['prices'])))


#Plotting the eleasticities
plt.scatter(np.array(C01Q1['prices']), elasticityq2_j1, alpha = 0.7)
plt.plot(np.array(C01Q1['prices']), line, color='red', label = f"slope = {correlation_coefficient:.2f})", linewidth=2)
plt.xlabel("Price")
plt.ylabel("Price Elasticity")
plt.title("BLP - Price Elasticities for market C01Q1")
plt.legend()  # Add the legend to display the correlation coefficient and slope
plt.show()



end=datetime.now()

print(end-start)
    
    
    



















def blp_summary(gmm_results, elasticities):
    """
    Generates a formatted summary of BLP model results based on the values from `gmm_results.x`.

    Args:
        gmm_results (OptimizeResult): The optimization result containing the estimated parameters in `gmm_results.x`.
        elasticities (np.array): Elasticities computed from the BLP model.
    
    Returns:
        str: A formatted string representing the BLP model results.
    """
    # Extract coefficients from gmm_results.x
    beta_0 = gmm_results.x[0]
    beta_sugar = gmm_results.x[1]
    alpha_income = gmm_results.x[2]
    alpha_nu = gmm_results.x[3]
    beta_0_income = gmm_results.x[4]
    alpha_0 = gmm_results.x[5]

    # Header for the summary
    summary_str = ""
    summary_str += "===================================================\n"
    summary_str += "                BLP Model Results                  \n"
    summary_str += "===================================================\n"
    
    # Add estimated coefficients for beta_0, beta_sugar, and their income interactions
    summary_str += "\nCoefficients (Beta_it):\n"
    summary_str += "---------------------------------------------------\n"
    summary_str += f"Beta_0 (Intercept): {beta_0: .4f}\n"  # Constant term (beta_0)
    summary_str += f"Beta_sugar: {beta_sugar: .4f}\n"  # Coefficient for sugar
    summary_str += f"Beta_0_income (Interaction with income): {beta_0_income: .4f}\n"  # Interaction of the constant term with income
    
    # Add price coefficients (Alpha_it)
    summary_str += "\nPrice Coefficients (Alpha_it):\n"
    summary_str += "---------------------------------------------------\n"
    summary_str += f"Alpha_0 (Price coefficient): {alpha_0: .4f}\n"  # Base price coefficient
    summary_str += f"Alpha_income (Interaction with income): {alpha_income: .4f}\n"  # Interaction with income for price
    summary_str += f"Alpha_nu (Random taste for price): {alpha_nu: .4f}\n"  # Random taste coefficient for price

    # Add price elasticities
    summary_str += "\nPrice Elasticities:\n"
    summary_str += "---------------------------------------------------\n"
    for i, elasticity in enumerate(elasticities):
        summary_str += f"Elasticity[{i}]: {elasticity: .4f}\n"

    summary_str += "===================================================\n"
    
    return summary_str






# Create the PDF
with PdfPages('model_results.pdf') as pdf:
    
    # 1. OLS Summary
    plt.figure(figsize=(8, 10))
    plt.text(0.01, 0.99, str(ols.summary()), {'fontsize': 8}, va="top", ha="left")
    plt.axis('off')  # Turn off the axis as it's not needed for the text display
    pdf.savefig()  # Save this page with OLS summary
    plt.close()

    # 2. TSLS Summary
    plt.figure(figsize=(8, 10))
    plt.text(0.01, 0.99, str(tsls.summary()).replace("OLS Regression Results", "TSLS Regression Results"), {'fontsize': 8}, va="top", ha="left")
    plt.axis('off')  # Turn off the axis for text display
    pdf.savefig()  # Save this page with TSLS summary
    plt.close()
    
    # 3. Plot the TSLS Price Elasticities
    plt.figure()
    plt.scatter(C01Q1["prices"], price_elasticity)
    plt.xlabel("Price")
    plt.ylabel("Price Elasticity")
    plt.title("2SLS - Price Elasticities for market C01Q1")
    pdf.savefig()  # Save this page with TSLS elasticity plot
    plt.close()
    
    # 4. BLP Model Results (gamma, sigma, cons)
    blp_summary_str = blp_summary(gmm_results, elasticityq2_j1)  # Use gmm_results and elasticityq2_j1
    plt.figure(figsize=(8, 10))
    plt.text(0.01, 0.99, blp_summary_str, {'fontsize': 8}, va="top", ha="left")
    plt.axis('off')  # Turn off the axis for text display
    pdf.savefig()  # Save this page with BLP summary
    plt.close()
    
    # 5. Plot the BLP Price Elasticities
    plt.figure()
    plt.scatter(np.array(C01Q1['prices']), elasticityq2_j1, alpha=0.7)
    plt.plot(np.array(C01Q1['prices']), line, color='red', label = f"slope = {correlation_coefficient:.2f})", linewidth=2)
    plt.xlabel("Price")
    plt.ylabel("Price Elasticity")
    plt.title("BLP - Price Elasticities for market C01Q1")
    plt.legend()  # Add the legend to display the correlation coefficient and slope
    pdf.savefig()  # Save this page with BLP elasticity plot
    plt.close()












"""
# Create the PDF
with PdfPages('model_results.pdf') as pdf:
    
    # 1. OLS Summary
    plt.figure(figsize=(8, 10))
    plt.text(0.01, 0.99, str(ols.summary()), {'fontsize': 8}, va="top", ha="left")
    plt.axis('off')  # Turn off the axis as it's not needed for the text display
    pdf.savefig()  # Save this page with OLS summary
    plt.close()

    # 2. TSLS Summary
    plt.figure(figsize=(8, 10))
    plt.text(0.01, 0.99, str(tsls.summary()).replace("OLS Regression Results", "TSLS Regression Results"), {'fontsize': 8}, va="top", ha="left")
    plt.axis('off')  # Turn off the axis for text display
    pdf.savefig()  # Save this page with TSLS summary
    plt.close()
    
    # 3. Plot the TSLS Price Elasticities
    plt.figure()
    plt.scatter(C01Q1["prices"], price_elasticity)
    plt.xlabel("Price")
    plt.ylabel("Price Elasticity")
    plt.title("2SLS - Price Elasticities for market C01Q1")
    pdf.savefig()  # Save this page with TSLS elasticity plot
    plt.close()
    
    # 4. BLP Model Results (gamma, sigma, cons)
    blp_summary_str = blp_summary(gamma, sigma, cons, elasticityq2_j1)
    plt.figure(figsize=(8, 10))
    plt.text(0.01, 0.99, blp_summary_str, {'fontsize': 8}, va="top", ha="left")
    plt.axis('off')  # Turn off the axis for text display
    pdf.savefig()  # Save this page with BLP summary
    plt.close()
    
    # 5. Plot the BLP Price Elasticities
    plt.figure()
    plt.scatter(np.array(C01Q1['prices']), elasticityq2_j1, alpha=0.7)
    plt.xlabel("Price")
    plt.ylabel("Price Elasticity")
    plt.title("BLP - Price Elasticities for market C01Q1")
    pdf.savefig()  # Save this page with BLP elasticity plot
    plt.close()

    


# Save OLS and TSLS summaries to text files
with open("ols_summary.txt", "w") as f:
    f.write(str(ols.summary()))

with open("tsls_summary.txt", "w") as f:
    f.write(str(tsls.summary()))

# Create a PDF instance
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Add OLS Summary to PDF
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, txt="OLS Summary")
with open("ols_summary.txt", "r") as f:
    for line in f:
        pdf.multi_cell(0, 10, txt=line)

# Add TSLS Summary to PDF
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, txt="TSLS Summary")
with open("tsls_summary.txt", "r") as f:
    for line in f:
        pdf.multi_cell(0, 10, txt=line)

# Plot the Price Elasticities for 2SLS
plt.scatter(C01Q1["prices"], price_elasticity)
plt.xlabel("Price")
plt.ylabel("Price Elasticity")
plt.title("2SLS - Price Elasticities for market C01Q1")
tsls_plot_path = "tsls_elasticities_plot.png"
plt.savefig(tsls_plot_path, dpi=400)
plt.close()  # Close the plot

# Add 2SLS Elasticities plot to PDF
pdf.add_page()
pdf.image(tsls_plot_path, x=10, y=30, w=pdf.w / 2)

# GMM Results (Optimization)
# Assume you've already run the optimization and obtained the values for gamma, sigma, etc.
gmm_results = minimize(optimization, theta, args=(D, A_input, T, J, I, S), method='L-BFGS-B', options={'disp': True})

gamma = np.zeros((K+1, L))
sigma = np.zeros((K+1, K+1))
cons = np.zeros(K+1)
gamma[0] = gmm_results.x[0]
gamma[1] = gmm_results.x[1]
gamma[2] = gmm_results.x[2]
sigma[K, K] = gmm_results.x[3]
cons = gmm_results.x[4], 0, gmm_results.x[5]

mean_utilities = predict_delta(np.random.rand(S), gamma, sigma, D, A_input, T, J, I)

# Plot BLP - Price Elasticities for market C01Q1
plt.scatter(np.array(C01Q1['prices']), elasticityq2_j1, alpha=0.7)
plt.xlabel("Price")
plt.ylabel("Price Elasticity")
plt.title("BLP - Price Elasticities for market C01Q1")
blp_plot_path = "blp_elasticities_plot.png"
plt.savefig(blp_plot_path, dpi=400)
plt.close()  # Close the plot

# Add BLP Elasticities plot to PDF
pdf.add_page()
pdf.image(blp_plot_path, x=10, y=30, w=pdf.w / 2)

# Save the PDF
pdf_output_path = "model_results.pdf"
pdf.output(pdf_output_path)

# Clean up: remove temporary plot files and text files
os.remove(tsls_plot_path)
os.remove(blp_plot_path)
os.remove("ols_summary.txt")
os.remove("tsls_summary.txt")
"""


    
    








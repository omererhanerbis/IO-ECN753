# This code is going to try to estimate Rust 1987 paper.
# We are required to estimate RC and theta_11
# Also, to compartmentalize, all functions are grouped and held in their respective operation files, are to be imported.

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from time import time
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add the rust_1987 directory to sys.path
module_path = os.path.join(os.getcwd(), 'rust_1987')
sys.path.append(module_path)
# Define the output PDF file path
pdf_file_name = os.path.join(module_path, "Rust_1987_Estimation_Results.pdf")

from config import beta, theta_3, N, tolerance, data, tex_content
from utility_functions import c, u, find_u
from probability_functions import g, find_P
from value_iteration import update, value_function_iteration
from maximum_likelihood import log_likelihood, objective_function
from results_table import rust_1987_summary




# Initial values for estimates (RC, theta_11)
theta_11    =   2.29
RC          =   10.7
estimates = (RC, theta_11)
# Create the whole parameter vector, (beta, theta_1, theta_3), beware the order
theta       =   (beta, theta_11, theta_3, RC)


# Start time
# start_time = time()

search = minimize(objective_function, estimates, args=(data, beta, theta_3, N, tolerance), method='L-BFGS-B')

# End time
# end_time = time()
# Calculate and print runtime
# runtime = end_time - start_time
# print(f"Optimization completed in {runtime:.2f} seconds")


RC2, theta_112=(search.x[0],search.x[1])
Theta2=(beta,theta_112,theta_3,RC2)
EV2,err2,U2=value_function_iteration(Theta2, N, tolerance)
loglike2=log_likelihood(EV2,U2,Theta2, data, N)

theta_estimates = {
    "beta": beta,
    "theta_11": theta_112,
    "theta_3": theta_3,
    "RC": RC2
}
loglike_value = loglike2  # Example log-likelihood value from the model


# Generate summary text for PDF
rust_summary_text = rust_1987_summary(theta_estimates, loglike_value)

# Enable LaTeX rendering in matplotlib for bold text and other LaTeX formatting
plt.rc('text', usetex=True)

# Combine summary text with TeX content
full_text = rust_summary_text + tex_content

# Create PDF with PdfPages and save it in the specified folder
with PdfPages(pdf_file_name) as pdf:
    # Create a page for the summary with TeX content
    plt.figure(figsize=(8, 10))
    plt.text(0.01, 0.99, full_text, {'fontsize': 10}, va="top", ha="left", wrap=True)
    plt.axis('off')  # Hide axes as we only need the text
    pdf.savefig()  # Save the current figure to the PDF as a page
    plt.close()

print(f"PDF generated successfully in: {pdf_file_name}")
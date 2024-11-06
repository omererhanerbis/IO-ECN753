# Function to create the Rust 1987 summary string
def rust_1987_summary(theta_estimates, loglike_value):
    summary_str = ""
    summary_str += "===================================================\n"
    summary_str += "                Rust 1987 Model Results            \n"
    summary_str += "===================================================\n"
    
    summary_str += "\nEstimated Parameters:\n"
    summary_str += "---------------------------------------------------\n"
    
    # Convert each entry to float, checking for tuple values
    beta_value = float(theta_estimates['beta'][0] if isinstance(theta_estimates['beta'], tuple) else theta_estimates['beta'])
    theta_11_value = float(theta_estimates['theta_11'][0] if isinstance(theta_estimates['theta_11'], tuple) else theta_estimates['theta_11'])
    theta_3_value = float(theta_estimates['theta_3'][0] if isinstance(theta_estimates['theta_3'], tuple) else theta_estimates['theta_3'])
    rc_value = float(theta_estimates['RC'][0] if isinstance(theta_estimates['RC'], tuple) else theta_estimates['RC'])

    # Format the summary with extracted values
    summary_str += f"Beta (Discount Factor): {beta_value: .4f}\n"
    summary_str += f"Theta_11: {theta_11_value: .4f}\n"
    summary_str += f"Theta_3: {theta_3_value: .4f}\n"
    summary_str += f"RC (Replacement Cost): {rc_value: .4f}\n"
    
    summary_str += "\nLog-Likelihood:\n"
    summary_str += "---------------------------------------------------\n"
    summary_str += f"Log-Likelihood: {float(loglike_value): .4f}\n"
    summary_str += "===================================================\n"
    
    return summary_str
# This code is going to try to estimate Rust 1987 paper.

# Prelimimary Step: We need calibrated beta, which is given to be beta = 0.999



# maintenance cost
def c(x, theta_11):
    
    maintenance_cost = theta_11 * x
    
    return maintenance_cost

# replacement cost





# flow utility
def u(x, i, theta):
    
    flow_utility = i * (-RC - c(0, theta[0])) + (1 - i) * (-c(x, theta[1]))
    
    return flow_utility




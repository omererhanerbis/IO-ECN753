import sys
import os
from matplotlib.backends.backend_pdf import PdfPages
# Add the rust_1987 directory to sys.path
module_path = os.path.join(os.path.dirname(__file__), 'rust_1987')
sys.path.append(module_path)
import pandas as pd

# Prelimimary Step: We need calibrated beta, which is given to be beta = 0.9999, and number of states is 90
# First stage is already done and the estimates are provided
# Use the given transition parameter estimates
N = 90
beta = 0.9999
theta_3 = (0.3919, 0.5953)
tolerance = 10**-4
data        =   pd.read_csv("rust_1987/group_4.csv")



# Add TeX-formatted content for part b
tex_content = r"""
\textbf{Part b)}

I am not sure how to integrate the 3D arrays to ease the calculation, so everything is a bit in the air.

For using CCPs, referring to Hotz, Miller, Sanders, Smith (1994), we can use the following logic:

\textbullet \ Estimate CCPs from the data for each state. This is done only once.
\textbullet \ Estimate transition probabilities e.g. AR(1).
\textbullet \ Given theta values recover the continuation values with the estimated CCPs. Then recover lifetime utility.
\textbullet \  Write the likelihood function again and run a maximum likelihood
estimation in a similar fashion over theta values.

So we would need to create a module for CCP calculation where:

\textbullet \ def empirical_CCPs(x, i) : P(i given x) = number of choice i given state y over all state y instances.

Then we would need a simulation module for continuation value calculation given the empirical CCPs.

\textbullet \ def EV_calculator(x, i) : EV_continuation = sum over x [ P(y given x, i) * V_hat(x) ]
where P can be calculated through x and AR(1 provess), V-hat being approximated value function
at state x. This estimation can be done through simulation and utilizing transition probabilities
and CCPs (i.e. simulate the choices and transition of states for enough of a lenght and 
given parameter values and parametric form, iteratively sum the flow utility to get continuation values).

Then we can calculate the total utility with estimated EVs and flow utilities in utility function script.
This step can actually be merged with previous one in one function.

Then we can calculate the implied CCPs given model specifications and previous utility calculations

Then we can calculate the log-likelihood function and get its value, similarly in the maximum likelihood module.

Then we can search over the theta, to minimize the -log likelihood value.


The advantages are that this method does not require value function iteration and fixed point existence.
However, this method produces simulation error through length of simulation and averaging over instances finitely.

"""
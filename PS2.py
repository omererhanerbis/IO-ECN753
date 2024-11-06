#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:48:08 2024

@author: elanieloiswatson
"""

import pandas as pd
import numpy as np
import scipy
import math
import statsmodels.api as sm
import statsmodels
from statsmodels.sandbox.regression.gmm import IV2SLS
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize
import scipy.stats as stats
from datetime import datetime
from scipy.stats import gaussian_kde

data = pd.read_csv("/Users/elanieloiswatson/Downloads/group_4.csv")
theta_30 = 0.3919
theta_31 = 0.5953
beta = 0.9999
x = data.mileage.to_numpy()
state = data.state.to_numpy()
decision = data.decision.to_numpy()

"define functions"
def expected_value_0 (RC,theta_11):
    global ev0
    global ev1
    tol_0 = np.ones(90)
    while(np.all(tol_0)>0.0001):
        ev_1f = expected_value_1(RC, theta_11)
        #ev_1f = 1
        c_10 = max(-RC + beta*ev_1f,np.max(-0.001*theta_11*y + beta*ev0,0))
        term_10 = theta_30 *(np.log(np.exp(-RC + beta*ev1-c_10) + np.exp(-0.001*theta_11*y + beta*ev0-c_10))+c_10)
        #term_10 = theta_30 * np.log(np.exp(-rep_cost + beta*ev_1f) + np.exp(-0.001*theta_11*y + beta*ev_0))
        c_20 = max(-RC + beta*ev1,np.max(-0.001*theta_11*(y[0:88]+1) + beta*ev0[y[0:88]+1],0))
        term_20[0:89] = theta_31 * (np.log(np.exp(-RC + beta*ev1 - c_20) + np.exp(-0.001*theta_11*(y[0:89]+1) + beta*ev0[y[0:89]+1] - c_20)) + c_20)
        term_20[89] = term_20[88]                       # adjusting the last values of second and third term  
        c_30 =  max(-RC + beta*ev1,np.max(-0.001*theta_11*(y[0:87]+2) + beta*ev0[y[0:87]+2],0))
        term_30[0:88] = (1-theta_30-theta_31) * ( np.log(np.exp(-RC + beta*ev1 - c_30) + np.exp(-0.001*theta_11*(y[0:88]+2) + beta*ev0[y[0:88]+2] - c_30)) + c_30 )
        term_30[88] = term_30[87]
        term_30[89] = term_30[87]
        ev0_update = term_10 + term_20 + term_30
        tol_0 = ev0_update - ev0
        ev0 = ev0_update
    return ev0_update



def  expected_value_1 (RC,theta_11):
    tol_1 = 1
    global ev1
    while (tol_1 > 0.0001):
        u11 = 0.001*theta_11*0
        u21 = 0.001*theta_11*1
        u31 = 0.001*theta_11*2
        c_11 = max(-RC + beta* ev1, -u11 + beta*ev0[0])
        term_11 = theta_30 * (np.log(np.exp(-RC + beta* ev1 - c_11) + np.exp(-u11 + beta*ev0[0] - c_11)) + c_11)
        c_21 = max(-RC + beta* ev1, -0.001*theta_11*1 + beta*ev0[1])
        term_21 = theta_31 * (np.log(np.exp(-RC + beta* ev1 - c_21) + np.exp(-u21 + beta*ev0[1] - c_21)) + c_21)
        c_31 = max(-RC + beta* ev1, -0.001*theta_11*2 + beta*ev0[2])
        term_31 = (1-theta_30-theta_31) * (np.log(np.exp(-RC + beta* ev1 - c_31) + np.exp(-u31 + beta*ev0[2] - c_31)) + c_31)
        ev1_update = term_11 + term_21 + term_31
        tol_1 = abs(ev1_update - ev1)
        ev1 = ev1_update
    return ev1_update

"define likelihood"
def likelihood(theta):
    RC = theta[0]
    theta_11 = theta[1]
    t1=np.zeros((4292,1))
    ev0_l = expected_value_0(RC, theta_11)
    ev1_l = ev1
    c_ = max (np.max(-0.001 * theta_11 * state + beta * ev0_l[state]), -RC + beta * ev1_l)
    den = np.exp( -0.001 * theta_11 * state + beta * ev0_l[state] - c_) + np.exp(-RC + beta * ev1_l - c_)
    num = np.zeros(len(decision))
    num[decision_1] =  np.exp(-RC + beta * ev1_l - c_)
    num[decision_0] =  np.exp( -0.001 * theta_11 * state + beta * ev0_l[state] - c_)
    num[0] =  np.exp( -0.001 * theta_11 * state[0]+ beta * ev0_l[state[0]] - c_)
    t1 = np.divide(num, den)
    log_like = -np.sum(np.log(t1))
    print(f'log likelihood function is: {log_like}')
    return log_like

theta_11_i = 2
RC_i = 10   
ev0 = np.ones(90)
ev1 = 1
ev1_update = 1
y = np.arange(90)
term_20 = np.zeros(90)
term_30 = np.zeros(90)
term_20_ = np.zeros(90)
term_30_ = np.zeros(90)
num_1 = 0
num_0 = 0
u = np.zeros(len(decision))
decision_0= np.arange(len(decision))
decision_1 =  np.arange(len(decision))
        
for k in  range (len(decision)):
    if (decision[k]==1):
        decision_1[k] = k
        decision_0[k] = 0
    else:
        decision_1[k] = 0
        decision_0[k] = k

startime = datetime.now()
theta = np.array([RC_i, theta_11_i])
res = minimize(likelihood,theta)
print(res.x)
print(datetime.now()-startime)


"using Kernel Density Estimation for CCP"
states = state.T
kde = stats.gaussian_kde(states, bw_method='scott')
x_values = np.linspace(min(states)-1, max(states)+1, 1000)
pdf_values = kde(x_values)
def choice_probability(states):
    return kde(states)
choice_probs = np.array([choice_probability(s) for s in states])
choice_probs /= choice_probs.sum()
def make_choice(state):
    prob_choice_1 = choice_probability(state)
    return np.random.choice([0, 1], p=[1 - prob_choice_1, prob_choice_1])



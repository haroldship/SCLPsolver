import pytest

import os, sys
import numpy as np

proj = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path.append(proj)

from SCLPsolver.SCLP import SCLP, SCLP_settings
from SCLPsolver.doe.data_generators.autoscaling import *

from SCLPsolver.doe.doe import *

seeds = [1234]

# Data

def test_autoscaling():

    TT = 10.0                     # time horizon
    I = 1                       # 1 server
    J = 2                       # 2 services
    alpha = np.array((100.0,100.0)) # initial buffer quantities
    a = np.array((40.0,20.0))       # request arrival rates for services requests per unit time
    mu = np.array((20.0,10.0))      # request processing rates for services requests per unit time
    c = np.array((1.0,1.0))         # holding cost of services per unit time
    gamma = np.array((-20.0,-20.0))     # replica/cpu cost per unit time
    b = np.array((4.0,))          # number of cpus per server
    q = np.array((3.0,6.0))         # max sojourn time per service

    G, H, F, gamma, c, d, alpha, a, b, T, total_buffer_cost, cost = generate_autoscaling_data(a, b, c, gamma, mu, alpha, q)

    solution, STEPCOUNT, param_line, res = SCLP(G, H, F, a, b, c, d, alpha, gamma, TT)
    t, x, q, u, p, pivots, SCLP_obj, err, NN, tau_intervals, maxT = solution.get_final_solution(True)

    print(t)
    print(u)

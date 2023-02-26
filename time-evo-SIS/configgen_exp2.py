# -*- coding: utf-8 -*-
"""
Generate configuration files
"""

from hgcm import *
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.special import loggamma

def poisson(xvec, xmean):
    return np.exp(xvec*np.log(xmean)-xmean-loggamma(xvec+1))



#structure - delta
mmin = 2
mmax = 2
nmax = 10
pn = np.zeros(nmax+1)
qm = np.zeros(mmax+1)
pn[nmax] += 1
qm[mmax] += 1 #delta


#rates
ymax = 500
rate = np.linspace(0.0001,1.,ymax+1)
mu = 0.06
nu = 1.

#rate distribution
dist = 'weibull'
py_n = weibull_rate_distribution(mu,nu,rate,ymax,nmax)
mean_rate = np.sum(py_n[:,0]*rate)
print(mean_rate)

plt.semilogy(rate,py_n[:,0])
plt.show()

#contagion
initial_density = 10**(-4)
# t = np.concatenate((np.arange(0,0.1,0.001),np.arange(0.1,1,0.1),np.arange(1,101,1)))
t = np.concatenate((np.linspace(0,100,201),[10**(-2),10**(-1)]))
# t = np.concatenate(([0],np.logspace(-2,2,201)))
t.sort()

#experience info
exp_id = 2
desc = "weak coupling, delta"

#==================
#DO NOT EDIT BELOW
#==================

config_file = f"./dat/conf_exp{exp_id}.json"
res_file = f"./dat/res_exp{exp_id}.pk"
sim_file = f"./dat/sim_exp{exp_id}.pk"

#save in json file
data = dict()

data['nmax'] = nmax
data['mmax'] = mmax
data['mmin'] = mmin
data['qm'] = qm.tolist()
data['pn'] = pn.tolist()

data['mu'] = mu
data['nu'] = nu
data['py_n'] = py_n.tolist()
data['ymax'] = ymax
data['rate'] = rate.tolist()
data['mean_rate'] = mean_rate

data['initial_density'] = initial_density
data['t'] = t.tolist()

data['res_file'] = res_file
data['sim_file'] = sim_file
data['desc'] = desc

with open(config_file, 'w') as file:
    json.dump(data, file)

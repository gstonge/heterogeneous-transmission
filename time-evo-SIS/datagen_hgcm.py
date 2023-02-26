import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import argparse
from hgcm import *
from hgcm.ode import hrate
from hgcm.ode import nlrate
from scipy.integrate import odeint

parser = argparse.ArgumentParser(description='Solutions using rate equations')
parser.add_argument('file',  type=str, help='Configuration file')
args = parser.parse_args()

#import parameters
#=================
config_file = args.file

with open(config_file, 'r') as file:
    config = json.load(file)

#unpack
#structure
nmax = config['nmax']
mmax = config['mmax']
mmin = config['mmin']
qm = np.array(config['qm'])
pn = np.array(config['pn'])

#rate
py_n = np.array(config['py_n'])
pyn = get_joint(py_n,pn)
ymax = config['ymax']
rate = np.array(config['rate'])
mean_rate = config['mean_rate']

#contagion
initial_density = config['initial_density']
t = np.array(config['t'])

#experience info
res_file = config['res_file']
desc = config['desc']

#to save the results, including config
results = {key:res for key,res in config.items()}

do_comp = True
if do_comp:
    #Complete dynamics
    #==================

    #initialize
    state_meta = hrate.get_state_meta(mmax, nmax, ymax, rate, qm, pyn)
    Im,Sm,Gyni = hrate.initialize(state_meta, initial_density=initial_density)
    v = np.concatenate((Im,Sm,Gyni.reshape((ymax+1)*(nmax+1)**2)))

    #integrate manually
    vvec_comp = odeint(hrate.vector_field,v,t,args=(state_meta,'SIS'), hmax=10**(-2))
    S_comp = np.array([np.sum(v[mmax+1:2*mmax+2]) for v in vvec_comp])
    I_comp = 1 - S_comp

    #get effective rate, inf_mat, and rho at each time
    eff_rate_list = []
    rho_list = []
    Gyi_list = []
    dGyi_list = []
    # print(hrate.unflatten(vvec_comp[-1],state_meta)[1])
    for v in vvec_comp:
        Im,Sm,Gyni = hrate.unflatten(v,state_meta)
        dv = hrate.vector_field(v,0,state_meta) #dummy time not used
        dIm,dSm,dGyni = hrate.unflatten(dv, state_meta)
        Gyi_list.append(np.array([Gyni[:,nmax,i] for i in range(nmax+1)]).T)
        dGyi_list.append(np.array([dGyni[:,nmax,i] for i in range(nmax+1)]).T)
        eff_rate_list.append(exact_effective_rate(rate,Gyni,nmax))
        rho_list.append(hrate.get_rho(Sm,Gyni,state_meta))

    results['Gyi_list'] = Gyi_list
    results['dGyi_list'] = dGyi_list
    results['S_comp'] = S_comp
    results['I_comp'] = I_comp
    results['eff_rate_list'] = eff_rate_list
    results['rho_list'] = rho_list

#critical effective rate
#===========================
eff_rate = critical_effective_rate(rate,py_n,ymax,nmax)
inf_mat = eff_rate*np.arange(nmax+1)
state_meta = nlrate.get_state_meta(mmax, nmax, qm, pn)

Im,Sm,Gni = nlrate.initialize(state_meta, initial_density=initial_density)
v = np.concatenate((Im,Sm,Gni.reshape((nmax+1)**2)))

#integrate manually
vvec_crit = odeint(nlrate.vector_field,v,t,args=(inf_mat,state_meta,'SIS'))
S_crit = np.array([np.sum(v[mmax+1:2*mmax+2]) for v in vvec_crit])
I_crit = np.array([np.sum(v[:mmax+1]) for v in vvec_crit])

#save
results['crit_eff_rate'] = eff_rate
results['S_crit'] = S_crit
results['I_crit'] = I_crit

# #quasistatic rate approximation
# #===================================
# #initialize
# state_meta = nlrate.get_state_meta(mmax, nmax, qm, pn)
# Im,Sm,Gni = nlrate.initialize(state_meta, initial_density=initial_density)

# v = np.concatenate((Im,Sm,Gni.reshape((nmax+1)**2)))

# #integrate manually
# vvec_qs = odeint(qs_vector_field,v,t,
                 # args=(state_meta,rate,pyn,ymax))
# S_qs = np.array([np.sum(v[mmax+1:2*mmax+2]) for v in vvec_qs])
# I_qs = np.array([np.sum(v[:mmax+1]) for v in vvec_qs])


# #save
# results['I_qs'] = I_qs
# results['S_qs'] = S_qs


#eigenvector effective rate
#==========================
excess_membership = excess_susceptible_membership(np.arange(mmax+1),qm)
vyni = get_leading_eigenvector(excess_membership,rate,pn,pyn,ymax,nmax, nb_iter=10000, verbose=True,
                                     alpha=0.005, model='SIS')
eff_rate = exact_effective_rate(rate,vyni,nmax)
print(eff_rate)
inf_mat = eff_rate*np.arange(nmax+1)
state_meta = nlrate.get_state_meta(mmax, nmax, qm, pn)

Im,Sm,Gni = nlrate.initialize(state_meta, initial_density=initial_density)
v = np.concatenate((Im,Sm,Gni.reshape((nmax+1)**2)))

#integrate manually
vvec_ev = odeint(nlrate.vector_field,v,t,args=(inf_mat,state_meta,'SIS'))
S_ev = np.array([np.sum(v[mmax+1:2*mmax+2]) for v in vvec_ev])
I_ev = np.array([np.sum(v[:mmax+1]) for v in vvec_ev])

#save
results['ev_eff_rate'] = eff_rate
results['S_ev'] = S_ev
results['I_ev'] = I_ev

#Mean rate approximation
#=======================

inf_mat = np.array([mean_rate*np.arange(nmax+1) for n in range(nmax+1)])
state_meta = nlrate.get_state_meta(mmax, nmax, qm, pn)

Im,Sm,Gni = nlrate.initialize(state_meta, initial_density=initial_density)
v = np.concatenate((Im,Sm,Gni.reshape((nmax+1)**2)))

#integrate manually
vvec_mean = odeint(nlrate.vector_field,v,t,args=(inf_mat,state_meta,'SIS'))
S_mean = np.array([np.sum(v[mmax+1:2*mmax+2]) for v in vvec_mean])
I_mean = np.array([np.sum(v[:mmax+1]) for v in vvec_mean])


#save
results['vvec_mean'] = vvec_mean
results['I_mean'] = I_mean
results['S_mean'] = S_mean

pickle.dump(results, open(res_file, "wb" ))

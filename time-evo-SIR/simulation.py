import numpy as np
import matplotlib.pyplot as plt
from _schon import ContinuousSIR
import horgg
import argparse
import json
import pickle

parser = argparse.ArgumentParser(description='Solution using simulations')
parser.add_argument('file',  type=str, help='Configuration file')
args = parser.parse_args()

#import parameters
#=================
config_file = args.file

with open(config_file, 'r') as file:
    config = json.load(file)

#=======
#unpack
#=======
#structure
nmax = config['nmax']
mmax = config['mmax']
mmin = config['mmin']
qm = np.array(config['qm'])
pn = np.array(config['pn'])

#rate
py_n = np.array(config['py_n'])
ymax = config['ymax']
rate = np.array(config['rate'])
mean_rate = config['mean_rate']

#contagion
initial_infected_fraction = config['initial_density']
# initial_infected_fraction = 0.0001

#===============
#random network
#===============

#generate sequences
# np.random.seed(42) #optional, if nothing is given, it is seeded with time
N = 200000 #number of groups
n_list = horgg.utility.sequence_1(N, pn)
m_list = horgg.utility.sequence_2(n_list, qm)

graph_generator = horgg.BCMS(m_list,n_list)
# horgg.BCMS.seed(42) #optional, if nothing is given, it is seeded with time

#mcmc steps are additional edge swaps to ensure uniformity, I recommend O(N)
edge_list = graph_generator.get_random_graph(nb_steps=N)

#===================
#group transmission
#===================
pdf = py_n[:,nmax]
cum = np.cumsum(pdf)
group_transmission_rate = []
for _ in range(N):
    r = np.random.random()
    ind = np.searchsorted(cum,r)
    group_transmission_rate.append(rate[ind])
print(np.mean(group_transmission_rate))
# plt.hist(group_transmission_rate, bins=40)
# plt.show()

#infection parameter
recovery_rate = 1.
infection_rate = np.zeros((nmax+1,nmax+1))
for n in range(2,nmax+1):
    for i in range(nmax+1):
        infection_rate[n][i] = i

sample_size = 100
Ilist = []
i = 0
while i < sample_size:
    cont = ContinuousSIR(edge_list,recovery_rate,infection_rate,
                            group_transmission_rate)
    cont.measure_prevalence()

    cont.infect_fraction(initial_infected_fraction)

    #evolve and measure
    dt = 100
    dec_dt = 1
    cont.evolve(dt,dec_dt,measure=True,quasistationary=False)

    #print the result measure
    for measure in cont.get_measure_vector():
        name = measure.get_name()
        if name == "prevalence":
            I = [initial_infected_fraction] + list(measure.get_result())

    if np.max(I) >= 10*initial_infected_fraction:
        print(f"sample {i+1}, final prevalence {I[-1]}")
        Ilist.append(I)
        i += 1

t = np.arange(0,dt,dec_dt)

for i in range(len(Ilist)):
    I = Ilist[i]
    Ilist[i] = np.pad(I,(0,dt-len(I)),'constant')

Imean = np.mean(Ilist,axis=0)
Istd = np.std(Ilist,axis=0)

plt.errorbar(t,Imean,yerr=Istd/np.sqrt(sample_size))
plt.yscale('log')
plt.show()

#save data
data = dict()
data['t'] = t
data['Ilist'] = Ilist

with open(config['sim_file'], 'wb') as file:
    pickle.dump(data,file)


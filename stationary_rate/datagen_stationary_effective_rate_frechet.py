from hgcm import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle

#color list
color_list = ["#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8", "#0c2c84"]
newcm = LinearSegmentedColormap.from_list('ColorMap',
                                          list(reversed(color_list[:-1])))

nmax = 20
ymax = 10000

#results
results = dict()

#weibull param
mu = 0.01
nu = 0.1
rate = np.linspace(0.01,0.1,ymax+1)

qy = frechet_rate_distribution(mu*np.ones(nmax+1),nu*np.ones(nmax+1),
                                  rate,ymax,nmax)

results['rate'] = rate
results['qy'] = qy

#plot distribution
width = 7.057/2
height = width/1.5
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(1,1,1)
plt.subplots_adjust(left=0.17, bottom=0.18, right=0.98,
                    top=0.98, wspace=0.3, hspace=0.25)


plt.loglog(rate,qy[:,-1], color = "#1d91c0")
plt.xlabel(r'Transmission rate $\lambda$')
plt.ylabel(r'Rate distribution')
plt.show()

#generate data
ivec = np.arange(nmax+1)
rho_vec = np.concatenate((np.logspace(-5,3,50), [0.1,1]))
rho_vec.sort()
effective_rate_vec = []
for j,rho in enumerate(rho_vec):
    effective_rate_vec.append(stationary_effective_rate(rate,rho,qy,ymax,nmax))

results['ivec'] = ivec[1:]
results['rho_vec'] = rho_vec
results['special_rho_list'] = [10**(-5),0.1,10**3]
results['eff_rate_vec'] = [effective_rate_vec[j][nmax][1:] for j in
                       range(len(rho_vec))]

#plot test
width = 7.057/2
height = width/1.5
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(1,1,1)
plt.subplots_adjust(left=0.17, bottom=0.18, right=0.98,
                    top=0.98, wspace=0.3, hspace=0.25)


for j,rho in enumerate(rho_vec):
    if rho in results['special_rho_list']:
        alpha = 1
        exponent = "{" + f"{int(np.log10(rho))}" + "}"
        label = fr"$\rho = 10^{exponent}$"
        zorder=2
        lw=1.5
    else:
        alpha = 0.2
        label = None
        zorder=0
        lw=0.7
    ax.plot(ivec[1:],results['eff_rate_vec'][j],
         color=newcm(j/len(rho_vec)),
         alpha=alpha,label=label,zorder=zorder,lw=lw)

ax.legend(frameon=False, ncol=1)
ax.set_ylabel(r"Effective rate")
ax.set_xlabel(r"Number of infected $i$")
plt.show()


with open('./dat/stationary_effective_rate_frechet.pk', 'wb') as outfile:
    pickle.dump(results,outfile)

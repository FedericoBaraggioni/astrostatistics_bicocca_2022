from astroML.datasets import fetch_dr7_quasar
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
import scipy.stats

N_bins=100

@jit
def get_bin(x):
	i=0
	while(i<N_bins):
		if(x<bin_edges[i]):
			break
		i+=1
	return i-1

#-----------------------------------------------------------------------------------#
#Load Data
#-----------------------------------------------------------------------------------#

# Fetch the quasar data
data = fetch_dr7_quasar()
# select the first 10000 points
data = data[:10000]
z = data['redshift']


hist, bin_edges=np.histogram(z, bins=N_bins, range=None, weights=None, density=True)
bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2 # mid location of bins


#-----------------------------------------------------------------------------------#
#Rejection Sampling
#-----------------------------------------------------------------------------------#

N=100000
good_x=[]
cont=0
#Distribution which captures the tails
x=np.random.random(N)*max(bin_edges)
#Uniform distribution from 0 to max(h(x)) 
y=np.random.random(N)*max(hist)

#if y<h(x) then accept
for j in range(N):
	if(y[j]<=hist[get_bin(x[j])]):
		good_x.append(x[j])

plt.title("Rejection Sampling")
plt.hist(z,density=True,histtype='step',lw=2,bins=N_bins)
plt.hist(good_x,density=True,histtype='step',lw=2, bins=N_bins)
plt.show()


#-----------------------------------------------------------------------------------#
#Inverse Trasformation Sampling
#-----------------------------------------------------------------------------------#

#I need the cumulative function and its inverse
H = np.cumsum(hist)/np.sum(hist)
inv_H=scipy.interpolate.interp1d(H, bin_mids)
#I generate a sample u from a uniform distribution from 0 and 1
u = np.random.uniform(0.001, 0.999, N) 
#I find the value of x below which a fraction u of the distribution is contained
x=inv_H(u)

plt.title("Inverse Transformation Sampling")
plt.hist(z,density=True,histtype='step',lw=2,bins=N_bins)
plt.hist(x,density=True,histtype='step',lw=2,bins=N_bins, );
plt.show()


#-----------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------#


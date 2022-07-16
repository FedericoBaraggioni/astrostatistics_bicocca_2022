import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from numba.typed import List

@jit
def trace_plot(N_sunny_days,w_list,today,i):
	if(today==0):
		prob= np.array([0.5,0.5])
	if(today ==1):
		prob= np.array([0.1,0.9])
	
	poss=np.array([0,1])
	today=poss[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")] #Random Choice for numba
	N_sunny_days+=today
	w_list.append(N_sunny_days/(i+1))
	return w_list,N_sunny_days,today


#-----------------------------------------------------------------------------------#
#Initialization
#-----------------------------------------------------------------------------------#
today=0
N=1000000
N_sunny_days=0
w=[0.]
w_list = List()
[w_list.append(e) for e in w]


#-----------------------------------------------------------------------------------#
#Markov Chain
#-----------------------------------------------------------------------------------#
for i in range(N):
	w_list,N_sunny_days,today=trace_plot(N_sunny_days,w_list,today,i)
	
w_list=np.array(w_list).astype(float)

x_i= np.linspace(0,N+1, N+1)
plt.plot(x_i,w_list)
plt.show()


#-----------------------------------------------------------------------------------#
#Histogram
#-----------------------------------------------------------------------------------#
plt.hist(w_list, bins=400)
plt.show()

burnin_index=100000
running_burn = w_list[burnin_index:]

plt.hist(running_burn, bins=200)
plt.show()
print(np.median(running_burn))

#-----------------------------------------------------------------------------------#
#
#-----------------------------------------------------------------------------------#


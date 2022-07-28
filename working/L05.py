import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from math import pi
from scipy.stats import norm

np.random.seed(18)
#-----------------------------------------------------------------------------------#
#Generate Fake Measurments
#-----------------------------------------------------------------------------------#
N=10
mu_sigma, sigma_sigma=0.2, 0.05
uncertainties = np.random.normal(mu_sigma, sigma_sigma, N)

mu = 1
data=[]
for sigma in uncertainties:
	data.append(np.random.normal(mu, sigma))
data=np.array(data)


#-----------------------------------------------------------------------------------#
#Build the Likelihood
#-----------------------------------------------------------------------------------#
gaussian=[]
xgrid = np.linspace(0,2,1000)

for i,pos in enumerate(data):
	gaussian.append(norm.pdf(xgrid,loc=pos,scale=uncertainties[i]))

L_prod=1

for gauss in gaussian:
	plt.plot(xgrid,gauss)
	a=gauss
	L_prod*=a #maybe try use np.log

plt.plot(xgrid,L_prod,ls='--')
plt.show()


#-----------------------------------------------------------------------------------#
#Parameter Estimation using ML
#-----------------------------------------------------------------------------------#
sorted_indices = np.argsort(L_prod)
index_max = sorted_indices[-1]
mu_meas=xgrid[index_max]
print("L is maximized at %.3f" % mu_meas)
#Analytical maximum likelihood estimator
print("Theoretical mu: ", round(np.sum(data/uncertainties**2)/np.sum(uncertainties**-2),3))


#-----------------------------------------------------------------------------------#
#Uncertainty on mu using Fisher Matrix
#-----------------------------------------------------------------------------------#
log_L=np.log(L_prod)
deriv=np.sqrt(-1*np.diff(log_L, n=2)/((xgrid[1]-xgrid[0])**2))

sigma_mu=1/deriv[0]
print("Uncertainty on mu: %.3f" % sigma_mu)
print('Theoretical sigma: ', round(np.sum(uncertainties**-2)**(-1./2),3))


#-----------------------------------------------------------------------------------#
#Compare Distributions
#-----------------------------------------------------------------------------------#

plt.plot(xgrid,L_prod,ls='--')

gaus=norm.pdf(xgrid,loc=mu_meas,scale=sigma_mu)

C=max(L_prod)/max(gaus)

plt.plot(xgrid,C*gaus)
plt.show()

#-----------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------#




#import
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
import scipy.optimize
import emcee
import scipy.stats
import corner

np.random.seed(42)

def sinefunc(t,A,T,phi):
    return A*np.sin(2*np.pi*t/T+phi)

def grafici(x,y,title):
	fig, axs = plt.subplots(3,figsize=(10,15))

	for ax,s in zip(axs,y.T):
		ax.plot(x,s)
		ax.set_xlabel('Time')
	axs[0].set_title(title)
	plt.show()

def LogLikelihood(theta, y, model=sinefunc):
	A,T,phi=theta
	x=np.linspace(0, 8, data.shape[0])
	sigma_y = np.ones(len(x))*0.01
	y_fit = model(x,A,T,phi)
	return -0.5 * np.sum((y-y_fit)**2 / sigma_y**2 ) 

def Logprior(theta):
	A,T,phi=theta 
	Amin=np.exp(-4)
	Amax=np.exp(1)
	bmin=2
	bmax=4
	phimin=-0.01
	phimax=5.3
	if Amin < A < Amax and bmin < T < bmax and phimin < phi < phimax:
		return - np.log(A) + 0.0 + 0.0 +0.
	return -np.inf

def LogPosterior(theta,data, model=sinefunc):
	lp=Logprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return LogLikelihood(theta,data, model) + lp

#-----------------------------------------------------------------------------------#
#Load Data
#-----------------------------------------------------------------------------------#
data=np.load('noisydetector.npy')
time = np.linspace(0, 8, data.shape[0])

grafici(time,data,'Outputs from three detectors')


#-----------------------------------------------------------------------------------#
#PCA
#-----------------------------------------------------------------------------------#

n_components = 3
pca = PCA(n_components)
a=pca.fit_transform(data)
grafici(time,a,'PCA components')

#-----------------------------------------------------------------------------------#
#ICA
#-----------------------------------------------------------------------------------#
ica = FastICA(n_components)
S_ = ica.fit_transform(data)  # Reconstruct signals
grafici(time,S_,'ICA components')


#-----------------------------------------------------------------------------------#
#FIT scipy
#-----------------------------------------------------------------------------------#
#Numero=input("Input ICA component to fit: ")
Numero=2

mysine = S_[:,int(Numero)]
plt.plot(time,mysine)
#plt.show()

paropt,pcov = scipy.optimize.curve_fit(sinefunc,time,mysine, p0=[0.04,4,0])
perr = np.sqrt(np.diag(pcov))
#plt.plot(time, mysine)
bestmodel = sinefunc(time,*paropt)

plt.plot(time, bestmodel)
plt.show()
print("Curve_fit")
print("Recovered time",round(paropt[1],3), "+-", round(perr[1],3))
print(paropt)


#-----------------------------------------------------------------------------------#
#FIT MCMC
#-----------------------------------------------------------------------------------#

A_quick=paropt[0]
T_quick=paropt[1]
phi_quick=paropt[1]
theta_quick=np.array([A_quick,T_quick,phi_quick])

ndim = 3  # number of parameters in the model
nwalkers = 16  # number of MCMC walkers
nsteps = int(1e4)

starting_guesses = theta_quick + 1e-5* np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, LogPosterior, args=[mysine, sinefunc])
sampler.run_mcmc(starting_guesses, nsteps,progress=True)

fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["A","T","phi"]
for i in range(ndim):
	ax = axes[i]
	ax.plot(samples[:, :, i], "k", alpha=0.3)
	ax.set_xlim(0, len(samples))
	ax.set_ylabel(labels[i])
	ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.show()

tau = sampler.get_autocorr_time()
#print(tau)
flat_samples = sampler.get_chain(discard=3*int(max(tau)), thin=int(max(tau)), flat=True)
#print(flat_samples.shape)
fig = corner.corner(
	flat_samples, labels=labels, levels=[0.68,0.95]
)
plt.show()

chosen_samples= flat_samples[np.random.choice(len(flat_samples),size=30)]
tgrid=np.linspace(0,10,100)

for chosen_theta in chosen_samples:
	ygrid =  sinefunc(tgrid,chosen_theta[0],chosen_theta[1],chosen_theta[2])
	plt.plot(tgrid,ygrid,alpha=0.3,c='gray')
	
plt.plot(time, mysine)
plt.xlabel("time")
plt.ylabel("flux")
plt.show()

for i,l in enumerate(labels):
	low,med, up = np.percentile(flat_samples[:,i],[5,50,95]) 
	print(l+"\t=\t"+str(round(med,3))+"\t+"+str(round(up-med,4))+"\t-"+str(round(med-low,4)))

#-----------------------------------------------------------------------------------#
#
#-----------------------------------------------------------------------------------#

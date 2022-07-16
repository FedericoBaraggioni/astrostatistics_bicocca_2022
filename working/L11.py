import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
import scipy
from scipy.stats import norm
from scipy.stats import cauchy

#----------------------------------------------------------------------------#
#Parameter Constraints
#----------------------------------------------------------------------------#
t0min,t0max = 0,100
Amin,Amax=0,50
bmin,bmax=0,50
alphamin,alphamax=np.exp(-5),np.exp(5)


#----------------------------------------------------------------------------#
#Function Definitions
#----------------------------------------------------------------------------#
def burst(theta,t):
    A,b,t0,alpha=theta 
    return np.where(t<t0,b,b+A*np.exp(-alpha*(t-t0)))

def LogLikelihood(theta, data, model=burst):
    x, y, sigma_y = data.T
    y_fit = model(theta, x)
    return -0.5 * np.sum((y-y_fit)**2 / sigma_y**2 ) 

def Logprior(theta):
    A,b,t0,alpha=theta 
    if Amin < A < Amax and bmin < b < bmax and t0min < t0 < t0max and alphamin < alpha < alphamax:
        return 0.0 + 0.0 + 0.0 - np.log(alpha)
    return -np.inf
                   
def LogPosterior(theta,data, model=burst):
    return LogLikelihood(theta,data, model) + Logprior(theta)


#----------------------------------------------------------------------------#
#Load Data
#----------------------------------------------------------------------------#
data=np.load("transient.npy")
t=data[:,0]
flux=data[:,1]
sigma=data[:,2]

plt.errorbar(t,flux,sigma,fmt='o')
plt.show()


#----------------------------------------------------------------------------#
#Initialize MCMC
#----------------------------------------------------------------------------#
ndim = 4  # number of parameters in the model
nwalkers = 20  # number of MCMC walkers
nsteps = 10000  # number of MCMC steps to take **for each walker**

tgrid=np.linspace(0,100,100)
t0_quick=50
A_quick=5
b_quick=10
alpha_quick=0.1

theta_quick= np.array([A_quick,b_quick,t0_quick,alpha_quick])
y_quick =  burst(theta_quick,tgrid)
plt.errorbar(t,flux,sigma,fmt='o')
plt.plot(tgrid,y_quick)
plt.title('Fit by Eye')
plt.show()


#----------------------------------------------------------------------------#
#Running MCMC
#----------------------------------------------------------------------------#
starting_guesses = theta_quick + 1e-1* np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, LogPosterior, args=[data, burst])
sampler.run_mcmc(starting_guesses, nsteps)

fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["A","b","t0","alpha"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

plt.show()


#----------------------------------------------------------------------------#
#Plot Fit Results
#----------------------------------------------------------------------------#
tau = sampler.get_autocorr_time()
print("Tau:",tau)

flat_samples = sampler.get_chain(discard=3*int(max(tau)), thin=int(max(tau)), flat=True)	#burn-in + thin
#print(flat_samples.shape)

fig = corner.corner(
    flat_samples, labels=labels, levels=[0.68,0.95]
)

plt.show()


#----------------------------------------------------------------------------#
#Plot Model
#----------------------------------------------------------------------------#
chosen_samples= flat_samples[np.random.choice(len(flat_samples),size=30)]

for chosen_theta in chosen_samples:

    ygrid =  burst(chosen_theta,tgrid)
    plt.plot(tgrid,ygrid,alpha=0.3,c='gray')
    
plt.errorbar(t,flux,yerr=sigma,fmt='o')
plt.xlabel("time")
plt.ylabel("flux")

plt.show()

for i,l in enumerate(labels):
    low,med, up = np.percentile(flat_samples[:,i],[5,50,95]) 
    print(l+"\t=\t"+str(round(med,2))+"\t+"+str(round(up-med,2))+"\t-"+str(round(med-low,2)))

#----------------------------------------------------------------------------#
#
#----------------------------------------------------------------------------#












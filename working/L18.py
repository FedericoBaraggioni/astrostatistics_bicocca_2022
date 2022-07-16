#import
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
from astroML.datasets import generate_mu_z
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from astroML.linear_model import LinearRegression
from astroML.linear_model import PolynomialRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import emcee
import scipy.stats as scistats
import corner
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
import math
#----------------------------------------------------------------------------#
#Function Definitions
#----------------------------------------------------------------------------#
def gpr_plot(x,mu,err):
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot()
	ax.plot(x, mu, '-', color='gray')
	ax.fill_between(x, mu - err, mu + err, alpha = 0.5, label = r'$1\sigma$')
	ax.fill_between(x, mu - 2*err, mu + 2*err, alpha = 0.5, label = r'$2\sigma$')
	ax.errorbar(z_sample, mu_sample, dmu, fmt='.k', ms=6)
	ax.legend()
	plt.show()

def Gauss_process(z,mu_sample,dmu,Const_var,Rbf_par):
	kernel1 =ConstantKernel(Const_var, (1e-3, 1e3)) * RBF(Rbf_par, (0.01, 100)) #GPR already tweaks hyperparameter, cv not necessary in most cases (https://stats.stackexchange.com/questions/182535/calibrating-a-gaussian-process?rq=1)
	gp = GaussianProcessRegressor(kernel=kernel1,alpha=(dmu)**2)
	gp.fit(z,mu_sample)
	return gp

def generativemodel(z, H0, Om=1):
	H0=np.asscalar(H0)
	cosmo_tmp = LambdaCDM(H0=H0, Om0=Om, Ode0=1-Om)
	return cosmo_tmp.distmod(z).value

def LogLikelihood(theta,model=0):
	if model == 0:
		Om,H0 = theta    
		if Om<0:
		    return -np.inf
		else:
		    mu_model = generativemodel(z_sample, H0, Om)
		
		return -0.5 * np.sum((mu_sample-mu_model)**2 / dmu**2 )
	
	if model == 1:
		H0 = theta    
		mu_model = generativemodel(z_sample, H0)

		return -0.5 * np.sum((mu_sample-mu_model)**2 / dmu**2 )

def ptform(u,model):
	x = np.array(u)  # copy u
	if model == 0:
		x[0] = scistats.uniform(loc=0.1,scale=1-0.1).ppf(u[0])
		x[1] = scistats.uniform(loc=50,scale=100).ppf(u[1])
	if model == 1:
		x[0] = scistats.uniform(loc=50,scale=100).ppf(u[0])
	return x


def Logprior(theta):
    Om,H0 = theta
    if 50 < H0 < 100 and 0.05 < Om < 1:
        return 0.0
    return -np.inf

def LogPosterior(theta):
    return LogLikelihood(theta) + Logprior(theta)

def plot(samples_equal,title):
	tgrid=np.linspace(0,2.5,100)
	chosen_samples= samples_equal[np.random.choice(len(samples_equal),size=30)]

	for chosen_theta in chosen_samples:
		if(title == 'Dark Energy'):
			ygrid =  generativemodel(tgrid,chosen_theta[1],chosen_theta[0])
		else:
			ygrid =  generativemodel(tgrid,chosen_theta[0])
		plt.plot(tgrid,ygrid,alpha=0.3,c='gray')
		
	plt.errorbar(z_sample,mu_sample,yerr=dmu,fmt='o')
	plt.xlabel("z")
	plt.ylabel("mu")
	plt.title(title)
	plt.show()
#----------------------------------------------------------------------------#
#Load data
#----------------------------------------------------------------------------#
n=100
z_sample, mu_sample, dmu = generate_mu_z(n, random_state=1234)
z_sample1=z_sample[:,np.newaxis]
x = np.linspace(0, 2.5, n)


#----------------------------------------------------------------------------#
#GPR
#----------------------------------------------------------------------------#

gp=Gauss_process(z_sample1,mu_sample,dmu,1,10)
mu, err = gp.predict(x[:,None], return_std=True)
print(gp.kernel_.get_params())
gpr_plot(x,mu,err)


#----------------------------------------------------------------------------#
#MCMC
#----------------------------------------------------------------------------#

ndim = 2  # number of parameters in the model
nwalkers = 5  # number of MCMC walkers
nsteps = int(1e4)  # number of MCMC steps to take **for each walker**

starting_guesses = np.array([0.5,80]) + 1e-1* np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, LogPosterior)
sampler.run_mcmc(starting_guesses, nsteps,progress=True)
samples = sampler.get_chain()

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = ["Om","H0"]
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

fig = corner.corner(
    flat_samples, labels=["Om","H0"], levels=[0.68,0.95], show_titles=True
)

plt.show()

zlin = np.linspace(0,2,1000)[1:]
for Om,H0, in flat_samples[::1000]:
    plt.plot(zlin, generativemodel(zlin, H0, Om),c='C3',alpha=0.2)
    
plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.xlim(0,2)
plt.ylim(35,50)

plt.show()


#----------------------------------------------------------------------------#
#Dynesty
#----------------------------------------------------------------------------#
def Dinesty_Fit(model):
	ndim = 2-model

	sampler = dynesty.NestedSampler(LogLikelihood, ptform, ndim,logl_args=[model],ptform_args=[model],nlive=100)
	sampler.run_nested()
	sresults = sampler.results

	rfig, raxes = dyplot.runplot(sresults)
	plt.show()

	tfig, taxes = dyplot.traceplot(sresults)
	plt.show()
	
	samples = sresults.samples  # samples
	weights = np.exp(sresults.logwt - sresults.logz[-1])  # normalized weights
	
	if model == 0:
		labels = ["Om","H0"]
	if model == 1:
		labels = ["H0"]

	samples_equal = dyfunc.resample_equal(samples, weights)
	corner.corner(samples_equal,labels=labels);

	plt.show()

	quantiles = [dyfunc.quantile(samps, [0.05, 0.5, 0.95], weights=weights)
		         for samps in samples.T]
	for q,l in zip(quantiles,labels):
		low,med,up=q
		print(l+"   "+str(med)+" +"+str(up-med)+" -"+str(med-low))


	sresults.summary()
	
	return samples_equal, np.exp(sresults.logz[-1])

samples_equal,B1=Dinesty_Fit(0)
#----------------------------------------------------------------------------#
#Dynesty NO Dark Energy
#----------------------------------------------------------------------------#
samples_equal2,B2=Dinesty_Fit(1)


print("Bayes Factor:")
B=B1/B2
print(B)

if(B1>B2):
	pref_model = "Dark Energy"
else:
	pref_model = "No Dark Energy"
	B=-1*B

class_list=["No","Substantial","Strong","Very Strong","Decisive"]
index=math.floor(np.log10(B)*2)
classific=class_list[index]

print("Based on the Jefferys's scale there is ",classific, "evidence in favor of ", pref_model)

#----------------------------------------------------------------------------#
#Plot Model
#----------------------------------------------------------------------------#
plot(samples_equal,"Dark Energy")
plot(samples_equal2,"No Dark Energy")

#----------------------------------------------------------------------------#
#Data Cloning
#----------------------------------------------------------------------------#

#I used a KDE to sample the z distribution

plt.hist(z_sample1,bins=20,density=True)
xgrid = np.linspace(z_sample.min(),z_sample.max(),int(n))
band_list=np.linspace(0.001,10,1000)
kde=KernelDensity(kernel="gaussian")

#Cross validation for the bandwidth

grid = GridSearchCV(kde, param_grid={'bandwidth': band_list}, cv=3 , verbose=0,n_jobs=-1)
grid.fit(z_sample1)
best_kde = grid.best_estimator_

#Plot
best_kde.fit(z_sample1)
log_pdf = best_kde.score_samples(xgrid[:, np.newaxis]) # sore_samples returns log(density)
KDE=np.exp(log_pdf)

plt.plot(xgrid,KDE,label='Gaussian Kernel')
plt.legend()
plt.show()

#Sample -> Cloning
z_vals=best_kde.sample(n*10)
print(max(z_vals))
print(len(z_vals),n)
mu_gpr=[]
for z in z_vals:
    mu_fit, sigma = gp.predict(z[:,np.newaxis], return_std=True)
    mu_gpr.append(np.random.normal(loc=mu_fit,scale=sigma))

plt.scatter(z_vals,mu_gpr,alpha=0.2,label='GPR')

plt.xlabel("z")
plt.ylabel("$\mu$")
plt.xlim(0,2)
plt.ylim(35,50)
plt.title("Cloned data")
plt.legend()
plt.show()

#----------------------------------------------------------------------------#
#
#----------------------------------------------------------------------------#

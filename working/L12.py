import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
import dynesty
import scipy.stats
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
import math
from astroML.utils import pickle_results

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

np.random.seed(18)

#----------------------------------------------------------------------------#
#Parameter Constraints
#----------------------------------------------------------------------------#
t0min,t0max = 0,100
Amin,Amax=0,50
bmin,bmax=0,50
alphamin,alphamax=np.exp(-5),np.exp(5)
sigmaWmin,sigmaWmax=np.exp(-2),np.exp(2)

#----------------------------------------------------------------------------#
#Function Definitions
#----------------------------------------------------------------------------#
def burst(theta,t):
    A,b,t0,alpha=theta 
    return np.where(t<t0,b,b+A*np.exp(-alpha*(t-t0)))
    
def gprofile(theta,t):
	A,b,t0,sigma_w=theta
	return b+A*np.exp(-(t-t0)**2/(2*sigma_w**2))
	
def loglike(theta, data, model):
    x, y, sigma_y = data.T
    if model =='burst':
        y_fit = burst(theta, x)
    elif model == 'gprofile':
        y_fit = gprofile(theta, x)
    return -0.5 * np.sum((y-y_fit)**2 / sigma_y**2 )

def ptform(u,model):
    x = np.array(u)  # copy u
    x[0] = scipy.stats.uniform(loc=Amin,scale=Amax-Amin).ppf(u[0])
    x[1] = scipy.stats.uniform(loc=bmin,scale=bmax-bmin).ppf(u[1])
    x[2] = scipy.stats.uniform(loc=t0min,scale=t0max-t0min).ppf(u[2])
   
    if model =='burst':
        x[3] = scipy.stats.loguniform.ppf(u[3],alphamin,alphamax)
    elif model =='gprofile':
        x[3] = scipy.stats.loguniform.ppf(u[3],sigmaWmin,sigmaWmax)
    return x

def plot(samples_equal,func,title):
	tgrid=np.linspace(0,100,100)
	chosen_samples= samples_equal[np.random.choice(len(samples_equal),size=30)]

	for chosen_theta in chosen_samples:

		ygrid =  func(chosen_theta,tgrid)
		plt.plot(tgrid,ygrid,alpha=0.3,c='gray')
		
	plt.errorbar(t,y,yerr=yerr,fmt='o')
	plt.xlabel("time")
	plt.ylabel("flux")
	plt.title(title)
	plt.show()


def Dinesty_Fit(model):
	ndim = 4
	sampler = dynesty.NestedSampler(loglike, ptform, ndim,logl_args=[data,model],ptform_args=[model],nlive=50)
	sampler.run_nested()
	return sampler.results
	
def Dynesty_Res(sresults):
	rfig, raxes = dyplot.runplot(sresults)
	plt.show()

	tfig, taxes = dyplot.traceplot(sresults)
	plt.show()
	
	samples = sresults.samples  # samples
	weights = np.exp(sresults.logwt - sresults.logz[-1])  # normalized weights

	labels = ["A","b","t0","alpha"]

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
		

#----------------------------------------------------------------------------#
#Load Data
#----------------------------------------------------------------------------#
data= np.load('transient.npy')
t,y,yerr=data.T

#----------------------------------------------------------------------------#
#Run MCMC
#----------------------------------------------------------------------------#
res=Dinesty_Fit('burst')
res2=Dinesty_Fit('gprofile')

print("\n\n-------------")
print("Burst:")
samples_equal,B1=Dynesty_Res(res)
print("\n\n-------------")
print("Gaussian")
samples_equal2,B2=Dynesty_Res(res2)


#----------------------------------------------------------------------------#
#Plot Model
#----------------------------------------------------------------------------#
plot(samples_equal,burst,"Burst")
plot(samples_equal2,gprofile,"Gaussian")

#----------------------------------------------------------------------------#
#Model Comparison
#----------------------------------------------------------------------------#
print("Bayes Factor:")
B=B1/B2
print(B)

if(B1>B2):
	pref_model = "Burst Model"
else:
	pref_model = "Gaussian Model"
	B=-1*B

class_list=["No","Substantial","Strong","Very Strong","Decisive"]
index=math.floor(np.log10(B)*2)
classific=class_list[index]

print("Based on the Jefferys's scale there is ",classific, "evidence in favor of ", pref_model)








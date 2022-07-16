import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

#-----------------------------------------------------------------------------------#
#Load data
#-----------------------------------------------------------------------------------#
a=np.load("formationchannels.npy")
N_bins=100
#plt.hist(a,density=True,histtype='step',lw=2,bins=N_bins)
#plt.show()


#-----------------------------------------------------------------------------------#
#Test many Gaussian Mixture
#-----------------------------------------------------------------------------------#

aic_l=[]
for j in np.arange(10)+1:
	gmm = GaussianMixture(n_components=j,n_init=10)
	gmm.fit(a)
	aic_l.append(gmm.aic(a))

#Seek the minimum AIC value
plt.scatter(x=np.arange(len(aic_l))+1,y=aic_l)
plt.show()
minim=np.argmin(aic_l)+1 
print("AIC is minimum for ",minim,"Gaussians")


#-----------------------------------------------------------------------------------#
#Plot
#-----------------------------------------------------------------------------------#

print("-------------------")
gmm = GaussianMixture(n_components=minim,n_init=10)
gmm.fit(a)
print("Gaussian Mixture Results:")
print(gmm.means_)
print(np.sqrt(gmm.covariances_))

x = np.linspace(0, 60, 1000)

logprob = gmm.score_samples(x[:,np.newaxis])
pdf = np.exp(logprob)
plt.plot(x, pdf, '--k')
plt.hist(a,density=True,histtype='step',lw=2,bins=N_bins)
plt.show()

#-----------------------------------------------------------------------------------#
#Plot Individual Gaussian
#-----------------------------------------------------------------------------------#

responsibilities = gmm.predict_proba(x[:,np.newaxis])
pdf_individual = responsibilities * pdf[:, np.newaxis]

plt.hist(a, N_bins, density=True, histtype='step', alpha=0.4,color='black')

for i in range(minim):
    plt.plot(x, pdf_individual[:,i], c='C0')

plt.plot(x, pdf, '--k')
plt.show()

#-----------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------#


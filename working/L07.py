import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization.hist import hist as fancyhist
from scipy import stats
from sklearn.neighbors import KernelDensity
import scipy

def kde_sklearn(data, bandwidth = 1.0, kernel="linear"):
    kde_skl = KernelDensity(bandwidth = bandwidth, kernel=kernel)
    kde_skl.fit(data[:, np.newaxis])
    log_pdf = kde_skl.score_samples(xgrid[:, np.newaxis]) # sore_samples returns log(density)
    return np.exp(log_pdf)

def M_irr(M,x):
	return M*f(x)
	
def f(x):
	return ((1+(1-x**2)**0.5)/2)**0.5

#-----------------------------------------------------------------------------------#
#Fake Measurements Generetion
#-----------------------------------------------------------------------------------#
N=10000
mu=1
sigma=0.02

x=np.random.uniform(0.,1.,N)
M = np.random.normal(mu, sigma, N)

M_i=M_irr(M,x)

xgrid = np.linspace(M_i.min(),M_i.max(),1000)

plt.hist(M_i,bins=50, density=True,histtype='step',lw=2,label='Histogram')
_ = fancyhist(M_i, bins="scott", histtype="step",density=True,label="Scott's rule")

#-----------------------------------------------------------------------------------#
#Kernel Density Estimation
#-----------------------------------------------------------------------------------#

KDE = kde_sklearn(M_i,bandwidth=0.009,kernel="gaussian") #Complete
plt.plot(xgrid,KDE,label='Gaussian Kernel')

plt.legend()
plt.show()


#-----------------------------------------------------------------------------------#
#Kolmogorv-Smirnov Test (as a function of sigma)
#-----------------------------------------------------------------------------------#

print("KS test (M_irr, f || sigma=%.4f):"%sigma,stats.ks_2samp(M_i,f(x) ))
ksf = []
ksm = []
scales= np.logspace(-5,5,20)
for sigma in scales:
	M = np.random.normal(mu, sigma, N)
	M_i=M_irr(M,x)
	ksf.append(stats.ks_2samp(M_i,f(x) ))
	ksm.append(stats.ks_2samp(M_i,M ))
	
ksf= np.array(ksf)
ksm= np.array(ksm)

plt.plot(scales,ksf[:,0],label="KS$(M_{\\rm irr}, f)$")
plt.plot(scales,ksm[:,0],label="KS$(M_{\\rm irr}, M)$")
plt.semilogx()
plt.xlabel("$\sigma$")
plt.ylabel('KS statistics')
plt.legend()
plt.show()


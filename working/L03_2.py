import numpy as np
import pylab as plt
from numba import jit
import time
import scipy.stats

@jit
def integral(sigma,N):
	mu=0
	data = np.abs(np.random.normal(mu, sigma, N))
	return sigma*np.sqrt(np.pi/2)*np.sum(data**3)/(N)

def main(N):
	sigma = 1.5
	integrale=np.zeros(N)
	for j in range(N):
		integrale[j]=integral(sigma,N)


	mean = np.mean(integrale)
	std = np.std(integrale, ddof=1)
	gauss = scipy.stats.norm(loc=mean,scale=std)
	x = np.linspace(8,12,1000)

	plt.axvline(2*sigma**4,c='red',ls='dotted')
	plt.hist(integrale,density=True,bins=int(N/50))
	plt.plot(x,gauss.pdf(x))
	plt.xlim(8,12)
	plt.ylim(0,2)
	plt.show()

main(1000)
main(5000)
main(10000)
